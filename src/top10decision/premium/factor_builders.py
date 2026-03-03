#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium — Factor Packs Builders

职责：
- 根据 packs_used 构建特征 DataFrame（ts_code 粒度）
- Pack0：从 pred_source + market OHLCV（近N日）构建动量/波动/影线/量能比等
- Pack1：从 data/market_basic/daily_basic_YYYYMMDD.csv 读取流动性/市值硬因子
- Pack2：从 data/limit/limit_micro_YYYYMMDD.csv 读取涨停结构微观因子

要求：
- 缺失不报错：返回空表或填充缺失标记，由主流程合并后兜底
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PremiumConfig
from .market_truth import load_daily


def _to_yyyymmdd(x: object) -> str:
    s = str(x).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "")
    return s[:8]


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    last = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
            continue
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _list_market_dates(cfg: PremiumConfig) -> List[str]:
    # 通过文件名列出 data/market/daily_YYYYMMDD.csv 的日期（不依赖交易所日历）
    root = cfg.market_daily_dir()
    if not root.exists():
        return []
    out = []
    for p in root.glob("daily_*.csv"):
        s = p.stem.replace("daily_", "")
        s = _to_yyyymmdd(s)
        if len(s) == 8 and s.isdigit():
            out.append(s)
    return sorted(set(out))


def _pick_recent_dates(all_dates: List[str], trade_date: str, need: int) -> List[str]:
    td = _to_yyyymmdd(trade_date)
    d = [x for x in all_dates if x <= td]
    if not d:
        return []
    return d[-need:]


def _safe_ln_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(a / b)
    return pd.Series(r, index=a.index)


def build_pack0_base(cfg: PremiumConfig, trade_date: str, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    返回：ts_code 粒度特征表（Pack0）
    pred_df：主流程已读入并 normalize 的 pred_source_latest（至少含 ts_code/name/prob/StrengthScore/ThemeBoost/board）
    """
    td = _to_yyyymmdd(trade_date)

    # 先验因子（来自 pred_source）
    base = pred_df[["ts_code"]].copy()
    for c in ("prob", "StrengthScore", "ThemeBoost", "rank", "board"):
        if c in pred_df.columns:
            base[c] = pred_df[c]
        else:
            base[c] = pd.NA

    base.rename(
        columns={
            "prob": "f_prob",
            "StrengthScore": "f_strength",
            "ThemeBoost": "f_theme",
            "rank": "f_rank",
            "board": "f_board",
        },
        inplace=True,
    )

    # market OHLCV 衍生
    all_dates = _list_market_dates(cfg)
    # 至少需要 11 天：用于 vol_10d 与 ret_5d 等（不足则自动缩窗）
    want = int(getattr(cfg, "pack0_window_days", 20))
    dates = _pick_recent_dates(all_dates, td, max(2, want))
    if not dates or dates[-1] != td:
        # 没有当日文件也不报错（由主流程 pending close_T）
        # 返回只有先验因子
        return base

    # 组装最近窗口行情（只取必要列）
    frames = []
    for d in dates:
        try:
            dd = load_daily(cfg, d)
            if dd is None or dd.empty:
                continue
            cols = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"] if c in dd.columns]
            dd = dd[cols].copy()
            dd["trade_date"] = dd["trade_date"].astype(str).map(_to_yyyymmdd)
            frames.append(dd)
        except Exception:
            continue

    if not frames:
        return base

    mkt = pd.concat(frames, ignore_index=True)
    mkt["ts_code"] = mkt["ts_code"].astype(str).str.strip()

    # 取当日行（T）
    trow = mkt[mkt["trade_date"] == td].copy()
    if trow.empty:
        return base

    # close 序列（按 ts_code, trade_date 排序）
    mkt = mkt.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # ret_1d/3d/5d：基于 close
    # 用 groupby shift 实现；窗口不足会产生 NaN，后面由主流程兜底
    mkt["close"] = pd.to_numeric(mkt.get("close"), errors="coerce")
    for k in (1, 3, 5):
        mkt[f"_close_lag{k}"] = mkt.groupby("ts_code")["close"].shift(k)
        mkt[f"ret_{k}d"] = _safe_ln_ratio(mkt["close"], mkt[f"_close_lag{k}"])

    # vol_5d / vol_10d：ret_1d rolling std
    mkt["ret_1d"] = pd.to_numeric(mkt["ret_1d"], errors="coerce")
    mkt["vol_5d"] = (
        mkt.groupby("ts_code")["ret_1d"].rolling(window=5, min_periods=2).std().reset_index(level=0, drop=True)
    )
    mkt["vol_10d"] = (
        mkt.groupby("ts_code")["ret_1d"].rolling(window=10, min_periods=3).std().reset_index(level=0, drop=True)
    )

    # range_1d / upper/lower shadow：用当日 OHLC
    for c in ("open", "high", "low", "close"):
        if c in mkt.columns:
            mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    mkt["range_1d"] = (mkt["high"] - mkt["low"]) / mkt["close"]
    mkt["upper_shadow_1d"] = (mkt["high"] - np.maximum(mkt["open"], mkt["close"])) / mkt["close"]
    mkt["lower_shadow_1d"] = (np.minimum(mkt["open"], mkt["close"]) - mkt["low"]) / mkt["close"]

    # range_5d：近5日 max(high)/min(low)-1
    mkt["range_5d"] = (
        mkt.groupby("ts_code")
        .apply(lambda g: (g["high"].rolling(5, min_periods=2).max() / g["low"].rolling(5, min_periods=2).min() - 1))
        .reset_index(level=0, drop=True)
    )

    # amount_z_5d / vol_z_5d
    for c in ("amount", "vol"):
        if c in mkt.columns:
            mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    mkt["amount_z_5d"] = mkt["amount"] / (mkt.groupby("ts_code")["amount"].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True))
    mkt["vol_z_5d"] = mkt["vol"] / (mkt.groupby("ts_code")["vol"].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True))

    # 只取当日（T）的特征
    feat_cols = [
        "ts_code",
        "ret_1d", "ret_3d", "ret_5d",
        "vol_5d", "vol_10d",
        "range_1d", "upper_shadow_1d", "lower_shadow_1d", "range_5d",
        "amount_z_5d", "vol_z_5d",
    ]
    feats = mkt[mkt["trade_date"] == td][feat_cols].copy()

    # 合并先验 + market 特征
    out = base.merge(feats, on="ts_code", how="left")
    return out


def build_pack1_tushare_basic(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    p = cfg.market_basic_path(trade_date)
    df = _safe_read_csv(p)
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code"])
    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).map(_to_yyyymmdd)
    if "ts_code" not in df.columns:
        return pd.DataFrame(columns=["ts_code"])
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    # 统一字段名
    out = pd.DataFrame({"ts_code": df["ts_code"]})
    for c in ("turnover_rate", "circ_mv", "total_mv", "volume_ratio"):
        out[c] = df[c] if c in df.columns else pd.NA
    return out


def build_pack2_limit_micro(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    p = cfg.limit_micro_path(trade_date)
    df = _safe_read_csv(p)
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code"])
    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).map(_to_yyyymmdd)
    if "ts_code" not in df.columns:
        return pd.DataFrame(columns=["ts_code"])
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    out = pd.DataFrame({"ts_code": df["ts_code"]})
    for c in ("open_times", "seal_amount", "first_limit_time", "last_limit_time"):
        out[c] = df[c] if c in df.columns else pd.NA
    return out


def build_features_by_packs(cfg: PremiumConfig, trade_date: str, pred_df: pd.DataFrame, packs_used: List[str]) -> pd.DataFrame:
    """
    输入：trade_date, pred_df, packs_used
    输出：按 ts_code 合并后的总特征表
    """
    feats = pd.DataFrame({"ts_code": pred_df["ts_code"].astype(str).str.strip()})
    feats = feats.drop_duplicates(subset=["ts_code"], keep="first")

    if "Pack0_base" in packs_used:
        p0 = build_pack0_base(cfg, trade_date, pred_df)
        feats = feats.merge(p0, on="ts_code", how="left")

    if "Pack1_tushare_basic" in packs_used:
        p1 = build_pack1_tushare_basic(cfg, trade_date)
        feats = feats.merge(p1, on="ts_code", how="left")

    if "Pack2_limit_micro" in packs_used:
        p2 = build_pack2_limit_micro(cfg, trade_date)
        feats = feats.merge(p2, on="ts_code", how="left")

    return feats
