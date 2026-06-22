#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium — Factor Packs Builders

职责：
- 根据 packs_used 构建特征 DataFrame（ts_code 粒度）
- Pack0：从 pred_source + market OHLCV（近N日）构建动量/波动/影线/量能比等
- Pack1：从 data/market_basic/daily_basic_YYYYMMDD.csv 读取流动性/市值硬因子
- Pack2：从 data/limit/limit_micro_YYYYMMDD.csv 读取涨停结构微观因子
- Pack3：从 pred_source / market raw 中读取分钟级与竞价结构因子

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


def _has_pack(packs_used: List[str], *names: str) -> bool:
    used = {str(x).strip() for x in (packs_used or [])}
    return any(name in used for name in names)


def _num01(s: pd.Series, default: float = np.nan) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any() and float(x.dropna().abs().max()) > 1.5:
        x = x / 100.0
    return x.fillna(default).clip(0.0, 1.0)


def _num_clip(s: pd.Series, lo: float = 0.0, hi: float = 1.0, default: float = np.nan) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(default)
    return x.clip(lo, hi)


def _first_existing(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        hit = cols.get(str(name).strip().lower())
        if hit is not None:
            return str(hit)
    return None


def _intraday_raw_candidates(cfg: PremiumConfig, trade_date: str, filename: str) -> List[Path]:
    td = _to_yyyymmdd(trade_date)
    year = td[:4] if len(td) >= 4 else ""
    root = cfg.repo_root()
    return [
        root / "data" / "market" / "raw" / year / td / filename,
        root / "data" / "market" / "raw" / "latest" / filename,
        root / "data" / "raw" / year / td / filename,
        root / "data" / "latest" / filename,
    ]


def _load_first_existing(paths: List[Path]) -> Optional[pd.DataFrame]:
    for p in paths:
        df = _safe_read_csv(p)
        if df is not None and not df.empty:
            return df
    return None


def _repo_root(cfg: PremiumConfig) -> Path:
    fn = getattr(cfg, "repo_root", None)
    if callable(fn):
        try:
            return Path(fn()).resolve()
        except Exception:
            pass
    return Path.cwd().resolve()


def _market_cache_root(cfg: PremiumConfig) -> Path:
    fn = getattr(cfg, "market_cache_root", None)
    if callable(fn):
        try:
            return Path(fn()).resolve()
        except Exception:
            pass
    return (_repo_root(cfg) / getattr(cfg, "market_cache_dir", "data/market")).resolve()


def _optional_path_from_cfg(cfg: PremiumConfig, method_name: str, trade_date: str, fallback_parts: Tuple[str, ...]) -> Path:
    fn = getattr(cfg, method_name, None)
    if callable(fn):
        try:
            return Path(fn(trade_date)).resolve()
        except Exception:
            pass
    return (_repo_root(cfg).joinpath(*fallback_parts)).resolve()


def _list_market_dates(cfg: PremiumConfig) -> List[str]:
    # 通过文件名列出 data/market/daily_YYYYMMDD.csv 的日期（不依赖交易所日历）
    root = _market_cache_root(cfg)
    if not root.exists():
        return []
    out = []
    tpl = str(getattr(cfg, "market_daily_tpl", "daily_{trade_date}.csv"))
    prefix = tpl.split("{trade_date}")[0] if "{trade_date}" in tpl else "daily_"
    suffix = tpl.split("{trade_date}")[-1] if "{trade_date}" in tpl else ".csv"
    for p in root.glob(f"{prefix}*{suffix}"):
        s = p.name
        if prefix and s.startswith(prefix):
            s = s[len(prefix):]
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)]
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
    td = _to_yyyymmdd(trade_date)
    p = _optional_path_from_cfg(
        cfg,
        "market_basic_path",
        td,
        ("data", "market_basic", f"daily_basic_{td}.csv"),
    )
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
    td = _to_yyyymmdd(trade_date)
    p = _optional_path_from_cfg(
        cfg,
        "limit_micro_path",
        td,
        ("data", "limit", f"limit_micro_{td}.csv"),
    )
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


def build_pack3_intraday(cfg: PremiumConfig, trade_date: str, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    分钟级/竞价软特征包。

    第一优先级消费 a-top10 已经透传到 pred_source_latest 的分钟衍生字段；
    第二优先级宽松合并本仓库 data/market/raw 下的 intraday_features/stk_auction。
    所有输出列都使用 factor_intraday_* 前缀，避免和原始输入字段冲突。
    """
    base = pd.DataFrame({"ts_code": pred_df["ts_code"].astype(str).str.strip()}).drop_duplicates("ts_code")
    src = pred_df.copy()
    src["ts_code"] = src["ts_code"].astype(str).str.strip()
    src = src.drop_duplicates("ts_code", keep="last")

    out = base.merge(src[["ts_code"]], on="ts_code", how="left")
    idx = out.index

    def src_col(*names: str, default: float = np.nan) -> pd.Series:
        c = _first_existing(src, *names)
        if c is None:
            return pd.Series([default] * len(src), index=src.index, dtype="float64")
        return pd.to_numeric(src[c], errors="coerce")

    raw = pd.DataFrame({"ts_code": src["ts_code"].astype(str)})
    raw["factor_intraday_available"] = _num_clip(src_col("intraday_available", default=0.0), 0.0, 1.0, 0.0)
    raw["factor_intraday_quality"] = _num01(src_col("intraday_quality_score", "intraday_confidence_score", default=np.nan), default=0.5)
    raw["factor_intraday_confidence"] = _num01(src_col("intraday_confidence_score", "intraday_quality_score", default=np.nan), default=0.5)
    raw["factor_intraday_soft_risk"] = _num01(src_col("intraday_soft_risk_score", "intraday_risk_score", default=np.nan), default=0.0)
    raw["factor_intraday_risk"] = _num01(src_col("intraday_risk_score", "intraday_soft_risk_score", default=np.nan), default=0.0)
    raw["factor_intraday_hard_risk"] = _num_clip(src_col("intraday_hard_risk_flag", default=0.0), 0.0, 1.0, 0.0)
    raw["factor_late_withdraw"] = _num01(src_col("late_withdraw_score", default=np.nan), default=0.0)
    raw["factor_reseal"] = _num01(src_col("reseal_score", default=np.nan), default=0.5)
    raw["factor_open_board_count"] = _num_clip(src_col("open_board_count", default=np.nan), 0.0, 10.0, 0.0)
    raw["factor_auction_strength"] = _num01(src_col("auction_strength_score", default=np.nan), default=0.5)

    # 派生为直接可用于评分的 0~1 alpha / risk。
    raw["factor_intraday_attack_edge"] = (
        0.36 * raw["factor_auction_strength"]
        + 0.28 * raw["factor_reseal"]
        + 0.22 * raw["factor_intraday_quality"]
        + 0.14 * raw["factor_intraday_confidence"]
        - 0.18 * raw["factor_late_withdraw"]
        - 0.08 * (raw["factor_open_board_count"] / 5.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    raw["factor_intraday_execution_edge"] = (
        0.42 * raw["factor_intraday_quality"]
        + 0.28 * raw["factor_intraday_confidence"]
        + 0.20 * raw["factor_auction_strength"]
        + 0.10 * (1.0 - raw["factor_intraday_soft_risk"])
    ).clip(0.0, 1.0)
    raw["factor_intraday_risk_penalty"] = (
        0.42 * raw["factor_intraday_soft_risk"]
        + 0.25 * raw["factor_intraday_risk"]
        + 0.20 * raw["factor_late_withdraw"]
        + 0.13 * raw["factor_intraday_hard_risk"]
    ).clip(0.0, 1.0)

    out = base.merge(raw, on="ts_code", how="left")

    # 宽松合并 raw intraday/auction 文件里的数值列；缺失不影响主流程。
    for prefix, filename in (
        ("factor_raw_intraday", "intraday_features.csv"),
        ("factor_raw_auction", "stk_auction.csv"),
    ):
        extra = _load_first_existing(_intraday_raw_candidates(cfg, trade_date, filename))
        if extra is None or extra.empty:
            continue
        code_col = _first_existing(extra, "ts_code", "code", "symbol", "ticker", "股票代码", "代码")
        if code_col is None:
            continue
        extra = extra.copy()
        extra["ts_code"] = extra[code_col].astype(str).str.strip()
        keep = ["ts_code"]
        for col in extra.columns:
            if col == "ts_code" or col == code_col:
                continue
            if any(k in str(col).lower() for k in ("name", "date", "time", "status", "reason")):
                continue
            x = pd.to_numeric(extra[col], errors="coerce")
            if x.notna().mean() >= 0.50:
                new_col = f"{prefix}_{str(col).strip()}"
                extra[new_col] = x
                keep.append(new_col)
        if len(keep) > 1:
            out = out.merge(extra[keep].drop_duplicates("ts_code", keep="last"), on="ts_code", how="left")

    for c in out.columns:
        if c != "ts_code":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_features_by_packs(cfg: PremiumConfig, trade_date: str, pred_df: pd.DataFrame, packs_used: List[str]) -> pd.DataFrame:
    """
    输入：trade_date, pred_df, packs_used
    输出：按 ts_code 合并后的总特征表
    """
    feats = pd.DataFrame({"ts_code": pred_df["ts_code"].astype(str).str.strip()})
    feats = feats.drop_duplicates(subset=["ts_code"], keep="first")

    if _has_pack(packs_used, "Pack0", "Pack0_base"):
        p0 = build_pack0_base(cfg, trade_date, pred_df)
        feats = feats.merge(p0, on="ts_code", how="left")

    if _has_pack(packs_used, "Pack1", "Pack1_tushare_basic"):
        p1 = build_pack1_tushare_basic(cfg, trade_date)
        feats = feats.merge(p1, on="ts_code", how="left")

    if _has_pack(packs_used, "Pack2", "Pack2_limit_micro"):
        p2 = build_pack2_limit_micro(cfg, trade_date)
        feats = feats.merge(p2, on="ts_code", how="left")

    if _has_pack(packs_used, "Pack3", "Pack3_intraday"):
        p3 = build_pack3_intraday(cfg, trade_date, pred_df)
        feats = feats.merge(p3, on="ts_code", how="left")

    return feats
