#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Truth Layer（行情事实层）

目标：
- 以 data/market/daily_{YYYYMMDD}.csv 作为 Premium 的“真值行情缓存”
- 缓存不存在时：
  1) 尝试从本地 a-share-top3-data 读取 raw daily.csv
  2) 再尝试从 GitHub raw URL 拉取
- 严格校验字段契约：ts_code, trade_date, open, high, low, close, vol, amount

注意：
- 这里不做复权处理；复权开关留到后续（需要 adj_factor 数据）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import PremiumConfig


REQUIRED_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]


@dataclass(frozen=True)
class MarketFetchResult:
    ok: bool
    trade_date: str
    cache_path: Optional[str] = None
    reason: str = ""


def _year(trade_date: str) -> str:
    return str(trade_date)[:4]


def _normalize_daily_df(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    # 统一列名小写（防御）
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 有些源可能没有 amount/vol 的类型一致性，这里做安全转换
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"daily.csv missing columns: {missing}")

    # 仅保留所需列（契约锁死，避免上游字段漂移影响）
    df = df[REQUIRED_COLS].copy()

    # trade_date 强制为字符串 YYYYMMDD
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8)
    df = df[df["trade_date"] == str(trade_date)]

    # ts_code 也强制字符串
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    # 数值列安全转换
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts_code", "close"])
    if df.empty:
        raise ValueError(f"daily.csv after normalize is empty for trade_date={trade_date}")

    return df


def _try_read_local_top3(cfg: PremiumConfig, trade_date: str) -> Optional[pd.DataFrame]:
    """
    尝试从本地 a-share-top3-data 读 raw daily.csv
    允许多种可能路径（你本地/Actions checkout 方式不固定）
    """
    repo_root = cfg.repo_root()
    year = _year(trade_date)

    candidates = []

    # 1) repo_root/top3_local_dir/...
    candidates.append(
        repo_root / cfg.top3_local_dir / "data" / "raw" / year / str(trade_date) / "daily.csv"
    )
    # 2) repo_root/../top3_local_dir/...
    candidates.append(
        repo_root.parent / cfg.top3_local_dir / "data" / "raw" / year / str(trade_date) / "daily.csv"
    )
    # 3) repo_root/_warehouse/a-share-top3-data/...（你有时会放到 _warehouse）
    candidates.append(
        repo_root / "_warehouse" / "a-share-top3-data" / "data" / "raw" / year / str(trade_date) / "daily.csv"
    )

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df
    return None


def _try_fetch_remote_top3(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    """
    从 GitHub raw 拉取 a-share-top3-data 的 daily.csv
    """
    year = _year(trade_date)
    url = f"{cfg.top3_raw_base_url}/data/raw/{year}/{trade_date}/daily.csv"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"fetch remote daily.csv failed: {r.status_code} url={url}")
    # 用 pandas 读取文本
    from io import StringIO
    return pd.read_csv(StringIO(r.text))


def ensure_daily_cached(cfg: PremiumConfig, trade_date: str) -> MarketFetchResult:
    """
    确保 data/market/daily_{trade_date}.csv 存在且字段合法。
    """
    cache_path = cfg.market_daily_cache_path(trade_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # cache_only：只允许用缓存
    if cfg.market_fetch_mode.lower() == "cache_only":
        if not cache_path.exists():
            return MarketFetchResult(False, trade_date, None, "cache_only but cache not found")
        try:
            df = pd.read_csv(cache_path)
            _ = _normalize_daily_df(df, trade_date)
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(cache_only)")
        except Exception as e:
            return MarketFetchResult(False, trade_date, str(cache_path), f"cache invalid: {e}")

    # cache_first：优先缓存，缺则拉取并写入
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            _ = _normalize_daily_df(df, trade_date)
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(cache_hit)")
        except Exception:
            # 缓存坏了就重建
            pass

    # 先尝试本地 top3
    try:
        local_df = _try_read_local_top3(cfg, trade_date)
        if local_df is not None:
            df_norm = _normalize_daily_df(local_df, trade_date)
            df_norm.to_csv(cache_path, index=False)
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(fetched_local_top3)")
    except Exception as e:
        # 继续尝试远程
        local_err = str(e)
    else:
        local_err = ""

    # 再尝试远程
    try:
        remote_df = _try_fetch_remote_top3(cfg, trade_date)
        df_norm = _normalize_daily_df(remote_df, trade_date)
        df_norm.to_csv(cache_path, index=False)
        return MarketFetchResult(True, trade_date, str(cache_path), "ok(fetched_remote_top3)")
    except Exception as e:
        msg = f"fetch_remote_failed: {e}"
        if local_err:
            msg = f"fetch_local_failed: {local_err}; " + msg
        return MarketFetchResult(False, trade_date, None, msg)


def load_daily(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    """
    读取并返回规范化后的 daily（来自缓存；必要时会自动拉取并写缓存）
    """
    r = ensure_daily_cached(cfg, trade_date)
    if not r.ok:
        raise RuntimeError(f"ensure_daily_cached failed trade_date={trade_date}: {r.reason}")

    df = pd.read_csv(Path(r.cache_path))
    return _normalize_daily_df(df, trade_date)
