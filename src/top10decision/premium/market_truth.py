#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Truth Layer（行情事实层）

目标（锁死）：
- 以 data/market/daily_{YYYYMMDD}.csv 作为 Premium 的“真值行情缓存”
- 缓存不存在时：
  1) 尝试从本地 a-share-top3-data 读取 raw daily.csv
  2) 再尝试从 GitHub raw URL 拉取
- 严格校验字段契约：ts_code, trade_date, open, high, low, close, vol, amount

注意：
- 这里不做复权处理；复权开关留到后续（需要 adj_factor 数据）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import PremiumConfig


REQUIRED_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
_TRADE_DATE_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class MarketFetchResult:
    ok: bool
    trade_date: str
    cache_path: Optional[str] = None
    reason: str = ""


def _ensure_trade_date(trade_date: str) -> str:
    td = str(trade_date).strip().replace("-", "")[:8]
    if not _TRADE_DATE_RE.match(td):
        raise ValueError(f"invalid trade_date={trade_date} (expect YYYYMMDD)")
    return td


def _year(trade_date: str) -> str:
    return str(trade_date)[:4]


def _read_csv_smart(path: Path) -> pd.DataFrame:
    # 防御：不同环境可能出现 utf-8-sig / gbk
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # 兜底：让 pandas 自己猜
    if last_err:
        return pd.read_csv(path)
    return pd.read_csv(path)


def _normalize_daily_df(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    trade_date = _ensure_trade_date(trade_date)

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"daily.csv missing columns: {missing}")

    # 仅保留所需列（契约锁死，避免上游字段漂移影响）
    df = df[REQUIRED_COLS].copy()

    # trade_date 强制为 YYYYMMDD 字符串
    df["trade_date"] = (
        df["trade_date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    )
    df = df[df["trade_date"] == trade_date]

    # ts_code 强制字符串
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    # 数值列安全转换
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 必须有 close
    df = df.dropna(subset=["ts_code", "close"])
    if df.empty:
        raise ValueError(f"daily.csv after normalize is empty for trade_date={trade_date}")

    return df


def _cfg_get(cfg: PremiumConfig, name: str, default):
    return getattr(cfg, name, default)


def _try_read_local_top3(cfg: PremiumConfig, trade_date: str) -> Optional[pd.DataFrame]:
    """
    尝试从本地 a-share-top3-data 读 raw daily.csv
    允许多种可能路径（本地/Actions checkout 方式不固定）
    """
    trade_date = _ensure_trade_date(trade_date)
    repo_root = cfg.repo_root()
    year = _year(trade_date)

    top3_local_dir = _cfg_get(cfg, "top3_local_dir", "a-share-top3-data")

    candidates = [
        # 1) repo_root/top3_local_dir/...
        repo_root / top3_local_dir / "data" / "raw" / year / trade_date / "daily.csv",
        # 2) repo_root/../top3_local_dir/...
        repo_root.parent / top3_local_dir / "data" / "raw" / year / trade_date / "daily.csv",
        # 3) repo_root/_warehouse/a-share-top3-data/...
        repo_root / "_warehouse" / "a-share-top3-data" / "data" / "raw" / year / trade_date / "daily.csv",
        # 4) repo_root/_warehouse/top3_local_dir/...
        repo_root / "_warehouse" / top3_local_dir / "data" / "raw" / year / trade_date / "daily.csv",
    ]

    for p in candidates:
        if p.exists():
            return _read_csv_smart(p)

    return None


def _try_fetch_remote_top3(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    """
    从 GitHub raw 拉取 a-share-top3-data 的 daily.csv
    """
    trade_date = _ensure_trade_date(trade_date)
    year = _year(trade_date)

    top3_raw_base_url = _cfg_get(
        cfg,
        "top3_raw_base_url",
        "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main",
    )
    url = f"{top3_raw_base_url}/data/raw/{year}/{trade_date}/daily.csv"

    headers = {"User-Agent": "top10-decision-premium/1.0"}
    r = requests.get(url, timeout=30, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"fetch remote daily.csv failed: {r.status_code} url={url}")

    return pd.read_csv(StringIO(r.text))


def ensure_daily_cached(cfg: PremiumConfig, trade_date: str) -> MarketFetchResult:
    """
    确保 data/market/daily_{trade_date}.csv 存在且字段合法。
    """
    trade_date = _ensure_trade_date(trade_date)

    # 兼容不同 config：优先用 cfg.market_daily_cache_path()
    if hasattr(cfg, "market_daily_cache_path"):
        cache_path = cfg.market_daily_cache_path(trade_date)
    else:
        # 兜底：按常见路径拼
        market_cache_dir = _cfg_get(cfg, "market_cache_dir", "data/market")
        tpl = _cfg_get(cfg, "market_daily_tpl", "daily_{trade_date}.csv")
        cache_path = (cfg.repo_root() / market_cache_dir / tpl.format(trade_date=trade_date)).resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    fetch_mode = str(_cfg_get(cfg, "market_fetch_mode", "cache_first")).lower().strip()

    # cache_only：只允许用缓存
    if fetch_mode == "cache_only":
        if not cache_path.exists():
            return MarketFetchResult(False, trade_date, None, "cache_only but cache not found")
        try:
            df = _read_csv_smart(cache_path)
            _ = _normalize_daily_df(df, trade_date)
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(cache_only)")
        except Exception as e:
            return MarketFetchResult(False, trade_date, str(cache_path), f"cache invalid: {e}")

    # cache_first：优先缓存，缺则拉取并写入
    if cache_path.exists():
        try:
            df = _read_csv_smart(cache_path)
            _ = _normalize_daily_df(df, trade_date)
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(cache_hit)")
        except Exception:
            # 缓存坏了就重建
            pass

    # 先尝试本地 top3
    local_err = ""
    try:
        local_df = _try_read_local_top3(cfg, trade_date)
        if local_df is not None:
            df_norm = _normalize_daily_df(local_df, trade_date)
            df_norm.to_csv(cache_path, index=False, encoding="utf-8-sig")
            return MarketFetchResult(True, trade_date, str(cache_path), "ok(fetched_local_top3)")
    except Exception as e:
        local_err = str(e)

    # 再尝试远程
    try:
        remote_df = _try_fetch_remote_top3(cfg, trade_date)
        df_norm = _normalize_daily_df(remote_df, trade_date)
        df_norm.to_csv(cache_path, index=False, encoding="utf-8-sig")
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
    trade_date = _ensure_trade_date(trade_date)
    r = ensure_daily_cached(cfg, trade_date)
    if not r.ok:
        raise RuntimeError(f"ensure_daily_cached failed trade_date={trade_date}: {r.reason}")

    df = _read_csv_smart(Path(r.cache_path))
    return _normalize_daily_df(df, trade_date)


__all__ = ["MarketFetchResult", "ensure_daily_cached", "load_daily"]
