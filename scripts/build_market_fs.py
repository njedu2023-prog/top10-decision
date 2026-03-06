#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_market_fs.py

目标：
- 将 data/market/daily_{trade_date}.csv 规范化为 Feature Store 四件套：
  1) data/market/features_base_{trade_date}.csv
  2) data/market/features_limit_{trade_date}.csv
  3) data/market/truth_close_{trade_date}.csv
  4) data/market/_meta_{trade_date}.json

设计原则：
- 先把 FS 生产层跑通，再逐步提高字段丰富度
- 当前版本优先兼容已有 daily_* 输入，不要求一次达到 75% 丰富度
- 缺什么字段就审计什么字段，不静默假装存在
- 主键统一为：trade_date + ts_code
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_DIR = Path("data/market")
KEY_COLS = ["trade_date", "ts_code"]


# -----------------------------
# 通用工具
# -----------------------------
def _read_csv_any(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _safe_float(v: Any) -> float | None:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _safe_div(a: Any, b: Any) -> float | None:
    a = _safe_float(a)
    b = _safe_float(b)
    if a is None or b in (None, 0):
        return None
    return a / b


def _safe_pct(a: Any, b: Any) -> float | None:
    a = _safe_float(a)
    b = _safe_float(b)
    if a is None or b in (None, 0):
        return None
    return a / b - 1.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_trade_date_from_name(path: Path, prefix: str = "daily") -> str | None:
    if path is None:
        return None
    m = re.match(rf"^{re.escape(prefix)}_(\d{{8}})\.csv$", path.name)
    if not m:
        return None
    return m.group(1)


def _find_latest_daily_file() -> Path | None:
    if not MARKET_DIR.exists():
        return None
    files = sorted(MARKET_DIR.glob("daily_*.csv"))
    if not files:
        return None
    return files[-1]


def _resolve_daily_file(trade_date: str | None = None, source_file: str | None = None) -> tuple[Path | None, str | None]:
    if source_file:
        p = Path(source_file)
        if p.exists():
            td = _extract_trade_date_from_name(p, prefix="daily") or trade_date
            return p, td

    if trade_date:
        p = MARKET_DIR / f"daily_{trade_date}.csv"
        if p.exists():
            return p, trade_date

    latest = _find_latest_daily_file()
    if latest is None:
        return None, None
    td = _extract_trade_date_from_name(latest, prefix="daily")
    return latest, td


def _normalize_trade_date(v: Any, fallback: str | None = None) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return fallback
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    if not s:
        return fallback
    if re.fullmatch(r"\d{8}", s):
        return s
    return fallback


def _normalize_ts_code(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _coalesce_row(row: pd.Series, cols: list[str]) -> Any:
    for c in cols:
        if c in row.index:
            val = row[c]
            if pd.notna(val) and str(val).strip() != "":
                return val
    return None


def _standardize_daily(df: pd.DataFrame, trade_date_fallback: str | None) -> pd.DataFrame:
    """
    尽量兼容不同日表字段名，统一成常见列。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    alias_map = {
        "trade_date": ["trade_date", "date"],
        "ts_code": ["ts_code", "code", "jq_code"],
        "name": ["name", "stock_name"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "price"],
        "pre_close": ["pre_close", "prev_close", "last_close"],
        "pct_chg": ["pct_chg", "change_pct", "涨跌幅"],
        "vol": ["vol", "volume"],
        "amount": ["amount", "turnover", "成交额"],
        "turnover_rate": ["turnover_rate", "换手率"],
        "volume_ratio": ["volume_ratio", "量比"],
        "total_mv": ["total_mv", "market_value_total", "总市值"],
        "circ_mv": ["circ_mv", "market_value_circ", "流通市值"],
        "pe_ttm": ["pe_ttm", "pe", "市盈率"],
        "pb": ["pb", "市净率"],
        "moneyflow": ["moneyflow", "net_mf_amount", "主力净流入"],
        "northbound_net": ["northbound_net", "hsgt", "north_money"],
        "market_regime": ["market_regime"],
        "board": ["board", "industry", "concept"],
        "limit_type": ["limit_type"],
        "open_times": ["open_times"],
        "seal_amount": ["seal_amount"],
        "first_seal_time": ["first_seal_time"],
        "last_seal_time": ["last_seal_time"],
        "break_count": ["break_count", "炸板次数"],
        "up_limit": ["up_limit"],
        "down_limit": ["down_limit"],
        "hot_boards_score": ["hot_boards_score"],
        "board_crowding": ["board_crowding"],
        "top_list_net_buy": ["top_list_net_buy"],
        "abnormal_volume": ["abnormal_volume"],
    }

    new_cols: dict[str, Any] = {}
    for std_col, aliases in alias_map.items():
        src = _pick_first_existing(out, aliases)
        if src:
            new_cols[std_col] = out[src]
        else:
            new_cols[std_col] = None

    std = pd.DataFrame(new_cols)

    std["trade_date"] = std["trade_date"].apply(lambda x: _normalize_trade_date(x, fallback=trade_date_fallback))
    std["ts_code"] = std["ts_code"].apply(_normalize_ts_code)

    # 清洗数值列
    numeric_cols = [
        "open", "high", "low", "close", "pre_close", "pct_chg",
        "vol", "amount", "turnover_rate", "volume_ratio",
        "total_mv", "circ_mv", "pe_ttm", "pb", "moneyflow",
        "northbound_net", "open_times", "seal_amount", "break_count",
        "up_limit", "down_limit", "hot_boards_score", "board_crowding",
        "top_list_net_buy", "abnormal_volume",
    ]
    for c in numeric_cols:
        if c in std.columns:
            std[c] = pd.to_numeric(std[c], errors="coerce")

    std = std.dropna(subset=["trade_date", "ts_code"]).copy()
    std = std.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)

    return std


# -----------------------------
# Base 特征构造
# -----------------------------
def build_features_base(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = pd.DataFrame()
    out["trade_date"] = df["trade_date"]
    out["ts_code"] = df["ts_code"]
    out["name"] = df.get("name")
    out["board"] = df.get("board")

    # 基础行情
    for c in ["open", "high", "low", "close", "pre_close", "pct_chg"]:
        out[c] = df.get(c)

    # 当日价格行为派生
    out["returns_1d"] = df.get("pct_chg")
    if "pct_chg" in out.columns:
        out["returns_1d"] = out["returns_1d"].where(out["returns_1d"].isna(), out["returns_1d"] / 100.0)

    out["high_low_range"] = (df.get("high") - df.get("low")) / df.get("pre_close")
    out["candle_body"] = (df.get("close") - df.get("open")) / df.get("pre_close")
    out["gap_open"] = (df.get("open") - df.get("pre_close")) / df.get("pre_close")

    # 成交与流动性
    out["vol"] = df.get("vol")
    out["amount"] = df.get("amount")
    out["turnover_rate"] = df.get("turnover_rate")
    out["volume_ratio"] = df.get("volume_ratio")
    out["amihud_illiquidity"] = df.get("pct_chg").abs() / df.get("amount")
    if "pct_chg" in df.columns:
        out["amihud_illiquidity"] = (df.get("pct_chg").abs() / 100.0) / df.get("amount")

    # 波动与风险结构（当前日表不一定具备历史，先保留骨架）
    out["volatility_5d"] = None
    out["volatility_10d"] = None
    out["volatility_20d"] = None
    out["atr"] = None
    out["downside_vol"] = None
    out["max_drawdown_20d"] = None
    out["tail_risk_score"] = None

    # 估值与市值
    out["total_mv"] = df.get("total_mv")
    out["circ_mv"] = df.get("circ_mv")
    out["pe_ttm"] = df.get("pe_ttm")
    out["pb"] = df.get("pb")

    # 资金流与情绪
    out["moneyflow"] = df.get("moneyflow")
    out["northbound_net"] = df.get("northbound_net")
    out["market_regime"] = df.get("market_regime")

    # 题材热度
    out["hot_boards_score"] = df.get("hot_boards_score")
    out["board_crowding"] = df.get("board_crowding")

    # 龙虎榜/异动
    out["top_list_net_buy"] = df.get("top_list_net_buy")
    out["abnormal_volume"] = df.get("abnormal_volume")

    # 当前还没有多日序列时，先占位
    out["ret_2d"] = None
    out["ret_5d"] = None
    out["ret_10d"] = None
    out["bid_ask_proxy"] = None
    out["spread_proxy"] = None

    return out


# -----------------------------
# Limit 特征构造
# -----------------------------
def build_features_limit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = pd.DataFrame()
    out["trade_date"] = df["trade_date"]
    out["ts_code"] = df["ts_code"]
    out["name"] = df.get("name")

    out["limit_type"] = df.get("limit_type")
    out["open_times"] = df.get("open_times")
    out["seal_amount"] = df.get("seal_amount")
    out["first_seal_time"] = df.get("first_seal_time")
    out["last_seal_time"] = df.get("last_seal_time")
    out["break_count"] = df.get("break_count")
    out["up_limit"] = df.get("up_limit")
    out["down_limit"] = df.get("down_limit")

    # 是否涨停
    is_limit_up = None
    if "close" in df.columns and "up_limit" in df.columns:
        is_limit_up = (pd.to_numeric(df["close"], errors="coerce") >= pd.to_numeric(df["up_limit"], errors="coerce")).astype("float")
    out["is_limit_up"] = is_limit_up

    # 简单强度分数：后续可升级
    score = pd.Series(0.0, index=df.index)
    if "open_times" in df.columns:
        ot = pd.to_numeric(df["open_times"], errors="coerce").fillna(0.0)
        score += (1.0 / (1.0 + ot))
    if "seal_amount" in df.columns:
        sa = pd.to_numeric(df["seal_amount"], errors="coerce").fillna(0.0)
        max_sa = sa.max()
        if pd.notna(max_sa) and max_sa > 0:
            score += sa / max_sa
    out["limit_up_strength"] = score.replace([float("inf"), -float("inf")], pd.NA)

    return out


# -----------------------------
# Truth 层构造
# -----------------------------
def build_truth_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = pd.DataFrame()
    out["trade_date"] = df["trade_date"]
    out["ts_code"] = df["ts_code"]
    out["name"] = df.get("name")
    out["close"] = df.get("close")
    out["pre_close"] = df.get("pre_close")
    out["pct_chg"] = df.get("pct_chg")
    out["up_limit"] = df.get("up_limit")
    out["down_limit"] = df.get("down_limit")
    return out


# -----------------------------
# Meta
# -----------------------------
def _coverage_stats(df: pd.DataFrame, required_cols: list[str]) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": [],
            "coverage_ratio": 0.0,
            "missing_ratio": 1.0,
            "non_null_rate_by_col": {},
        }

    non_null_rate_by_col = {}
    covered = 0
    for c in required_cols:
        if c in df.columns:
            rate = float(df[c].notna().mean())
        else:
            rate = 0.0
        non_null_rate_by_col[c] = round(rate, 6)
        if rate > 0:
            covered += 1

    coverage_ratio = covered / len(required_cols) if required_cols else 0.0
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coverage_ratio": round(float(coverage_ratio), 6),
        "missing_ratio": round(float(1.0 - coverage_ratio), 6),
        "non_null_rate_by_col": non_null_rate_by_col,
    }


def build_meta(
    trade_date: str,
    source_file: Path,
    raw_df: pd.DataFrame,
    base_df: pd.DataFrame,
    limit_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> dict[str, Any]:
    base_required = [
        "open", "high", "low", "close", "pre_close", "returns_1d",
        "vol", "amount", "turnover_rate", "volume_ratio",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "total_mv", "circ_mv", "pe_ttm", "pb",
        "moneyflow", "northbound_net",
        "hot_boards_score", "board_crowding",
    ]
    limit_required = [
        "is_limit_up", "limit_type", "open_times", "seal_amount",
        "first_seal_time", "last_seal_time", "break_count", "limit_up_strength",
    ]
    truth_required = ["close", "pre_close", "pct_chg"]

    sha = os.getenv("GITHUB_SHA", "")
    snapshot_id = f"{trade_date}:{source_file.name}"

    base_stats = _coverage_stats(base_df, base_required)
    limit_stats = _coverage_stats(limit_df, limit_required)
    truth_stats = _coverage_stats(truth_df, truth_required)

    richness_estimate = round(
        (base_stats["coverage_ratio"] + limit_stats["coverage_ratio"] + truth_stats["coverage_ratio"]) / 3.0,
        6,
    )

    return {
        "trade_date": trade_date,
        "created_at_utc": _now_utc(),
        "source_snapshot_id": snapshot_id,
        "source_file": str(source_file),
        "source_rows": int(len(raw_df)) if raw_df is not None else 0,
        "commit_sha": sha,
        "fs_version": "v0.1",
        "richness_target": 0.75,
        "richness_estimate": richness_estimate,
        "tables": {
            "features_base": base_stats,
            "features_limit": limit_stats,
            "truth_close": truth_stats,
        },
        "notes": [
            "当前版本优先从 daily_* 日表构建 FS 四件套。",
            "多日波动、回撤、北向、limit 微结构、题材热度等字段后续可继续注入增强。",
            "若字段缺失，将在本 meta 中显式反映，不静默伪造。",
        ],
    }


# -----------------------------
# 输出
# -----------------------------
def write_outputs(
    trade_date: str,
    base_df: pd.DataFrame,
    limit_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    _ensure_dir(MARKET_DIR)

    base_path = MARKET_DIR / f"features_base_{trade_date}.csv"
    limit_path = MARKET_DIR / f"features_limit_{trade_date}.csv"
    truth_path = MARKET_DIR / f"truth_close_{trade_date}.csv"
    meta_path = MARKET_DIR / f"_meta_{trade_date}.json"

    base_df.to_csv(base_path, index=False, encoding="utf-8-sig")
    limit_df.to_csv(limit_path, index=False, encoding="utf-8-sig")
    truth_df.to_csv(truth_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 market Feature Store 四件套")
    parser.add_argument("--trade-date", dest="trade_date", default=None, help="交易日 YYYYMMDD")
    parser.add_argument("--source-file", dest="source_file", default=None, help="指定输入 daily csv 文件路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    daily_file, trade_date = _resolve_daily_file(
        trade_date=args.trade_date,
        source_file=args.source_file,
    )

    if daily_file is None or trade_date is None:
        print("[build_market_fs] ERROR: 未找到可用的 daily_{trade_date}.csv")
        return 2

    raw_df = _read_csv_any(daily_file)
    if raw_df.empty:
        print(f"[build_market_fs] ERROR: 输入文件为空或读取失败: {daily_file}")
        return 2

    std_df = _standardize_daily(raw_df, trade_date_fallback=trade_date)
    if std_df.empty:
        print(f"[build_market_fs] ERROR: 标准化后为空，无法构建 FS: {daily_file}")
        return 2

    base_df = build_features_base(std_df)
    limit_df = build_features_limit(std_df)
    truth_df = build_truth_close(std_df)
    meta = build_meta(
        trade_date=trade_date,
        source_file=daily_file,
        raw_df=std_df,
        base_df=base_df,
        limit_df=limit_df,
        truth_df=truth_df,
    )

    write_outputs(
        trade_date=trade_date,
        base_df=base_df,
        limit_df=limit_df,
        truth_df=truth_df,
        meta=meta,
    )

    print(f"[build_market_fs] OK trade_date={trade_date}")
    print(f"[build_market_fs] source={daily_file}")
    print(f"[build_market_fs] rows={len(std_df)}")
    print(f"[build_market_fs] outputs:")
    print(f"  - data/market/features_base_{trade_date}.csv")
    print(f"  - data/market/features_limit_{trade_date}.csv")
    print(f"  - data/market/truth_close_{trade_date}.csv")
    print(f"  - data/market/_meta_{trade_date}.json")
    print(f"[build_market_fs] richness_estimate={meta.get('richness_estimate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
