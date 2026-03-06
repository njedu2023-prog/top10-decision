#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_market_fs.py

目标：
- 从 data/market/raw/ 的多源原始文件构建 Feature Store 四件套：
  1) data/market/features_base_{trade_date}.csv
  2) data/market/features_limit_{trade_date}.csv
  3) data/market/truth_close_{trade_date}.csv
  4) data/market/_meta_{trade_date}.json

当前输入主结构：
- data/market/raw/daily_{trade_date}.csv
- data/market/raw/daily_basic_{trade_date}.csv
- data/market/raw/stock_basic_{trade_date}.csv
- data/market/raw/stk_limit_{trade_date}.csv
- data/market/raw/limit_list_d_{trade_date}.csv
- data/market/raw/limit_break_d_{trade_date}.csv
- data/market/raw/limit_up_tags_{trade_date}.csv
- data/market/raw/hot_boards_{trade_date}.csv
- data/market/raw/top_list_{trade_date}.csv
- data/market/raw/moneyflow_hsgt_{trade_date}.csv
- data/market/raw/namechange_{trade_date}.csv

设计原则：
- FS 是标准化特征层，不是原始快照层
- 主键统一：trade_date + ts_code
- 优先保证结构正确、审计完整，再逐步提高丰富度
- 缺失字段要显式反映到 meta，不静默伪造
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_DIR = Path("data/market")
RAW_DIR = MARKET_DIR / "raw"
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


def _safe_json_load(path: Path) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_trade_date(v: Any, fallback: str | None = None) -> str | None:
    if v is None:
        return fallback
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    if not s:
        return fallback
    if re.fullmatch(r"\d{8}", s):
        return s
    return fallback


def _normalize_ts_code(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _to_numeric(v: Any) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _extract_trade_date_from_name(path: Path, stem_prefix: str) -> str | None:
    if path is None:
        return None
    m = re.match(rf"^{re.escape(stem_prefix)}_(\d{{8}})\.csv$", path.name)
    if not m:
        return None
    return m.group(1)


def _find_latest_raw_trade_date() -> str | None:
    if not RAW_DIR.exists():
        return None
    vals: list[str] = []
    for p in RAW_DIR.glob("daily_*.csv"):
        td = _extract_trade_date_from_name(p, "daily")
        if td:
            vals.append(td)
    if not vals:
        return None
    return sorted(set(vals))[-1]


def _resolve_trade_date(trade_date: str | None = None) -> str | None:
    td = _normalize_trade_date(trade_date)
    if td:
        return td
    return _find_latest_raw_trade_date()


def _raw_path(stem: str, trade_date: str) -> Path:
    return RAW_DIR / f"{stem}_{trade_date}.csv"


def _load_raw_table(stem: str, trade_date: str) -> pd.DataFrame:
    return _read_csv_any(_raw_path(stem, trade_date))


def _ensure_keys(df: pd.DataFrame, trade_date_fallback: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].apply(lambda x: _normalize_trade_date(x, fallback=trade_date_fallback))
    else:
        out["trade_date"] = trade_date_fallback

    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].apply(_normalize_ts_code)
    elif "code" in out.columns:
        out["ts_code"] = out["code"].apply(_normalize_ts_code)
    else:
        out["ts_code"] = None

    out = out.dropna(subset=["trade_date", "ts_code"]).copy()
    out = out.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)
    return out


def _merge_left(base: pd.DataFrame, extra: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    if base is None or base.empty:
        return extra.copy() if extra is not None else pd.DataFrame()
    if extra is None or extra.empty:
        return base.copy()
    return base.merge(extra, on=on, how="left")


# -----------------------------
# 各 raw 表标准化
# -----------------------------
def _std_daily(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    ts_code, trade_date, open, high, low, close, vol, amount, pct_chg
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)

    std = out[KEY_COLS].copy()
    std["open"] = _to_numeric(out["open"]) if "open" in out.columns else None
    std["high"] = _to_numeric(out["high"]) if "high" in out.columns else None
    std["low"] = _to_numeric(out["low"]) if "low" in out.columns else None
    std["close"] = _to_numeric(out["close"]) if "close" in out.columns else None
    std["vol"] = _to_numeric(out["vol"]) if "vol" in out.columns else None
    std["amount"] = _to_numeric(out["amount"]) if "amount" in out.columns else None
    std["pct_chg"] = _to_numeric(out["pct_chg"]) if "pct_chg" in out.columns else None

    # daily.csv 当前没有 pre_close，使用 close 和 pct_chg 反推
    close = std["close"]
    pct = std["pct_chg"]
    std["pre_close_est"] = close / (1.0 + pct)
    return std


def _std_daily_basic(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio, total_mv, float_mv
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)
    std = out[KEY_COLS].copy()

    std["turnover_rate"] = _to_numeric(out["turnover_rate"]) if "turnover_rate" in out.columns else None
    std["turnover_rate_f"] = _to_numeric(out["turnover_rate_f"]) if "turnover_rate_f" in out.columns else None
    std["volume_ratio"] = _to_numeric(out["volume_ratio"]) if "volume_ratio" in out.columns else None
    std["total_mv"] = _to_numeric(out["total_mv"]) if "total_mv" in out.columns else None
    std["float_mv"] = _to_numeric(out["float_mv"]) if "float_mv" in out.columns else None

    return std


def _std_stock_basic(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = df.copy()
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].apply(_normalize_ts_code)
    elif "code" in out.columns:
        out["ts_code"] = out["code"].apply(_normalize_ts_code)
    else:
        out["ts_code"] = None

    out["trade_date"] = trade_date
    out = out.dropna(subset=["ts_code"]).copy()
    out = out.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)

    std = out[KEY_COLS].copy()
    std["name"] = out["name"] if "name" in out.columns else None
    std["industry"] = out["industry"] if "industry" in out.columns else None
    std["market"] = out["market"] if "market" in out.columns else None
    std["list_date"] = out["list_date"] if "list_date" in out.columns else None
    return std


def _std_stk_limit(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    ts_code, trade_date, up_limit, down_limit
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)
    std = out[KEY_COLS].copy()
    std["up_limit"] = _to_numeric(out["up_limit"]) if "up_limit" in out.columns else None
    std["down_limit"] = _to_numeric(out["down_limit"]) if "down_limit" in out.columns else None
    return std


def _std_limit_list(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    trade_date, ts_code, name, limit_type, close, up_limit, down_limit,
    open_times, fd_amount, first_time, last_time, seal_amount
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)

    std = out[KEY_COLS].copy()
    std["name"] = out["name"] if "name" in out.columns else None
    std["limit_type"] = out["limit_type"] if "limit_type" in out.columns else None
    std["close_limit"] = _to_numeric(out["close"]) if "close" in out.columns else None
    std["up_limit_list"] = _to_numeric(out["up_limit"]) if "up_limit" in out.columns else None
    std["down_limit_list"] = _to_numeric(out["down_limit"]) if "down_limit" in out.columns else None
    std["open_times_limit"] = _to_numeric(out["open_times"]) if "open_times" in out.columns else None
    std["fd_amount_limit"] = _to_numeric(out["fd_amount"]) if "fd_amount" in out.columns else None
    std["first_time_limit"] = out["first_time"] if "first_time" in out.columns else None
    std["last_time_limit"] = out["last_time"] if "last_time" in out.columns else None
    std["seal_amount_limit"] = _to_numeric(out["seal_amount"]) if "seal_amount" in out.columns else None
    return std


def _std_limit_break(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    trade_date, ts_code, name, open_times, first_time, last_time, fd_amount, seal_amount
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)

    std = out[KEY_COLS].copy()
    std["name_break"] = out["name"] if "name" in out.columns else None
    std["open_times_break"] = _to_numeric(out["open_times"]) if "open_times" in out.columns else None
    std["first_time_break"] = out["first_time"] if "first_time" in out.columns else None
    std["last_time_break"] = out["last_time"] if "last_time" in out.columns else None
    std["fd_amount_break"] = _to_numeric(out["fd_amount"]) if "fd_amount" in out.columns else None
    std["seal_amount_break"] = _to_numeric(out["seal_amount"]) if "seal_amount" in out.columns else None
    return std


def _std_limit_up_tags(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)
    tag_col = _pick_first_existing(out, ["tag", "tags", "concept", "reason"])
    std = out[KEY_COLS].copy()
    std["limit_up_tags"] = out[tag_col].astype(str) if tag_col else None
    return std


def _std_hot_boards(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    trade_date, industry, limit_up_count, rank

    这里做行业热度映射：
    industry -> board
    limit_up_count -> hot_boards_score
    rank -> board_crowding_rank
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["board", "hot_boards_score", "board_crowding_rank"])

    out = df.copy()
    std = pd.DataFrame()
    std["board"] = out["industry"].astype(str).str.strip() if "industry" in out.columns else None
    std["hot_boards_score"] = _to_numeric(out["limit_up_count"]) if "limit_up_count" in out.columns else None
    std["board_crowding_rank"] = _to_numeric(out["rank"]) if "rank" in out.columns else None

    std = std.dropna(subset=["board"]).copy()
    std = std.drop_duplicates(subset=["board"], keep="last").reset_index(drop=True)
    return std


def _std_top_list(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    ts_code, trade_date, name, close, pct_change, turnover_rate, amount,
    l_sell, l_buy, l_amount, net_amount, net_rate, amount_rate, float_values, reason
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_keys(df.copy(), trade_date)

    std = out[KEY_COLS].copy()
    std["name_top"] = out["name"] if "name" in out.columns else None
    std["top_close"] = _to_numeric(out["close"]) if "close" in out.columns else None
    std["top_pct_change"] = _to_numeric(out["pct_change"]) if "pct_change" in out.columns else None
    std["top_turnover_rate"] = _to_numeric(out["turnover_rate"]) if "turnover_rate" in out.columns else None
    std["top_amount"] = _to_numeric(out["amount"]) if "amount" in out.columns else None
    std["l_sell"] = _to_numeric(out["l_sell"]) if "l_sell" in out.columns else None
    std["l_buy"] = _to_numeric(out["l_buy"]) if "l_buy" in out.columns else None
    std["l_amount"] = _to_numeric(out["l_amount"]) if "l_amount" in out.columns else None
    std["top_list_net_buy"] = _to_numeric(out["net_amount"]) if "net_amount" in out.columns else None
    std["top_net_rate"] = _to_numeric(out["net_rate"]) if "net_rate" in out.columns else None
    std["amount_rate"] = _to_numeric(out["amount_rate"]) if "amount_rate" in out.columns else None
    std["float_values"] = _to_numeric(out["float_values"]) if "float_values" in out.columns else None
    std["top_reason"] = out["reason"] if "reason" in out.columns else None
    return std


def _std_moneyflow_hsgt_market(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    已确认表头：
    ts_code, trade_date, ggt_ss, ggt_sz, hgt, sgt, north_money, south_money

    从你给的样例看，ts_code 可能为空，因此这张表按“市场级”处理：
    trade_date -> north_money_market / south_money_market / hgt / sgt
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date", "north_money_market", "south_money_market", "hgt_market", "sgt_market"])

    out = df.copy()
    if "trade_date" not in out.columns:
        return pd.DataFrame(columns=["trade_date", "north_money_market", "south_money_market", "hgt_market", "sgt_market"])

    out["trade_date"] = out["trade_date"].apply(lambda x: _normalize_trade_date(x, fallback=trade_date))
    out = out.dropna(subset=["trade_date"]).copy()

    std = pd.DataFrame()
    std["trade_date"] = out["trade_date"]
    std["north_money_market"] = _to_numeric(out["north_money"]) if "north_money" in out.columns else None
    std["south_money_market"] = _to_numeric(out["south_money"]) if "south_money" in out.columns else None
    std["hgt_market"] = _to_numeric(out["hgt"]) if "hgt" in out.columns else None
    std["sgt_market"] = _to_numeric(out["sgt"]) if "sgt" in out.columns else None

    std = std.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    return std


# -----------------------------
# raw 装载
# -----------------------------
def load_raw_bundle(trade_date: str) -> dict[str, pd.DataFrame]:
    return {
        "daily": _std_daily(_load_raw_table("daily", trade_date), trade_date),
        "daily_basic": _std_daily_basic(_load_raw_table("daily_basic", trade_date), trade_date),
        "stock_basic": _std_stock_basic(_load_raw_table("stock_basic", trade_date), trade_date),
        "stk_limit": _std_stk_limit(_load_raw_table("stk_limit", trade_date), trade_date),
        "limit_list_d": _std_limit_list(_load_raw_table("limit_list_d", trade_date), trade_date),
        "limit_break_d": _std_limit_break(_load_raw_table("limit_break_d", trade_date), trade_date),
        "limit_up_tags": _std_limit_up_tags(_load_raw_table("limit_up_tags", trade_date), trade_date),
        "hot_boards": _std_hot_boards(_load_raw_table("hot_boards", trade_date), trade_date),
        "top_list": _std_top_list(_load_raw_table("top_list", trade_date), trade_date),
        "moneyflow_hsgt_market": _std_moneyflow_hsgt_market(_load_raw_table("moneyflow_hsgt", trade_date), trade_date),
        "namechange": _load_raw_table("namechange", trade_date),
    }


def build_master_table(bundle: dict[str, pd.DataFrame], trade_date: str) -> pd.DataFrame:
    daily = bundle.get("daily", pd.DataFrame())
    daily_basic = bundle.get("daily_basic", pd.DataFrame())
    stock_basic = bundle.get("stock_basic", pd.DataFrame())
    stk_limit = bundle.get("stk_limit", pd.DataFrame())
    limit_list_d = bundle.get("limit_list_d", pd.DataFrame())
    limit_break_d = bundle.get("limit_break_d", pd.DataFrame())
    limit_up_tags = bundle.get("limit_up_tags", pd.DataFrame())
    top_list = bundle.get("top_list", pd.DataFrame())

    if daily is None or daily.empty:
        return pd.DataFrame(columns=KEY_COLS)

    master = daily.copy()
    for tbl in [daily_basic, stock_basic, stk_limit, limit_list_d, limit_break_d, limit_up_tags, top_list]:
        master = _merge_left(master, tbl, on=KEY_COLS)

    # 补 board / industry
    if "industry" in master.columns and "board" not in master.columns:
        master["board"] = master["industry"]

    # 板块热度映射
    hot_boards = bundle.get("hot_boards", pd.DataFrame())
    if hot_boards is not None and not hot_boards.empty and "board" in master.columns:
        master["board"] = master["board"].astype(str).str.strip()
        master = master.merge(hot_boards, on="board", how="left")

    # 市场级资金流按 trade_date 广播
    money_market = bundle.get("moneyflow_hsgt_market", pd.DataFrame())
    if money_market is not None and not money_market.empty:
        master = master.merge(money_market, on="trade_date", how="left")

    # 统一关键字段
    master["trade_date"] = master["trade_date"].apply(lambda x: _normalize_trade_date(x, fallback=trade_date))
    master["ts_code"] = master["ts_code"].apply(_normalize_ts_code)
    master = master.dropna(subset=["trade_date", "ts_code"]).copy()
    master = master.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)

    return master


# -----------------------------
# Feature 构造
# -----------------------------
def build_features_base(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = df[KEY_COLS].copy()
    out["name"] = df.get("name")
    out["board"] = df.get("board")

    # 基础行情
    out["open"] = _to_numeric(df.get("open"))
    out["high"] = _to_numeric(df.get("high"))
    out["low"] = _to_numeric(df.get("low"))
    out["close"] = _to_numeric(df.get("close"))
    out["pct_chg"] = _to_numeric(df.get("pct_chg"))

    # daily.csv 没有 pre_close，反推
    out["pre_close_est"] = _to_numeric(df.get("pre_close_est"))

    # 收益与价格行为
    out["returns_1d"] = out["pct_chg"]
    pre = out["pre_close_est"]
    out["high_low_range"] = (out["high"] - out["low"]) / pre
    out["candle_body"] = (out["close"] - out["open"]) / pre
    out["gap_open"] = (out["open"] - pre) / pre

    # 流动性
    out["vol"] = _to_numeric(df.get("vol"))
    out["amount"] = _to_numeric(df.get("amount"))
    out["turnover_rate"] = _to_numeric(df.get("turnover_rate"))
    out["turnover_rate_f"] = _to_numeric(df.get("turnover_rate_f"))
    out["volume_ratio"] = _to_numeric(df.get("volume_ratio"))
    out["amihud_illiquidity"] = out["pct_chg"].abs() / out["amount"]

    # 波动骨架：当前仍无历史窗口，先保留占位
    out["volatility_5d"] = None
    out["volatility_10d"] = None
    out["volatility_20d"] = None
    out["atr"] = None
    out["downside_vol"] = None
    out["max_drawdown_20d"] = None
    out["tail_risk_score"] = None

    # 估值市值
    out["total_mv"] = _to_numeric(df.get("total_mv"))
    out["float_mv"] = _to_numeric(df.get("float_mv"))
    out["pe_ttm"] = _to_numeric(df.get("pe_ttm"))
    out["pb"] = _to_numeric(df.get("pb"))

    # 资金流/情绪
    out["north_money_market"] = _to_numeric(df.get("north_money_market"))
    out["south_money_market"] = _to_numeric(df.get("south_money_market"))
    out["hgt_market"] = _to_numeric(df.get("hgt_market"))
    out["sgt_market"] = _to_numeric(df.get("sgt_market"))
    out["market_regime"] = None

    # 题材热度
    out["hot_boards_score"] = _to_numeric(df.get("hot_boards_score"))
    out["board_crowding_rank"] = _to_numeric(df.get("board_crowding_rank"))

    # 龙虎榜/异动
    out["top_list_net_buy"] = _to_numeric(df.get("top_list_net_buy"))
    out["top_net_rate"] = _to_numeric(df.get("top_net_rate"))
    out["amount_rate"] = _to_numeric(df.get("amount_rate"))
    out["float_values"] = _to_numeric(df.get("float_values"))
    out["abnormal_volume"] = _to_numeric(df.get("amount_rate"))

    # 多周期占位
    out["ret_2d"] = None
    out["ret_5d"] = None
    out["ret_10d"] = None
    out["bid_ask_proxy"] = None
    out["spread_proxy"] = None

    return out


def build_features_limit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = df[KEY_COLS].copy()
    out["name"] = df.get("name")

    # limit_list_d 为主
    out["limit_type"] = df.get("limit_type")
    out["open_times"] = _to_numeric(df.get("open_times_limit"))
    out["fd_amount"] = _to_numeric(df.get("fd_amount_limit"))
    out["seal_amount"] = _to_numeric(df.get("seal_amount_limit"))
    out["first_seal_time"] = df.get("first_time_limit")
    out["last_seal_time"] = df.get("last_time_limit")

    # limit_break_d 补充
    out["open_times_break"] = _to_numeric(df.get("open_times_break"))
    out["fd_amount_break"] = _to_numeric(df.get("fd_amount_break"))
    out["seal_amount_break"] = _to_numeric(df.get("seal_amount_break"))
    out["first_time_break"] = df.get("first_time_break")
    out["last_time_break"] = df.get("last_time_break")

    # 涨跌停
    out["up_limit"] = _to_numeric(df.get("up_limit"))
    out["down_limit"] = _to_numeric(df.get("down_limit"))
    out["limit_up_tags"] = df.get("limit_up_tags")

    close = _to_numeric(df.get("close"))
    up_limit = _to_numeric(df.get("up_limit"))
    out["is_limit_up"] = (close >= up_limit).astype("float")

    # break_count 当前原始表没有直接字段，先做保守代理：
    # 有 limit_break_d 记录则视作 break_count>=1；否则为0
    break_exists = _to_numeric(df.get("open_times_break")).fillna(0.0)
    out["break_count_proxy"] = (break_exists > 0).astype(float)

    score = pd.Series(0.0, index=df.index)

    ot = out["open_times"].fillna(0.0)
    score += 1.0 / (1.0 + ot)

    sa = out["seal_amount"].fillna(0.0)
    max_sa = sa.max()
    if pd.notna(max_sa) and max_sa > 0:
        score += sa / max_sa

    bc = out["break_count_proxy"].fillna(0.0)
    score += 1.0 / (1.0 + bc)

    out["limit_up_strength"] = score.replace([float("inf"), -float("inf")], pd.NA)

    return out


def build_truth_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = df[KEY_COLS].copy()
    out["name"] = df.get("name")
    out["close"] = _to_numeric(df.get("close"))
    out["pre_close_est"] = _to_numeric(df.get("pre_close_est"))
    out["pct_chg"] = _to_numeric(df.get("pct_chg"))
    out["up_limit"] = _to_numeric(df.get("up_limit"))
    out["down_limit"] = _to_numeric(df.get("down_limit"))
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


def _raw_input_stats(bundle: dict[str, pd.DataFrame], trade_date: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stem, df in bundle.items():
        if stem == "moneyflow_hsgt_market":
            path = _raw_path("moneyflow_hsgt", trade_date)
        elif stem == "namechange":
            path = _raw_path("namechange", trade_date)
        else:
            path = _raw_path(stem, trade_date)

        out[stem] = {
            "path": str(path),
            "loaded": bool(df is not None and not df.empty),
            "rows": int(len(df)) if df is not None and not df.empty else 0,
            "columns": list(df.columns) if df is not None and not df.empty else [],
        }

    meta_path = RAW_DIR / f"_sync_meta_{trade_date}.json"
    out["_sync_meta"] = {
        "path": str(meta_path),
        "loaded": meta_path.exists(),
        "content": _safe_json_load(meta_path) if meta_path.exists() else {},
    }
    return out


def build_meta(
    trade_date: str,
    bundle: dict[str, pd.DataFrame],
    master_df: pd.DataFrame,
    base_df: pd.DataFrame,
    limit_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> dict[str, Any]:
    base_required = [
        "open", "high", "low", "close", "pre_close_est", "returns_1d",
        "vol", "amount", "turnover_rate", "volume_ratio",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "total_mv", "float_mv", "pe_ttm",
        "north_money_market", "south_money_market",
        "hot_boards_score", "board_crowding_rank",
    ]
    limit_required = [
        "is_limit_up", "limit_type", "open_times", "seal_amount",
        "first_seal_time", "last_seal_time", "break_count_proxy", "limit_up_strength",
    ]
    truth_required = ["close", "pre_close_est", "pct_chg"]

    sha = os.getenv("GITHUB_SHA", "")
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
        "commit_sha": sha,
        "fs_version": "v0.3-real-header-mapped",
        "richness_target": 0.75,
        "richness_estimate": richness_estimate,
        "raw_inputs": _raw_input_stats(bundle, trade_date),
        "master_rows": int(len(master_df)) if master_df is not None else 0,
        "tables": {
            "features_base": base_stats,
            "features_limit": limit_stats,
            "truth_close": truth_stats,
        },
        "notes": [
            "本版本已按真实上游表头修正 raw -> FS 的字段映射。",
            "daily.csv 无 pre_close，当前使用 close 和 pct_chg 反推 pre_close_est。",
            "moneyflow_hsgt 当前按市场级数据处理，并按 trade_date 广播到个股特征层。",
            "break_count 当前无直接原始字段，暂用 break_count_proxy 作为保守代理。",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trade_date = _resolve_trade_date(args.trade_date)

    if not trade_date:
        print("[build_market_fs] ERROR: 无法解析 trade_date，且 raw 目录中无可用 daily_*.csv")
        return 2

    bundle = load_raw_bundle(trade_date)
    daily_df = bundle.get("daily", pd.DataFrame())

    if daily_df.empty:
        print(f"[build_market_fs] ERROR: 缺少核心输入 daily_{trade_date}.csv")
        return 2

    master_df = build_master_table(bundle, trade_date)
    if master_df.empty:
        print(f"[build_market_fs] ERROR: 多源合并后 master 为空: trade_date={trade_date}")
        return 2

    base_df = build_features_base(master_df)
    limit_df = build_features_limit(master_df)
    truth_df = build_truth_close(master_df)
    meta = build_meta(
        trade_date=trade_date,
        bundle=bundle,
        master_df=master_df,
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
    print(f"[build_market_fs] master_rows={len(master_df)}")
    print("[build_market_fs] outputs:")
    print(f"  - data/market/features_base_{trade_date}.csv")
    print(f"  - data/market/features_limit_{trade_date}.csv")
    print(f"  - data/market/truth_close_{trade_date}.csv")
    print(f"  - data/market/_meta_{trade_date}.json")
    print(f"[build_market_fs] richness_estimate={meta.get('richness_estimate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
