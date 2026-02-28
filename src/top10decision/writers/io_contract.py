#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd


# =========================
# IO 契约常量（不允许改动含义）
# =========================

TOPK_DEFAULT = 100
TOPN_DEFAULT = 10

W_MAX_DEFAULT = 0.12
THEME_CAP_DEFAULT = 0.35
GROSS_CAP_DEFAULT = 1.00


# =========================
# 通用工具（保持原行为）
# =========================

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


def norm_ymd(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    s = str(v).strip()
    if not s:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 8 and s.isdigit():
        return s
    try:
        i = int(float(s))
        s2 = str(i)
        return s2 if (len(s2) == 8 and s2.isdigit()) else s2
    except Exception:
        return s


def get_first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    if col in ("trade_date", "target_trade_date", "exec_date", "exit_date", "signal_date", "verify_date"):
        return norm_ymd(s.iloc[0])
    return str(s.iloc[0])


def fmt_num(x, nd=6):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "" if x is None else str(x)


def choose_exec_date(trade_date: str, target_trade_date: str) -> str:
    td = norm_ymd(trade_date)
    ttd = norm_ymd(target_trade_date)
    return ttd or td


# =========================
# 固定路径（IO 契约：绝对不改）
# =========================

# 输入快照
PRED_SNAPSHOT_PATH = Path("data/pred/pred_source_latest.csv")

# 输出：signals
SIGNAL_LATEST = Path("docs/signals/top10_latest.csv")
SIGNAL_DATED_FMT = "docs/signals/top10_{yyyymmdd}.csv"

# 输出：weights
WEIGHTS_LATEST = Path("docs/weights/weights_latest.csv")
WEIGHTS_DATED_FMT = "docs/weights/weights_{yyyymmdd}.csv"

# 输出：decision candidates
CANDIDATES_FMT = "data/decision/decision_candidates_{yyyymmdd}.csv"

# 输出：execution table
EXECUTION_FMT = "data/decision/decision_execution_{yyyymmdd}.csv"

# 输出：learning table
LEARNING_PATH = Path("data/decision/decision_learning.csv")

# 输出：report / eval
REPORT_FMT = "outputs/decision/decision_report_{yyyymmdd}.md"
EVAL_FMT = "outputs/decision/eval_{yyyymmdd}.json"
