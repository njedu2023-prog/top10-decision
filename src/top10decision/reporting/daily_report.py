# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _pick_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]


def _md_table(df: pd.DataFrame, cols: List[str]) -> str:
    if df is None or df.empty:
        return "_（空）_\n"
    if not cols:
        return "_（无可展示字段）_\n"

    lines = []
    lines.append("| " + " | ".join(cols) + " |\n")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |\n")
    return "".join(lines)


def _first_non_empty(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    return str(s.iloc[0])


def write_daily_report_human(
    merged_df: pd.DataFrame,
    out_path: str,
    title: str = "Daily Decision Report",
    extra_notes: Optional[str] = None,
) -> Path:
    """
    人类可读日报：以 a-top10 原信息为主，并附带 V2 决策字段
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _first_non_empty(merged_df, "trade_date")
    target_trade_date = _first_non_empty(merged_df, "target_trade_date")

    preferred = [
        "rank",
        "ts_code",
        "name",
        "board",
        "Probability",
        "StrengthScore",
        "ThemeBoost",
        "st_flag",
        "st_penalty",
        "score",
        "jq_code",
        "target_weight",
        "risk_budget",
        "regime",
        "reason",
    ]
    show_cols = _pick_cols(merged_df, preferred)

    lines = []
    lines.append(f"# {title}\n\n")
    lines.append(f"- trade_date（信号生成日）: **{trade_date}**\n")
    lines.append(f"- target_trade_date（执行交易日）: **{target_trade_date}**\n\n")
    lines.append("本页为**人类可读**版本：原始信息（ts_code/name/board/Probability…）+ 决策字段（target_weight/regime…）。\n\n")
    if extra_notes:
        lines.append(extra_notes.strip() + "\n\n")
    lines.append(_md_table(merged_df, show_cols))

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def write_exec_check_report(
    signal_df: pd.DataFrame,
    out_path: str,
) -> Path:
    """
    执行核验页：只展示聚宽执行字段
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _first_non_empty(signal_df, "trade_date")
    target_trade_date = _first_non_empty(signal_df, "target_trade_date")

    preferred = ["jq_code", "target_weight", "risk_budget", "regime", "reason"]
    show_cols = _pick_cols(signal_df, preferred)

    lines = []
    lines.append("# Exec Check\n\n")
    lines.append(f"- trade_date: **{trade_date}**\n")
    lines.append(f"- target_trade_date: **{target_trade_date}**\n\n")
    lines.append(_md_table(signal_df, show_cols))

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
