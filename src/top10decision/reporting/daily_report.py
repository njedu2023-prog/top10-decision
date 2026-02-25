# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


def _first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    return "" if s.empty else str(s.iloc[0])


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


def write_daily_report_human(
    merged_df: pd.DataFrame,
    out_path: str = "docs/reports/daily_latest.md",
    title: str = "Daily Decision Report (latest)",
) -> Path:
    """
    ✅ 人类可读日报（不出现 jq_code）
    - 以 a-top10 原信息为主：ts_code / name / board / Probability / StrengthScore / ThemeBoost ...
    - 附带 V2 决策字段：target_weight / regime / risk_budget / reason
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _first_value(merged_df, "trade_date")
    target_trade_date = _first_value(merged_df, "target_trade_date")

    # 你要求：日报不要聚宽识别字段（jq_code）
    preferred = [
        "trade_date",
        "target_trade_date",
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
        # 决策字段（人类也能读）
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
    lines.append("本页为人类可读：原始信息 + 决策字段（不含 jq_code）。\n\n")
    lines.append(_md_table(merged_df, show_cols))

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
