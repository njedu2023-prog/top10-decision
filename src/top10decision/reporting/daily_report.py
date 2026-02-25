# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _pick_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]


def write_daily_report(signal_df: pd.DataFrame, out_path: str = "docs/reports/daily_latest.md") -> Path:
    """
    简报：主要用于“今天系统输出是否正常”
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Daily Decision Report (latest)\n\n")
    lines.append("本页用于快速核验：信号是否生成、数量是否为 10、权重是否合理。\n\n")

    # 只做简表，避免冗长
    lines.append("| jq_code | target_weight | regime | risk_budget | reason |\n")
    lines.append("|---|---:|---|---:|---|\n")
    for _, r in signal_df.iterrows():
        lines.append(
            f"| {r.get('jq_code','')} | {r.get('target_weight','')} | {r.get('regime','')} | {r.get('risk_budget','')} | {r.get('reason','')} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def write_human_top10_list(
    merged_df: pd.DataFrame,
    out_path: str = "docs/reports/top10_latest.md",
) -> Path:
    """
    人类可读 Top10 名单（用于审阅/复盘/展示）
    merged_df：建议包含 pred 原字段 + signal 字段
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 优先展示的字段（存在则展示，不存在就跳过）
    preferred = [
        "trade_date",
        "target_trade_date",
        "rank",
        "ts_code",
        "jq_code",
        "name",
        "board",
        "Probability",
        "StrengthScore",
        "ThemeBoost",
        "st_flag",
        "st_penalty",
        "score",
        "target_weight",
        "risk_budget",
        "regime",
        "reason",
    ]
    show_cols = _pick_cols(merged_df, preferred)

    # 生成 markdown 表
    lines = []
    lines.append("# Top10 执行名单（latest）\n\n")
    lines.append("说明：本名单由 top10-decision 生成；CSV 用于聚宽执行，本页用于人类审阅。\n\n")

    # 表头
    lines.append("| " + " | ".join(show_cols) + " |\n")
    lines.append("|" + "|".join(["---"] * len(show_cols)) + "|\n")

    for _, row in merged_df.iterrows():
        vals = []
        for c in show_cols:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
