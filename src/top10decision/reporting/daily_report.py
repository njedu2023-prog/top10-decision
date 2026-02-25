# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


def write_daily_report(df: pd.DataFrame, out_path="docs/reports/daily_latest.md") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Daily Decision Report (latest)\n")
    lines.append("| jq_code | target_weight |\n")
    lines.append("|--------|---------------|\n")
    for _, row in df.iterrows():
        lines.append(f"| {row.get('jq_code','')} | {row.get('target_weight','')} |\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
