# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    return "" if s.empty else str(s.iloc[0])


def _fmt_num(x, nd=6) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "" if x is None else str(x)


def write_daily_report_human(
    merged_df: pd.DataFrame,
    out_path: str = "docs/reports/daily_latest.md",
    title: str = "Daily Decision Report (latest)",
) -> Path:
    """
    ✅ 人类可读日报（目标：表格内容像你截图那样）
    - 不出现 jq_code
    - 表格列固定为：
        排名 / 代码 / 股票 / Probability / 强度得分 / 题材加成 / 板块
    - 头部保留日期信息：trade_date / target_trade_date
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _first_value(merged_df, "trade_date")
    target_trade_date = _first_value(merged_df, "target_trade_date")

    # 兼容字段名：Probability / StrengthScore / ThemeBoost / board / name / ts_code / rank
    # 如果某些字段缺失，则输出为空，但不报错
    d = merged_df.head(10).copy()

    lines: list[str] = []
    lines.append(f"# {title}\n\n")
    lines.append(f"- trade_date（信号生成日）: **{trade_date if trade_date else '未知'}**\n")
    lines.append(f"- target_trade_date（执行交易日）: **{target_trade_date if target_trade_date else '未知/未填'}**\n\n")

    # 表格（中文列名，与你截图一致）
    lines.append("| 排名 | 代码 | 股票 | Probability | 强度得分 | 题材加成 | 板块 |\n")
    lines.append("|---:|---|---|---:|---:|---:|---|\n")

    for _, r in d.iterrows():
        rank = r.get("rank", "")
        ts_code = r.get("ts_code", "")
        name = r.get("name", "")
        board = r.get("board", "")

        prob = r.get("Probability", r.get("prob", r.get("probability", "")))
        strength = r.get("StrengthScore", r.get("strength", ""))
        theme = r.get("ThemeBoost", r.get("theme", ""))

        lines.append(
            f"| {rank} | {ts_code} | {name} | {_fmt_num(prob, 6)} | {_fmt_num(strength, 4)} | {_fmt_num(theme, 6)} | {board} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
