#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 runner

Outputs:
- docs/signals/top10_latest.csv          (聚宽执行)
- docs/signals/top10_YYYYMMDD.csv        (按 trade_date 归档，永远新增 ✅新增)
- docs/reports/daily_latest.md           (人类可读，覆盖更新：表格风格如截图)
- docs/reports/daily_YYYYMMDD.md         (人类可读，按 trade_date 归档)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.utils import to_jq_code
from top10decision.position.allocator import allocate_equal_weight
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


def _norm_ymd(v) -> str:
    """
    把 trade_date 规范成 'YYYYMMDD'（去掉 .0 / 科学计数法 / 空值）
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
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


def build_signal_df(pred_df: pd.DataFrame, risk_budget: float, regime_name: str) -> pd.DataFrame:
    """
    聚宽执行用信号（保留 jq_code 等字段）
    """
    df = pred_df.head(10).copy()
    _ensure_cols(df, ["ts_code"])

    df["jq_code"] = df["ts_code"].apply(to_jq_code)

    if "trade_date" not in df.columns:
        df["trade_date"] = ""
    if "target_trade_date" not in df.columns:
        df["target_trade_date"] = ""

    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)
    df["reason"] = "P0_equal_weight"

    df = allocate_equal_weight(df, risk_budget=float(risk_budget))
    if "target_weight" not in df.columns:
        raise ValueError("allocator 未产生 target_weight，请检查 src/top10decision/position/allocator.py")

    return df[
        ["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
    ].copy()


def _get_first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    return _norm_ymd(s.iloc[0]) if col in ("trade_date", "target_trade_date") else str(s.iloc[0])


def _fmt_num(x, nd=6):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "" if x is None else str(x)


def _write_human_report(pred_top10: pd.DataFrame, out_path: str, title: str, stop_note: str | None = None) -> None:
    """
    生成你截图那种“人类可读 Top10 表格”
    列：排名 / 代码 / 股票 / Probability / 强度得分 / 题材加成 / 板块
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _get_first_value(pred_top10, "trade_date")
    target_trade_date = _get_first_value(pred_top10, "target_trade_date")

    lines: list[str] = []
    lines.append(f"# {title}\n\n")
    lines.append(f"- trade_date（信号生成日）: **{trade_date if trade_date else '未知'}**\n")
    lines.append(f"- target_trade_date（执行交易日）: **{target_trade_date if target_trade_date else '未知/未填'}**\n\n")

    if stop_note:
        lines.append(f"**停手：{stop_note}**\n\n")

    lines.append("| 排名 | 代码 | 股票 | Probability | 强度得分 | 题材加成 | 板块 |\n")
    lines.append("|---:|---|---|---:|---:|---:|---|\n")

    d = pred_top10.head(10).copy()

    for _, r in d.iterrows():
        rank = r.get("rank", "")
        ts_code = r.get("ts_code", "")
        name = r.get("name", "")
        prob = r.get("Probability", r.get("prob", r.get("probability", "")))
        strength = r.get("StrengthScore", r.get("strength", ""))
        theme = r.get("ThemeBoost", r.get("theme", ""))
        board = r.get("board", r.get("industry", ""))

        lines.append(
            f"| {rank} | {ts_code} | {name} | {_fmt_num(prob, 6)} | {_fmt_num(strength, 4)} | {_fmt_num(theme, 6)} | {board} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_signals(latest_df: pd.DataFrame, trade_date: str) -> None:
    """
    写两份信号：
    - latest：docs/signals/top10_latest.csv（覆盖）
    - dated：docs/signals/top10_YYYYMMDD.csv（永远新增）
    """
    # 1) latest（覆盖）
    write_latest_signal(latest_df, out_path="docs/signals/top10_latest.csv")

    # 2) dated（永远新增）
    td = _norm_ymd(trade_date)
    if td:
        dated_path = f"docs/signals/top10_{td}.csv"
        write_latest_signal(latest_df, out_path=dated_path)


def main() -> int:
    pred_df = load_latest_pred()
    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    routed_df = score_router(pred_df).head(10).copy()

    trade_date = _get_first_value(routed_df, "trade_date")
    dated_name = f"daily_{trade_date}.md" if trade_date else "daily_unknown.md"

    if getattr(gr, "stop_trading", False):
        empty = pd.DataFrame(
            columns=["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
        )

        # ✅ stop 分支：也落盘 dated（空表），保证可追溯
        _write_signals(empty, trade_date=trade_date)

        stop_note = getattr(gr, "reason", "STOP_TRADING")
        _write_human_report(
            routed_df,
            out_path="docs/reports/daily_latest.md",
            title="Daily Decision Report (latest)",
            stop_note=stop_note,
        )
        _write_human_report(
            routed_df,
            out_path=f"docs/reports/{dated_name}",
            title=f"Daily Decision Report ({trade_date})" if trade_date else "Daily Decision Report (unknown)",
            stop_note=stop_note,
        )
        return 0

    signal_df = build_signal_df(
        pred_df=routed_df,
        risk_budget=float(getattr(reg, "risk_budget", 1.0)),
        regime_name=str(getattr(reg, "regime", "RISK_ON")),
    )

    trade_date2 = _get_first_value(signal_df, "trade_date") or trade_date
    dated_name2 = f"daily_{trade_date2}.md" if trade_date2 else dated_name

    # ✅ 正常分支：写 latest + dated
    _write_signals(signal_df, trade_date=trade_date2)

    _write_human_report(
        routed_df,
        out_path="docs/reports/daily_latest.md",
        title="Daily Decision Report (latest)",
        stop_note=None,
    )
    _write_human_report(
        routed_df,
        out_path=f"docs/reports/{dated_name2}",
        title=f"Daily Decision Report ({trade_date2})" if trade_date2 else "Daily Decision Report (unknown)",
        stop_note=None,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
