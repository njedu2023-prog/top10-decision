#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 pipeline runner

Outputs:
- docs/signals/top10_latest.csv                      (给聚宽执行)
- docs/reports/daily_latest.md                       (覆盖版：人类日报 latest)
- docs/reports/daily_YYYYMMDD.md                     (归档版：人类日报 dated)
- docs/reports/exec_check_latest.md                  (覆盖版：执行核验)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.utils import to_jq_code
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.position.allocator import allocate_equal_weight
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal
from top10decision.reporting.daily_report import write_daily_report_human, write_exec_check_report


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}. 现有字段：{list(df.columns)}")


def build_signal_df(pred_df: pd.DataFrame, risk_budget: float, regime_name: str) -> pd.DataFrame:
    df = pred_df.copy().head(10).copy()
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
        raise ValueError("Position allocator 未产生 target_weight，请检查 src/top10decision/position/allocator.py")

    out_cols = [
        "trade_date",
        "target_trade_date",
        "jq_code",
        "target_weight",
        "risk_budget",
        "regime",
        "reason",
    ]
    return df[out_cols].copy()


def _get_trade_date(df: pd.DataFrame) -> str:
    if df is None or df.empty or "trade_date" not in df.columns:
        return ""
    s = df["trade_date"].dropna()
    return "" if s.empty else str(s.iloc[0])


def main() -> int:
    pred_df = load_latest_pred()

    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    # router
    routed_df = score_router(pred_df).head(10).copy()

    # build signal
    if getattr(gr, "stop_trading", False):
        empty = pd.DataFrame(
            columns=["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
        )
        write_latest_signal(empty, out_path="docs/signals/top10_latest.csv")
        write_exec_check_report(empty, out_path="docs/reports/exec_check_latest.md")

        rep_path = Path("docs/reports/daily_latest.md")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(
            f"# Daily Decision Report (latest)\n\n"
            f"- stop_trading: True\n"
            f"- reason: {getattr(gr, 'reason', '')}\n",
            encoding="utf-8",
        )
        return 0

    signal_df = build_signal_df(
        pred_df=routed_df,
        risk_budget=float(getattr(reg, "risk_budget", 1.0)),
        regime_name=str(getattr(reg, "regime", "RISK_ON")),
    )

    # write signal
    write_latest_signal(signal_df, out_path="docs/signals/top10_latest.csv")

    # merge pred+signal for human report
    if "jq_code" not in routed_df.columns:
        routed_df["jq_code"] = routed_df["ts_code"].apply(to_jq_code)
    merged = pd.merge(routed_df, signal_df, on="jq_code", how="left", suffixes=("", "_sig"))

    # report paths
    trade_date = _get_trade_date(signal_df)  # 通常是 8位 YYYYMMDD
    dated_name = f"daily_{trade_date}.md" if trade_date else "daily_unknown.md"

    write_daily_report_human(
        merged,
        out_path="docs/reports/daily_latest.md",
        title="Daily Decision Report (latest)",
    )
    write_daily_report_human(
        merged,
        out_path=f"docs/reports/{dated_name}",
        title=f"Daily Decision Report ({trade_date})" if trade_date else "Daily Decision Report (unknown)",
    )

    write_exec_check_report(signal_df, out_path="docs/reports/exec_check_latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
