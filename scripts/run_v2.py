#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 pipeline runner

Pipeline (P0):
1) Ingest: load latest pred CSV (data/pred/pred_top10_latest.csv)
2) Regime: decide regime + risk_budget (P0 fixed)
3) Guardrails: stop_trading gate (P0 allow)
4) Strategy Router: (P0 passthrough)
5) Position: allocate weights (equal weight * risk_budget)
6) Output: write docs/signals/top10_latest.csv (for GitHub Pages)
7) Report:
   - docs/reports/daily_latest.md (简报)
   - docs/reports/top10_latest.md (人类可读 Top10 名单)

Notes:
- This script assumes you already ran: python scripts/sync_from_a_top10.py
- GitHub Pages should point to /docs as source
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
from top10decision.reporting.daily_report import write_daily_report, write_human_top10_list


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}. 现有字段：{list(df.columns)}")


def build_signal_df(pred_df: pd.DataFrame, risk_budget: float, regime_name: str) -> pd.DataFrame:
    """
    Convert pred_df -> signal_df (for JoinQuant execution)
    Required input columns: ts_code
    Optional columns: trade_date, target_trade_date
    """
    df = pred_df.copy()

    # 固定取前10（你已确认每日固定10支；但这里仍做保护）
    df = df.head(10).copy()

    _ensure_cols(df, ["ts_code"])

    df["jq_code"] = df["ts_code"].apply(to_jq_code)

    # 日期字段保留（有则保留，无则置空）
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


def main() -> int:
    # 1) ingest
    pred_df = load_latest_pred()

    # 2) regime
    reg = simple_regime(pred_df)

    # 3) guardrails
    gr = guardrails(pred_df)
    if getattr(gr, "stop_trading", False):
        empty = pd.DataFrame(
            columns=["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
        )
        empty_out = write_latest_signal(empty, out_path="docs/signals/top10_latest.csv")

        rep_out = Path("docs/reports/daily_latest.md")
        rep_out.parent.mkdir(parents=True, exist_ok=True)
        rep_out.write_text(
            f"# Daily Decision Report (latest)\n\n"
            f"- stop_trading: True\n"
            f"- reason: {getattr(gr, 'reason', '')}\n",
            encoding="utf-8",
        )

        # 也写一份人类可读名单（空）
        write_human_top10_list(empty, out_path="docs/reports/top10_latest.md")

        print(f"[run_v2] STOP_TRADING -> wrote {empty_out} and reports")
        return 0

    # 4) strategies router (P0 passthrough)
    routed_df = score_router(pred_df)

    # 5) build signal (include position sizing)
    signal_df = build_signal_df(
        pred_df=routed_df,
        risk_budget=float(getattr(reg, "risk_budget", 1.0)),
        regime_name=str(getattr(reg, "regime", "RISK_ON")),
    )

    # 6) write latest signal for GitHub Pages
    out_sig = write_latest_signal(signal_df, out_path="docs/signals/top10_latest.csv")

    # 7) reports
    out_daily = write_daily_report(signal_df, out_path="docs/reports/daily_latest.md")

    # 人类可读 Top10：把 pred 原字段和 signal 字段合并展示
    # 用 jq_code 作为连接键（routed_df 里可能没 jq_code，所以先补一列）
    show_pred = routed_df.head(10).copy()
    if "jq_code" not in show_pred.columns and "ts_code" in show_pred.columns:
        show_pred["jq_code"] = show_pred["ts_code"].apply(to_jq_code)

    merged = pd.merge(show_pred, signal_df, on=["jq_code"], how="left", suffixes=("", "_sig"))
    out_human = write_human_top10_list(merged, out_path="docs/reports/top10_latest.md")

    print(f"[run_v2] wrote signal: {out_sig}")
    print(f"[run_v2] wrote report: {out_daily}")
    print(f"[run_v2] wrote human list: {out_human}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
