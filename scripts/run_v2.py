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
7) Report: write docs/reports/daily_latest.md

Notes:
- This script assumes you already ran: python scripts/sync_from_a_top10.py
- GitHub Pages should point to /docs as source
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.utils import to_jq_code
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.position.allocator import allocate_equal_weight
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal
from top10decision.reporting.daily_report import write_daily_report


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

    # code mapping
    df["jq_code"] = df["ts_code"].apply(to_jq_code)

    # 补齐日期字段
    if "trade_date" not in df.columns:
        df["trade_date"] = ""
    if "target_trade_date" not in df.columns:
        df["target_trade_date"] = ""

    # regime fields
    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)

    # 先留 reason，后面策略路由/风控可覆盖
    df["reason"] = "P0_equal_weight"

    # allocate weights
    df = allocate_equal_weight(df, risk_budget=float(risk_budget))

    # 输出契约（聚宽拉取的最小字段集合）
    out_cols = [
        "trade_date",
        "target_trade_date",
        "jq_code",
        "target_weight",
        "risk_budget",
        "regime",
        "reason",
    ]
    # 保险：万一 allocator 没写 target_weight
    if "target_weight" not in df.columns:
        raise ValueError("Position allocator 未产生 target_weight，请检查 src/top10decision/position/allocator.py")

    return df[out_cols].copy()


def main() -> int:
    # 1) ingest
    pred_df = load_latest_pred()

    # 2) regime
    reg = simple_regime(pred_df)

    # 3) guardrails
    gr = guardrails(pred_df)
    if getattr(gr, "stop_trading", False):
        # 停手：输出一个空信号（但仍落盘，保证系统“有输出”）
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
        print(f"[run_v2] STOP_TRADING -> wrote {empty_out} and {rep_out}")
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

    # 7) write report
    out_rep = write_daily_report(signal_df, out_path="docs/reports/daily_latest.md")

    print(f"[run_v2] wrote signal: {out_sig}")
    print(f"[run_v2] wrote report: {out_rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
