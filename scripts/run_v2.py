#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 runner

Outputs:
- docs/signals/top10_latest.csv          (聚宽执行)
- docs/reports/daily_latest.md           (人类可读，覆盖更新，不含 jq_code)
- docs/reports/daily_YYYYMMDD.md         (人类可读，按 trade_date 归档)
"""

from __future__ import annotations

import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.utils import to_jq_code
from top10decision.position.allocator import allocate_equal_weight
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal
from top10decision.reporting.daily_report import write_daily_report_human


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


def build_signal_df(pred_df: pd.DataFrame, risk_budget: float, regime_name: str) -> pd.DataFrame:
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


def _get_trade_date(df: pd.DataFrame) -> str:
    if df is None or df.empty or "trade_date" not in df.columns:
        return ""
    s = df["trade_date"].dropna()
    return "" if s.empty else str(s.iloc[0])


def main() -> int:
    pred_df = load_latest_pred()
    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    routed_df = score_router(pred_df).head(10).copy()

    # 停手：仍输出空 signal，日报也写（含原因）
    if getattr(gr, "stop_trading", False):
        empty = pd.DataFrame(columns=["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"])
        write_latest_signal(empty, out_path="docs/signals/top10_latest.csv")
        # 日报写一份最小说明
        merged = routed_df.copy()
        merged["trade_date"] = merged.get("trade_date", "")
        merged["target_trade_date"] = merged.get("target_trade_date", "")
        merged["target_weight"] = ""
        merged["risk_budget"] = ""
        merged["regime"] = "STOP"
        merged["reason"] = getattr(gr, "reason", "STOP_TRADING")
        write_daily_report_human(merged, out_path="docs/reports/daily_latest.md", title="Daily Decision Report (latest)")
        return 0

    signal_df = build_signal_df(
        pred_df=routed_df,
        risk_budget=float(getattr(reg, "risk_budget", 1.0)),
        regime_name=str(getattr(reg, "regime", "RISK_ON")),
    )

    # 1) 写聚宽执行 CSV（包含 jq_code）
    write_latest_signal(signal_df, out_path="docs/signals/top10_latest.csv")

    # 2) 生成日报（不含 jq_code）：用 ts_code 作为 merge 键更直观
    merged = pd.merge(
        routed_df,
        signal_df.drop(columns=["jq_code"]),
        on=["trade_date", "target_trade_date"],
        how="left",
        suffixes=("", "_sig"),
    )

    # 如果 routed_df 里没有 trade_date/target_trade_date，按当前 signal 补齐
    if "trade_date" not in merged.columns or merged["trade_date"].isna().all():
        merged["trade_date"] = _get_trade_date(signal_df)
    if "target_trade_date" not in merged.columns:
        merged["target_trade_date"] = routed_df.get("target_trade_date", "")

    # 3) 写覆盖版 + 归档版
    trade_date = _get_trade_date(signal_df)
    write_daily_report_human(merged, out_path="docs/reports/daily_latest.md", title="Daily Decision Report (latest)")

    dated_name = f"daily_{trade_date}.md" if trade_date else "daily_unknown.md"
    write_daily_report_human(merged, out_path=f"docs/reports/{dated_name}", title=f"Daily Decision Report ({trade_date})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
