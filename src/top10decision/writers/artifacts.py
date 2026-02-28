#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from top10decision.utils import to_jq_code
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal

from top10decision.writers.io_contract import (
    norm_ymd,
    ensure_cols,
    SIGNAL_LATEST,
    SIGNAL_DATED_FMT,
    WEIGHTS_LATEST,
    WEIGHTS_DATED_FMT,
    CANDIDATES_FMT,
)


def write_signals(latest_df: pd.DataFrame, trade_date: str) -> None:
    """
    signals IO 契约：
    - docs/signals/top10_latest.csv
    - docs/signals/top10_YYYYMMDD.csv
    """
    write_latest_signal(latest_df, out_path=str(SIGNAL_LATEST))
    td = norm_ymd(trade_date)
    if td:
        write_latest_signal(latest_df, out_path=SIGNAL_DATED_FMT.format(yyyymmdd=td))


def write_weights(weights_df: pd.DataFrame, exec_date: str) -> Tuple[str, str]:
    """
    weights IO 契约：
    - docs/weights/weights_latest.csv
    - docs/weights/weights_YYYYMMDD.csv
    """
    Path("docs/weights").mkdir(parents=True, exist_ok=True)

    latest_path = str(WEIGHTS_LATEST)
    dated = norm_ymd(exec_date)
    dated_path = WEIGHTS_DATED_FMT.format(yyyymmdd=dated) if dated else "docs/weights/weights_unknown.csv"

    weights_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(dated_path, index=False, encoding="utf-8-sig")
    return latest_path, dated_path


def write_candidates_snapshot(cand_df: pd.DataFrame, signal_date: str) -> str:
    """
    candidates IO 契约：
    - data/decision/decision_candidates_YYYYMMDD.csv
    """
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    sd = norm_ymd(signal_date) or "unknown"
    path = CANDIDATES_FMT.format(yyyymmdd=sd)
    cand_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_signal_df_for_joinquant(
    weights_df: pd.DataFrame,
    risk_budget: float,
    regime_name: str,
    trade_date: str,
    target_trade_date: str,
) -> pd.DataFrame:
    """
    兼容旧 joinquant 信号格式：
    - 只输出 weight>0 的目标行（候补不进入 signals）
    输出字段保持原契约：
    ["trade_date","target_trade_date","jq_code","target_weight","risk_budget","regime","reason"]
    """
    ensure_cols(weights_df, ["ts_code", "weight"])
    df = weights_df.copy()
    df = df[df["weight"].astype(float) > 0].copy()

    df["jq_code"] = df["ts_code"].apply(to_jq_code)
    df["trade_date"] = norm_ymd(trade_date)
    df["target_trade_date"] = norm_ymd(target_trade_date)
    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)
    df["reason"] = "P0_EV_weight"
    df["target_weight"] = df["weight"].astype(float)

    return df[["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]].copy()
