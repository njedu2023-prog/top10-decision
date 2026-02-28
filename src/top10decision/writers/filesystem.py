#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd

from top10decision.writers.io_contract import norm_ymd


def ensure_dirs() -> None:
    Path("data/pred").mkdir(parents=True, exist_ok=True)
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    Path("docs/signals").mkdir(parents=True, exist_ok=True)
    Path("docs/reports").mkdir(parents=True, exist_ok=True)
    Path("docs/weights").mkdir(parents=True, exist_ok=True)
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    Path("outputs/learning").mkdir(parents=True, exist_ok=True)


def ensure_execution_table(exec_date: str) -> str:
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    ed = norm_ymd(exec_date) or "unknown"
    path = f"data/decision/decision_execution_{ed}.csv"

    if not Path(path).exists():
        empty = pd.DataFrame(
            columns=[
                "exec_date",
                "ts_code",
                "jq_code",
                "filled_flag",
                "buy_time",
                "buy_price",
                "fail_reason",
                "buy_slippage_bp",
            ]
        )
        empty.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def ensure_learning_table() -> str:
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    path = "data/decision/decision_learning.csv"
    if not Path(path).exists():
        empty = pd.DataFrame(
            columns=[
                "signal_date",
                "exec_date",
                "exit_date",
                "ts_code",
                "jq_code",
                "filled_flag",
                "buy_price",
                "sell_price",
                "ret_exec",
                "p_fill_pred",
                "e_ret_pred",
                "ev_pred",
            ]
        )
        empty.to_csv(path, index=False, encoding="utf-8-sig")
    return path
