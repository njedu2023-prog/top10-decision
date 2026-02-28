#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def fill_model_rule(df: pd.DataFrame) -> pd.Series:
    """
    只计算 p_fill_pred，不写文件
    """
    base = 0.35
    open_times = df.get("open_times", pd.Series([None] * len(df)))
    seal_amount = df.get("seal_amount", pd.Series([None] * len(df)))
    turnover = df.get("turnover_rate", pd.Series([None] * len(df)))

    p = []
    for i in range(len(df)):
        ot = _safe_float(open_times.iloc[i], default=float("nan"))
        sa = _safe_float(seal_amount.iloc[i], default=float("nan"))
        tr = _safe_float(turnover.iloc[i], default=float("nan"))

        pi = base
        if not pd.isna(ot):
            pi += min(max(ot, 0.0), 5.0) * 0.06
        if not pd.isna(sa):
            pi -= min(sa / 1e8, 5.0) * 0.05
        if not pd.isna(tr):
            pi += min(max(tr, 0.0), 20.0) * 0.005

        pi = max(0.02, min(0.98, pi))
        p.append(pi)

    return pd.Series(p, index=df.index, name="p_fill_pred")
