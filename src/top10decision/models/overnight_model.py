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


def overnight_model_rule(df: pd.DataFrame, regime: str) -> pd.Series:
    """
    只计算 e_ret_pred，不写文件
    """
    prob = df.get("Probability", df.get("prob", df.get("probability", pd.Series([None] * len(df)))))
    strength = df.get("StrengthScore", df.get("strength", pd.Series([None] * len(df))))
    theme = df.get("ThemeBoost", df.get("theme", pd.Series([None] * len(df))))

    e = []
    for i in range(len(df)):
        p = _safe_float(prob.iloc[i], default=0.3)
        s = _safe_float(strength.iloc[i], default=0.0)
        t = _safe_float(theme.iloc[i], default=0.0)

        ei = (max(0.0, min(1.0, p)) - 0.2) * 0.03
        ei += max(-2.0, min(10.0, s)) * 0.001
        ei += max(-1.0, min(3.0, t)) * 0.003

        if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
            ei -= 0.006

        ei = max(-0.05, min(0.08, ei))
        e.append(ei)

    return pd.Series(e, index=df.index, name="e_ret_pred")
