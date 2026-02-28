#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


COST_BP_DEFAULT = 8.0         # 成本估计 bp
RISK_PENALTY_OFF = 0.00
RISK_PENALTY_ON = 0.02


def cost_estimate_rule() -> float:
    return float(COST_BP_DEFAULT) / 10000.0


def risk_penalty_rule(regime: str) -> float:
    if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
        return float(RISK_PENALTY_ON)
    return float(RISK_PENALTY_OFF)
