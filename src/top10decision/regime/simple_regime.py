# -*- coding: utf-8 -*-

from dataclasses import dataclass
import pandas as pd


@dataclass
class RegimeResult:
    regime: str
    risk_budget: float
    reason: str


def simple_regime(df: pd.DataFrame) -> RegimeResult:
    # P0：永远 RISK_ON，后续再用情绪/宽度/趋势替换
    return RegimeResult(regime="RISK_ON", risk_budget=1.0, reason="P0_fixed")
