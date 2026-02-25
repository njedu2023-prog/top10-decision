# -*- coding: utf-8 -*-

from dataclasses import dataclass
import pandas as pd


@dataclass
class GuardrailResult:
    stop_trading: bool
    reason: str


def guardrails(df: pd.DataFrame) -> GuardrailResult:
    # P0：永远允许交易，后续再接入回撤/亏损/停手机制
    return GuardrailResult(stop_trading=False, reason="P0_none")
