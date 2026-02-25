# -*- coding: utf-8 -*-

import pandas as pd

from top10decision.utils import to_jq_code


def decision_p0(df: pd.DataFrame) -> pd.DataFrame:
    # 只取前10（你说固定10支）
    df = df.head(10).copy()

    df["jq_code"] = df["ts_code"].apply(to_jq_code)

    df["risk_budget"] = 1.0
    df["regime"] = "RISK_ON"

    # 等权
    df["target_weight"] = 1.0 / len(df)

    # 如果原文件没有 trade_date/target_trade_date，就用空值占位
    if "trade_date" not in df.columns:
        df["trade_date"] = ""
    if "target_trade_date" not in df.columns:
        df["target_trade_date"] = ""

    df["reason"] = "P0_equal_weight"
    return df[
        ["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
    ]
