#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Labels（真实对照打标）

职责：
- 构建 RealPremiumRet(2→3) 标签：
    real_premium_ret = close(T+2) / close(T+1) - 1
  其中：
    trade_date = T+1（第2日）
    next_trade_date = T+2（第3日）

输入：
- close_df：包含至少 (trade_date, ts_code, close) 的真实行情表（可多天、多股票）

输出：
- label_df：包含 (trade_date, next_trade_date, ts_code, close_2, close_3, real_premium_ret)
- meta：包含 pending/原因等（便于上层决定是否训练/评估）

注意：
- 本模块不做任何“预测收益”，只做真实对照标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .schemas import CloseLabelSchema


@dataclass(frozen=True)
class LabelBuildMeta:
    pending: bool
    reason: str
    trade_date: str
    next_trade_date: Optional[str]


def _to_yyyymmdd(s: str) -> str:
    s = str(s).strip()
    # 允许 "YYYY-MM-DD" -> "YYYYMMDD"
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


def _ensure_trade_date_str(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).map(_to_yyyymmdd)


def _infer_next_trade_date(sorted_dates: list, trade_date: str) -> Optional[str]:
    """
    在已排序的交易日列表里，找到 trade_date 的下一个交易日。
    若 trade_date 不在列表或已是最后一天，返回 None。
    """
    try:
        i = sorted_dates.index(trade_date)
    except ValueError:
        return None
    if i + 1 >= len(sorted_dates):
        return None
    return sorted_dates[i + 1]


def build_premium_labels(
    close_df: pd.DataFrame,
    trade_date: str,
) -> Tuple[pd.DataFrame, LabelBuildMeta]:
    """
    构建指定 trade_date（第2日）的 Premium 标签。

    参数：
    - close_df：真实收盘价表（多天多股票）
    - trade_date：YYYYMMDD（第2日）

    返回：
    - labels_df：每个 ts_code 一行
    - meta：是否 pending/原因
    """
    trade_date = _to_yyyymmdd(trade_date)

    if close_df is None or close_df.empty:
        meta = LabelBuildMeta(
            pending=True,
            reason="close_df 为空：没有真实收盘价数据，无法打标",
            trade_date=trade_date,
            next_trade_date=None,
        )
        return pd.DataFrame(), meta

    # 解析所需列（兼容别名）
    req_map, _ = CloseLabelSchema.resolve(close_df.columns)
    col_date = req_map["trade_date"]
    col_code = req_map["ts_code"]
    col_close = req_map["close"]

    df = close_df.copy()
    df[col_date] = _ensure_trade_date_str(df, col_date)
    df[col_code] = df[col_code].astype(str).str.strip()
    # close 可能是字符串，强制转 float
    df[col_close] = pd.to_numeric(df[col_close], errors="coerce")

    # 取交易日序列，推断 next_trade_date
    dates = sorted([d for d in df[col_date].dropna().unique() if str(d).isdigit() and len(str(d)) == 8])
    next_trade_date = _infer_next_trade_date(dates, trade_date)

    if next_trade_date is None:
        meta = LabelBuildMeta(
            pending=True,
            reason="找不到 next_trade_date：可能第3日真实数据尚未到来（正常 pending）",
            trade_date=trade_date,
            next_trade_date=None,
        )
        return pd.DataFrame(), meta

    # 过滤出第2日/第3日
    df2 = df[df[col_date] == trade_date][[col_code, col_close]].rename(
        columns={col_code: "ts_code", col_close: "close_2"}
    )
    df3 = df[df[col_date] == next_trade_date][[col_code, col_close]].rename(
        columns={col_code: "ts_code", col_close: "close_3"}
    )

    if df2.empty:
        meta = LabelBuildMeta(
            pending=True,
            reason="第2日 close 数据缺失：无法计算 2→3 溢价标签",
            trade_date=trade_date,
            next_trade_date=next_trade_date,
        )
        return pd.DataFrame(), meta

    if df3.empty:
        meta = LabelBuildMeta(
            pending=True,
            reason="第3日 close 数据缺失：对照日尚未产生（正常 pending）",
            trade_date=trade_date,
            next_trade_date=next_trade_date,
        )
        return pd.DataFrame(), meta

    # 合并并计算 real_premium_ret
    out = df2.merge(df3, on="ts_code", how="left")
    out.insert(0, "trade_date", trade_date)
    out.insert(1, "next_trade_date", next_trade_date)

    # real_premium_ret = close_3 / close_2 - 1
    out["real_premium_ret"] = out["close_3"] / out["close_2"] - 1

    meta = LabelBuildMeta(
        pending=False,
        reason="ok",
        trade_date=trade_date,
        next_trade_date=next_trade_date,
    )
    return out, meta


__all__ = ["LabelBuildMeta", "build_premium_labels"]
