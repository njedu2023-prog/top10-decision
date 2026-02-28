#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Features（特征工程）

P0 目标：最小可跑、严格不泄漏未来信息。
- 输入：第2日预测表（由第1日数据生成），属于“<=1日收盘后可得信息”的派生结果
- 输出：
  1) X 特征矩阵（DataFrame，数值列）
  2) meta_df（保留 ts_code/name/上游分数等，用于最终输出与解释）
  3) risk_df（结构化风险提示字段，P0 版本先做流动性风险）

注意：
- 本模块不引入任何 2日盘中/2日收盘后信息（比如 2日涨跌幅、开盘成交等）。
- 仅使用 decision 表内的列。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schemas import DecisionInputSchema


@dataclass(frozen=True)
class FeatureBuildResult:
    trade_date: str
    X: pd.DataFrame          # 纯数值特征
    meta: pd.DataFrame       # 输出所需信息（非训练特征也可放这里）
    risk: pd.DataFrame       # 风险提示（结构化）
    feature_cols: List[str]  # X 的列名列表


def _to_yyyymmdd(s: str) -> str:
    s = str(s).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _rank_to_score(rank: pd.Series) -> pd.Series:
    """
    把 rank（越小越好）转成一个 0~1 的分数，便于模型使用。
    """
    r = _safe_numeric(rank)
    if r.notna().sum() == 0:
        return pd.Series([np.nan] * len(rank), index=rank.index)
    # 1 / rank 归一化
    inv = 1.0 / r.replace(0, np.nan)
    inv = (inv - np.nanmin(inv)) / (np.nanmax(inv) - np.nanmin(inv) + 1e-12)
    return inv


def _quantile_bucket(x: pd.Series, q: float) -> pd.Series:
    """
    简单分位数：返回 x 是否低于 q 分位（True=风险高），用于 P0 风险提示。
    """
    v = _safe_numeric(x)
    if v.notna().sum() < 5:
        return pd.Series([pd.NA] * len(x), index=x.index)
    thr = float(v.quantile(q))
    return v < thr


def build_features_from_decision_df(decision_df: pd.DataFrame) -> FeatureBuildResult:
    """
    从上游 decision 预测表构建特征。

    约束：
    - decision_df 必须包含 trade_date + ts_code（别名允许）
    - trade_date 如非唯一，将取最大值作为兜底（但会在 meta 里保留原始情况）
    """
    if decision_df is None or decision_df.empty:
        raise ValueError("[premium.features] decision_df 为空，无法构建特征")

    # 解析列别名
    req_map, opt_map = DecisionInputSchema.resolve(decision_df.columns)
    col_date = req_map["trade_date"]
    col_code = req_map["ts_code"]

    df = decision_df.copy()
    df[col_date] = df[col_date].astype(str).map(_to_yyyymmdd)
    df[col_code] = df[col_code].astype(str).str.strip()

    # trade_date 取唯一值（兜底：最大值）
    unique_dates = sorted([d for d in df[col_date].dropna().unique() if str(d).isdigit() and len(str(d)) == 8])
    if not unique_dates:
        trade_date = str(df[col_date].dropna().iloc[0]) if df[col_date].notna().any() else "unknown"
    else:
        trade_date = unique_dates[-1]

    # ---- meta：用于最终输出与解释 ----
    meta_cols: Dict[str, Optional[str]] = {
        "trade_date": col_date,
        "ts_code": col_code,
        "name": opt_map.get("name"),
        "decision_rank": opt_map.get("rank"),
        "strength_score": opt_map.get("strength_score"),
        "theme_boost": opt_map.get("theme_boost"),
        "probability": opt_map.get("probability"),
        "final_score": opt_map.get("final_score"),
        "regime_weight": opt_map.get("regime_weight"),
        "industry": opt_map.get("industry"),
        "theme": opt_map.get("theme"),
        "turnover_rate": opt_map.get("turnover_rate"),
        "amount": opt_map.get("amount"),
        "vol": opt_map.get("vol"),
        "fill_risk_hint": opt_map.get("fill_risk_hint"),
    }

    meta = pd.DataFrame({
        k: (df[v] if v and v in df.columns else pd.Series([pd.NA] * len(df)))
        for k, v in meta_cols.items()
    })
    # 规范列名
    meta["trade_date"] = meta["trade_date"].astype(str).map(_to_yyyymmdd)
    meta["ts_code"] = meta["ts_code"].astype(str).str.strip()

    # ---- X：数值特征（P0：自动挑选）----
    # 先手工加入“rank_score”这种派生特征
    x_parts = {}

    if meta["decision_rank"].notna().any():
        x_parts["rank_score"] = _rank_to_score(meta["decision_rank"])
    else:
        x_parts["rank_score"] = pd.Series([np.nan] * len(meta), index=meta.index)

    # 把 meta 中可能的数值列加入候选
    numeric_candidates = [
        "strength_score", "theme_boost", "probability", "final_score",
        "regime_weight", "turnover_rate", "amount", "vol",
    ]
    for c in numeric_candidates:
        x_parts[c] = _safe_numeric(meta[c])

    # 额外：从原 df 中自动扫描“可转数值”的列作为补充特征（但排除明显非特征）
    blacklist = set([col_date, col_code])
    blacklist |= set([v for v in meta_cols.values() if v])  # 已经纳入 meta 的列先不重复处理

    auto_numeric = {}
    for col in df.columns:
        if col in blacklist:
            continue
        # 排除明显的文本列
        if any(key in str(col).lower() for key in ("name", "industry", "theme", "concept", "题材", "行业")):
            continue
        s_num = _safe_numeric(df[col])
        # 至少有一定比例可转数值才纳入
        if s_num.notna().mean() >= 0.6:
            auto_numeric[f"auto__{str(col)}"] = s_num

    x_parts.update(auto_numeric)

    X = pd.DataFrame(x_parts)

    # 缺失填充（P0：先用中位数，避免模型报错）
    for c in X.columns:
        if X[c].notna().any():
            med = float(X[c].median())
            X[c] = X[c].fillna(med)
        else:
            X[c] = X[c].fillna(0.0)

    # 简单标准化（P0：均值0方差1，避免特征尺度差太大；后续可移到训练 pipeline）
    for c in X.columns:
        v = X[c].astype(float)
        std = float(v.std(ddof=0))
        if std < 1e-12:
            X[c] = 0.0
        else:
            X[c] = (v - float(v.mean())) / std

    feature_cols = list(X.columns)

    # ---- risk：结构化风险提示（P0）----
    # P0 先做流动性风险：成交额/换手率低分位 -> 风险高
    risk = pd.DataFrame({
        "risk_liquidity": pd.NA,
        "risk_volatility": pd.NA,
        "risk_crowding": pd.NA,
        "risk_event": pd.NA,
        "confidence": pd.NA,
    })

    # 量化一个粗风险：成交额低于 20% 分位 -> HIGH
    liq_flag = None
    if meta["amount"].notna().any():
        liq_flag = _quantile_bucket(meta["amount"], 0.2)
    elif meta["turnover_rate"].notna().any():
        liq_flag = _quantile_bucket(meta["turnover_rate"], 0.2)

    if liq_flag is not None:
        risk["risk_liquidity"] = liq_flag.map(lambda x: "HIGH" if x is True else ("LOW" if x is False else pd.NA))

    return FeatureBuildResult(
        trade_date=trade_date,
        X=X,
        meta=meta,
        risk=risk,
        feature_cols=feature_cols,
    )


__all__ = ["FeatureBuildResult", "build_features_from_decision_df"]
