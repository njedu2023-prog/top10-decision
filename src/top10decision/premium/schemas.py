#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — 字段契约（Schemas）

本文件的职责：
- 统一定义 Premium 模块的输入/输出/标签/评估表的字段规范
- 提供“列名别名映射（alias）”以适配上游 decision 表列名差异
- 提供最小校验函数：确保关键字段存在，避免 silent wrong

⚠️ 设计原则（工程约束）：
1) PremiumRet(2→3) 固定为：RealPremiumRet = Close[3] / Close[2] - 1
2) Premium 预测阶段只允许使用“<= 1日收盘后可得信息” + “1日生成的2日预测表”。
   任何 2日盘中/2日收盘后才知道的信息，不能进入特征（防未来函数泄漏）。
3) 字段分为：必需(required) 与可选(optional)。上游字段缺失时可降级，但必须显式告警。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# 1) 通用工具：列名映射与校验
# =========================

def normalize_columns(cols: Iterable[str]) -> List[str]:
    """把列名统一成小写，用于鲁棒匹配（不改变原表列名，只用于判断）。"""
    return [str(c).strip().lower() for c in cols]


def first_present(candidates: Sequence[str], cols_lower: Sequence[str]) -> Optional[str]:
    """在 cols_lower 中寻找第一个出现的 candidate（candidate 已经是 lower 形式）。"""
    for c in candidates:
        if c in cols_lower:
            return c
    return None


def resolve_required_columns(
    columns: Sequence[str],
    required_aliases: Dict[str, Sequence[str]],
) -> Dict[str, str]:
    """
    根据 required_aliases 在 columns 中解析出“规范字段名 -> 实际列名”的映射。

    required_aliases:
      - key: 规范字段名（canonical）
      - value: 可能出现的列名别名（alias list）

    返回：
      resolved: {canonical: actual_column_name_in_input_df}

    如果某个 canonical 找不到任何 alias，则抛 ValueError。
    """
    cols = list(columns)
    cols_lower = normalize_columns(cols)
    lower_to_actual = {c.lower(): c for c in cols}  # 输入表可能大小写不一致，这里保留原名

    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for canonical, aliases in required_aliases.items():
        aliases_lower = [a.strip().lower() for a in aliases]
        hit = first_present(aliases_lower, cols_lower)
        if hit is None:
            missing.append(canonical)
        else:
            resolved[canonical] = lower_to_actual[hit]

    if missing:
        raise ValueError(
            f"[premium.schemas] 输入表缺少必需列：{missing}。"
            f"当前列：{cols}"
        )
    return resolved


def resolve_optional_columns(
    columns: Sequence[str],
    optional_aliases: Dict[str, Sequence[str]],
) -> Dict[str, str]:
    """与 resolve_required_columns 类似，但缺失时不报错，仅跳过。"""
    cols = list(columns)
    cols_lower = normalize_columns(cols)
    lower_to_actual = {c.lower(): c for c in cols}

    resolved: Dict[str, str] = {}
    for canonical, aliases in optional_aliases.items():
        aliases_lower = [a.strip().lower() for a in aliases]
        hit = first_present(aliases_lower, cols_lower)
        if hit is not None:
            resolved[canonical] = lower_to_actual[hit]
    return resolved


# =========================
# 2) 输入契约：上游 decision 预测表（用于构建 Premium 特征）
# =========================

@dataclass(frozen=True)
class DecisionInputSchema:
    """
    Premium 模块读取的“第2日预测表”（由1日数据生成）字段契约。

    这是 Premium 的核心输入：
    - trade_date：该表对应的交易日（=第2日）
    - ts_code：股票代码（tushare 格式）
    - name：股票名称（可选，但强烈建议）
    - rank：该表内部排序（可选，但建议有，用于衍生特征）
    - 上游得分/概率/题材等列：可选（越多越好，但必须是<=1日可得信息派生）

    注意：
    - 上游列名可能不一致，因此用 aliases 做兼容。
    """

    # 必需（最低可跑）
    REQUIRED_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore

    # 可选（缺失可降级）
    OPTIONAL_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore

    @staticmethod
    def required_aliases() -> Dict[str, Sequence[str]]:
        return {
            # 规范字段名 -> 可能出现的列名
            "trade_date": ("trade_date", "date", "dt", "交易日期", "日期"),
            "ts_code": ("ts_code", "code", "symbol", "ticker", "股票代码", "代码"),
        }

    @staticmethod
    def optional_aliases() -> Dict[str, Sequence[str]]:
        return {
            "name": ("name", "stock_name", "股票名称", "名称"),
            "rank": ("rank", "rank_no", "排名", "order"),
            # 上游常见评分/因子（示例：你系统里可能叫 StrengthScore/ThemeBoost 等）
            "strength_score": ("strengthscore", "strength_score", "强度得分", "strength"),
            "theme_boost": ("themeboost", "theme_boost", "题材加权", "theme"),
            "probability": ("probability", "prob", "_prob", "p", "概率"),
            "final_score": ("finalscore", "final_score", "最终得分", "score"),
            "regime_weight": ("regime_weight", "regime", "情绪权重", "市场状态权重"),
            "industry": ("industry", "行业"),
            "theme": ("theme", "题材", "concept", "concept_name"),
            "turnover_rate": ("turnover_rate", "换手率"),
            "amount": ("amount", "成交额"),
            "vol": ("vol", "volume", "成交量"),
            # 预留：若上游带“可成交风险提示”也可作为特征/风险提示（但本模块不做 P_fill）
            "fill_risk_hint": ("fill_risk_hint", "fillrisk", "成交风险", "买不到风险"),
        }

    @staticmethod
    def resolve(columns: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        返回：
          required_map: canonical -> actual col name
          optional_map: canonical -> actual col name
        """
        required_map = resolve_required_columns(columns, DecisionInputSchema.required_aliases())
        optional_map = resolve_optional_columns(columns, DecisionInputSchema.optional_aliases())
        return required_map, optional_map


# =========================
# 3) 标签/对照契约：真实收盘价数据（用于计算 RealPremiumRet）
# =========================

@dataclass(frozen=True)
class CloseLabelSchema:
    """
    计算 RealPremiumRet(2→3) 的最小字段契约（真实对照）。

    必需：
    - trade_date：交易日（用于定位第2日/第3日）
    - ts_code：股票代码
    - close：收盘价

    说明：
    - 你可以从任何可靠来源拿到 (trade_date, ts_code, close)
      例如：本仓库已有落盘、或从 data 仓库同步等。
    """

    REQUIRED_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore
    OPTIONAL_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore

    @staticmethod
    def required_aliases() -> Dict[str, Sequence[str]]:
        return {
            "trade_date": ("trade_date", "date", "dt", "交易日期", "日期"),
            "ts_code": ("ts_code", "code", "symbol", "ticker", "股票代码", "代码"),
            "close": ("close", "close_price", "收盘价"),
        }

    @staticmethod
    def optional_aliases() -> Dict[str, Sequence[str]]:
        return {
            "open": ("open", "开盘价"),
            "high": ("high", "最高价"),
            "low": ("low", "最低价"),
            "pre_close": ("pre_close", "昨收", "前收盘"),
            "pct_chg": ("pct_chg", "pct_change", "涨跌幅"),
            "amount": ("amount", "成交额"),
            "vol": ("vol", "volume", "成交量"),
        }

    @staticmethod
    def resolve(columns: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        required_map = resolve_required_columns(columns, CloseLabelSchema.required_aliases())
        optional_map = resolve_optional_columns(columns, CloseLabelSchema.optional_aliases())
        return required_map, optional_map


# =========================
# 4) Premium 输出契约：给人类看的“溢价预测排序表”（csv/md）
# =========================

@dataclass(frozen=True)
class PremiumRankOutputSchema:
    """
    Premium 最终输出表字段（写入 outputs/premium/rank/*.csv 以及用于 md 报告）。

    核心字段：
    - trade_date：第2日（这张表的日期语义统一按 2日）
    - next_trade_date：第3日（用于对照 RealPremiumRet）
    - ts_code, name
    - pred_up_prob：预测 2→3 为正的概率（0~1）
    - pred_ret_mean：预测 2→3 的平均收益（回归输出，单位：比例，如 0.03=3%）
    - pred_ev：预测期望收益（建议 = pred_up_prob * pred_ret_mean）
    - rank_pred_ev：按 pred_ev 排序后的名次

    风险提示字段（先预留，后续在 features/labels 里逐步填充）：
    - risk_liquidity / risk_volatility / risk_crowding / risk_event
    - confidence：模型置信度（高/中/低 或 0~1）

    可追溯字段：
    - run_id / commit_sha / model_version / data_snapshot
    """

    # 建议输出列顺序（便于阅读与前端展示）
    COLUMNS: Tuple[str, ...] = (
        "trade_date",
        "next_trade_date",
        "ts_code",
        "name",
        # 上游信息（可选）
        "decision_rank",
        "strength_score",
        "theme_boost",
        "probability",
        "final_score",
        # Premium 预测
        "pred_up_prob",
        "pred_ret_mean",
        "pred_ev",
        "rank_pred_ev",
        # 风险提示（结构化）
        "risk_liquidity",
        "risk_volatility",
        "risk_crowding",
        "risk_event",
        "fill_risk_hint",
        "confidence",
        # 真实对照（训练/回测时可填；纯预测日可为空）
        "real_premium_ret",
        # 追溯
        "run_id",
        "commit_sha",
        "model_version",
        "data_snapshot",
        "created_at_utc",
    )


# =========================
# 5) 学习/评估落库契约：用于 learning/ 目录
# =========================

@dataclass(frozen=True)
class PremiumEvalHistorySchema:
    """
    learning/premium_eval_history.csv 的字段建议（滚动评估）。

    评估口径（按 2日为主）：
    - trade_date：第2日
    - next_trade_date：第3日（对照日）
    - n：样本数（当日参与排序的股票数量）
    - topk：评估用的 K（例如 10/20）
    - hit_rate_at_k：TopK 中 real_premium_ret>0 的比例
    - mean_ret_at_k：TopK 真实平均收益
    - rank_ic：pred_ev 与 real_premium_ret 的秩相关（Spearman）
    - model_version / run_id / commit_sha：追溯
    """

    COLUMNS: Tuple[str, ...] = (
        "trade_date",
        "next_trade_date",
        "n",
        "topk",
        "hit_rate_at_k",
        "mean_ret_at_k",
        "rank_ic",
        "model_version",
        "run_id",
        "commit_sha",
        "created_at_utc",
    )


# =========================
# 6) 统一导出：便于其它模块引用
# =========================

__all__ = [
    "DecisionInputSchema",
    "CloseLabelSchema",
    "PremiumRankOutputSchema",
    "PremiumEvalHistorySchema",
    "resolve_required_columns",
    "resolve_optional_columns",
]
