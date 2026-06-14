#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — 字段契约（Schemas｜V3.1：E_ret_plus / EHX 字段契约增强版）

本文件的职责：
- 统一定义 Premium 模块的输入/输出/标签/评估表的字段规范。
- 提供“列名别名映射（alias）”以适配上游 decision / pred_source 表列名差异。
- 提供最小校验函数：确保关键字段存在，避免 silent wrong。
- 将 E_ret_plus / EHX 新增字段纳入正式契约，避免后续写入、校验、展示或历史评估时被旧 schema 截断。

⚠️ 设计原则（工程约束）：
1) PremiumRet(2→3) 固定为：RealPremiumRet = Close[3] / Close[2] - 1。
2) Premium 预测阶段只允许使用“<= 1日收盘后可得信息” + “1日生成的2日预测表”。
   任何 2日盘中/2日收盘后才知道的信息，不能进入特征（防未来函数泄漏）。
3) 字段分为：必需(required) 与可选(optional)。上游字段缺失时可降级，但必须显式告警。
4) V3 主线新增 EHX：
   eret_plus_value = eret_pred_raw + eret_plus_delta。
   raw / plus 后验误差必须可落库、可报告、可回灌。
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
# 2) 输入契约：上游 decision / pred_source 预测表（用于构建 Premium 特征）
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
    - 上游得分/概率/题材/成本/风险/E_ret 等列：可选。

    注意：
    - 上游列名可能不一致，因此用 aliases 做兼容。
    - 这里仅声明字段契约，不代表所有字段都必须存在。
    """

    # 必需（最低可跑）
    REQUIRED_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore

    # 可选（缺失可降级）
    OPTIONAL_ALIASES: Dict[str, Sequence[str]] = None  # type: ignore

    @staticmethod
    def required_aliases() -> Dict[str, Sequence[str]]:
        return {
            "trade_date": ("trade_date", "date", "dt", "交易日期", "日期"),
            "ts_code": ("ts_code", "code", "symbol", "ticker", "股票代码", "代码"),
        }

    @staticmethod
    def optional_aliases() -> Dict[str, Sequence[str]]:
        return {
            "name": ("name", "stock_name", "股票名称", "名称"),
            "rank": ("rank", "rank_no", "排名", "order"),
            "decision_rank": ("decision_rank", "dec_rank", "rank_pred_ev", "rank_eret_plus", "决策排名"),

            # 上游常见评分/因子
            "strength_score": ("strengthscore", "strength_score", "强度得分", "strength", "f_strength"),
            "theme_boost": ("themeboost", "theme_boost", "题材加权", "themeboost_score", "f_theme"),
            "probability": ("probability", "prob", "_prob", "p", "p_premium", "up_prob", "f_prob", "概率"),
            "final_score": ("finalscore", "final_score", "最终得分", "score", "score_ev", "ev", "pred_ev"),
            "regime_weight": ("regime_weight", "regime", "情绪权重", "市场状态权重"),
            "industry": ("industry", "board", "sector", "sw_industry", "申万行业", "行业", "板块", "所属行业", "所属板块"),
            "theme": ("theme", "题材", "concept", "concept_name"),
            "turnover_rate": ("turnover_rate", "换手率"),
            "amount": ("amount", "成交额"),
            "vol": ("vol", "volume", "成交量"),
            "close": ("close", "close_t", "close_T", "收盘价"),
            "pct_chg": ("pct_chg", "pct_change", "涨跌幅"),
            "amplitude": ("amplitude", "range_1d", "振幅"),

            # P_fill / 成本 / 风险：EHX 与排序解释会用到
            "p_fill_pred": ("p_fill_pred", "p_fill_pred_final", "p_fill", "P_fill", "dec_p_fill"),
            "cost_total": ("cost_total", "cost", "cost_value", "cost_all", "trade_cost"),
            "risk_penalty_total": (
                "risk_penalty_total",
                "risk_penalty",
                "riskpenalty",
                "risk_penalty_score",
                "risk_score",
            ),

            # 原始 E_ret / E_ret_plus 字段别名，需与 train.py / predict.py 对齐
            "eret_pred_raw": (
                "eret_pred_raw",
                "e_ret_pred_raw",
                "raw_eret_pred",
                "raw_e_ret_pred",
                "eret_pred",
                "e_ret_pred",
                "E_ret",
                "e_ret",
                "eret_pred_final",
                "e_premium",
                "pred_ret",
                "pred_return",
                "ret",
                "premium_ret",
                "pred_premium_ret",
                "pred_ret_mean",
            ),
            "eret_plus_value": (
                "eret_plus_value",
                "eret_plus",
                "e_ret_plus",
                "E_ret_plus",
                "eret_plus_pred",
                "e_ret_plus_pred",
            ),
            "eret_plus_delta": ("eret_plus_delta", "ehx_delta", "delta_hat", "delta_ret_hat"),
            "eret_plus_direction": ("eret_plus_direction", "ehx_direction"),
            "eret_plus_conf": ("eret_plus_conf", "ehx_conf", "ehx_confidence"),
            "eret_plus_conf_score": ("eret_plus_conf_score", "ehx_conf_score"),
            "eret_plus_src": ("eret_plus_src", "ehx_src", "ehx_source"),

            # 预留：若上游带“可成交风险提示”也可作为特征/风险提示
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
# 4) Premium 输出契约：预测排序表 / full 表 / report 表字段
# =========================

@dataclass(frozen=True)
class PremiumRankOutputSchema:
    """
    Premium 最终输出表字段建议。

    兼容两条线：
    - 旧线：pred_up_prob / pred_ret_mean / pred_ev / rank_pred_ev。
    - V3 线：eret_pred_raw / eret_plus_value / eret_plus_delta / rank_eret_plus。

    注意：
    - 这是字段契约和建议顺序，不强制所有文件都必须完整输出全部列。
    - predict.py 当前会按自身 out_cols 输出；本 schema 用于校验、排序、后续接线和历史兼容。
    """

    COLUMNS: Tuple[str, ...] = (
        "rank",
        "trade_date",
        "next_trade_date",
        "target_date",
        "ts_code",
        "name",
        "sector",
        "close_T",

        # 上游信息（可选）
        "decision_rank",
        "dec_rank",
        "dec_weight",
        "dec_can_buy",
        "dec_p_fill",
        "dec_reason",
        "strength_score",
        "theme_boost",
        "probability",
        "final_score",

        # 旧 Premium 预测
        "pred_up_prob",
        "pred_ret_mean",
        "pred_ev",
        "rank_pred_ev",

        # V3 / EHX 主线预测
        "eret_pred_raw",
        "eret_plus_value",
        "eret_plus_delta",
        "eret_plus_direction",
        "eret_plus_conf",
        "eret_plus_conf_score",
        "eret_plus_src",
        "rank_eret_plus",

        # 分布预测
        "rank_r_p50",
        "r_p05",
        "r_p25",
        "r_p50",
        "r_p75",
        "r_p95",
        "close_T2_p05",
        "close_T2_p25",
        "close_T2_p50",
        "close_T2_p75",
        "close_T2_p95",

        # 风险提示（结构化）
        "risk_liquidity",
        "risk_volatility",
        "risk_crowding",
        "risk_event",
        "fill_risk_hint",
        "risk_flags",
        "confidence",
        "data_quality",

        # 真实对照（训练/回测时可填；纯预测日可为空）
        "real_premium_ret",
        "actual_ret",
        "close_T2_actual",
        "raw_abs_err",
        "plus_abs_err",
        "improve_flag",
        "hit_up",

        # 追溯
        "run_id",
        "commit_sha",
        "model_version",
        "data_snapshot",
        "created_at_utc",
    )


# =========================
# 5) Premium 验证输出契约：premium_verify_{T}.csv
# =========================

@dataclass(frozen=True)
class PremiumVerifyOutputSchema:
    """Premium 验证表字段建议，重点服务 EHX raw-vs-plus 后验验收。"""

    COLUMNS: Tuple[str, ...] = (
        "rank",
        "trade_date",
        "target_date",
        "ts_code",
        "name",
        "close_T",
        "close_T2_actual",
        "r_actual",
        "r_p50",
        "eret_pred_raw",
        "eret_plus_value",
        "eret_plus_delta",
        "eret_plus_direction",
        "eret_plus_conf",
        "in_p10",
        "in_p50",
        "err_r_p50",
        "err_close_p50",
        "actual_ret",
        "raw_abs_err",
        "plus_abs_err",
        "improve_flag",
        "hit_up",
    )


# =========================
# 6) 学习/评估落库契约：用于 learning/ 目录
# =========================

@dataclass(frozen=True)
class PremiumEvalHistorySchema:
    """
    learning/premium_eval_history.csv 的字段建议（滚动评估）。

    旧评估口径：
    - hit_rate_at_k / mean_ret_at_k / rank_ic。

    V3 / EHX 评估口径：
    - ehx_trained / ehx_reason / delta_mae / delta_rmse。
    - plus_improve_rate：最后一个可验证交易日中，Plus 绝对误差小于 Raw 绝对误差的比例。
    """

    COLUMNS: Tuple[str, ...] = (
        "trade_date",
        "next_trade_date",
        "n",
        "topk",
        "hit_rate_at_k",
        "mean_ret_at_k",
        "rank_ic",

        # EHX 训练与评估追溯
        "ehx_trained",
        "ehx_reason",
        "ehx_n_samples",
        "ehx_min_samples",
        "delta_mae",
        "delta_rmse",
        "plus_improve_rate",

        # 追溯
        "model_version",
        "run_id",
        "commit_sha",
        "created_at_utc",
    )


# =========================
# 7) 统一导出：便于其它模块引用
# =========================

__all__ = [
    "DecisionInputSchema",
    "CloseLabelSchema",
    "PremiumRankOutputSchema",
    "PremiumVerifyOutputSchema",
    "PremiumEvalHistorySchema",
    "resolve_required_columns",
    "resolve_optional_columns",
]
