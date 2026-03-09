#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ============================================================
# Cost / RiskPenalty 模块（Decision 主线）
# ------------------------------------------------------------
# 目标：
# 1) 不再停留在“固定 8bp + regime 二值开关”的占位状态
# 2) 升级为可交易、可解释、可逐股分化的轻量实用版本
# 3) 保持向后兼容：老调用方式不报错
#
# 说明：
# - 当前阶段仍属于“规则增强版”，不是学习模型版
# - 下一步需要 run_v2.py 真正逐股接线，EV 才会做实
# ============================================================

# -----------------------------
# 默认参数（可后续外置配置）
# -----------------------------
COST_BP_DEFAULT = 8.0  # 兼容旧逻辑：默认总成本 8bp

# 基础手续费 / 过户 / 交易摩擦的保守近似（单边）
BASE_FEE_BP = 2.0

# 默认滑点 bp（按板块）
SLIPPAGE_MAIN_BP = 5.0
SLIPPAGE_GROWTH_BP = 7.0   # 创业板
SLIPPAGE_STAR_BP = 9.0     # 科创板

# 冲击成本参数
IMPACT_TURNOVER_REF = 8.0
IMPACT_MIN_BP = 0.0
IMPACT_MAX_BP = 18.0

# 风险惩罚默认值（兼容旧逻辑）
RISK_PENALTY_OFF = 0.00
RISK_PENALTY_ON = 0.02

# 风险惩罚上限，避免把 EV 一次性扣穿得太离谱
RISK_PENALTY_CAP = 0.06

# 各风险子项权重（百分比收益口径，不是 bp）
W_REGIME_DEFENSE = 0.020
W_LIMIT_OPEN = 0.012
W_HIGH_OPEN_TIMES = 0.010
W_VOLATILITY = 0.008
W_LOW_LIQUIDITY = 0.010
W_HOT_THEME = 0.006
W_GEM_STAR = 0.004
W_EXTREME_TURNOVER = 0.005


# ============================================================
# 基础工具
# ============================================================

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_str(v: Any, default: str = "") -> str:
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return str(v).strip()
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _normalize_bp(bp: float) -> float:
    return float(bp) / 10000.0


def _pick_first(row: pd.Series, *candidates: str, default: Any = None) -> Any:
    for c in candidates:
        if c in row.index:
            v = row.get(c)
            if v is not None and not pd.isna(v):
                return v
    return default


def _board_bucket(row: pd.Series) -> str:
    """
    粗分板块：
    - STAR: 科创板 688*
    - GROWTH: 创业板 300*
    - MAIN: 其余主板/中小板等
    """
    ts_code = _safe_str(_pick_first(row, "ts_code", default=""))
    board = _safe_str(_pick_first(row, "board", "market_board", default="")).upper()

    if ts_code.startswith("688") or "STAR" in board or "科创" in board:
        return "STAR"
    if ts_code.startswith("300") or "GEM" in board or "创业" in board:
        return "GROWTH"
    return "MAIN"


def _turnover_rate(row: pd.Series) -> float:
    return _safe_float(
        _pick_first(
            row,
            "turnover_rate",
            "turnover_rate_f",
            "换手率",
            default=np.nan,
        ),
        default=np.nan,
    )


def _amp_pct(row: pd.Series) -> float:
    """
    日内振幅/波动代理，口径按百分数值处理，例如 8.3 表示 8.3%
    """
    return _safe_float(
        _pick_first(
            row,
            "amp",
            "amplitude",
            "pct_amplitude",
            "振幅",
            default=np.nan,
        ),
        default=np.nan,
    )


def _open_times(row: pd.Series) -> float:
    return _safe_float(
        _pick_first(row, "open_times", "炸板次数", default=np.nan),
        default=np.nan,
    )


def _is_limit_open_risk(row: pd.Series) -> bool:
    """
    开板/炸板风险代理：
    - open_times > 0
    - 或 seal 弱、limit 状态不强
    """
    ot = _open_times(row)
    if ot > 0:
        return True

    status = _safe_str(
        _pick_first(row, "limit_status", "涨停状态", "封板状态", default="")
    ).upper()
    if status in {"OPEN", "BROKEN", "WEAK"}:
        return True
    return False


def _theme_heat_score(row: pd.Series) -> float:
    return _safe_float(
        _pick_first(
            row,
            "ThemeBoost",
            "theme_boost",
            "theme_heat",
            default=np.nan,
        ),
        default=np.nan,
    )


def _seal_strength(row: pd.Series) -> float:
    """
    封单强度代理，值越大越稳。
    """
    seal_amount = _safe_float(
        _pick_first(row, "seal_amount", "封单额", default=np.nan),
        default=np.nan,
    )
    if np.isnan(seal_amount):
        return np.nan
    return seal_amount


def _slippage_bp_by_board(row: pd.Series) -> float:
    bucket = _board_bucket(row)
    if bucket == "STAR":
        return SLIPPAGE_STAR_BP
    if bucket == "GROWTH":
        return SLIPPAGE_GROWTH_BP
    return SLIPPAGE_MAIN_BP


# ============================================================
# Cost 模块
# ============================================================

def _impact_cost_bp(row: pd.Series) -> float:
    """
    冲击成本（bp）：
    用换手率做一个非常轻量但有分化能力的近似。
    逻辑：
    - 换手率过低：流动性差，成交冲击更高
    - 换手率极高：博弈拥挤，也会有冲击/抢筹风险
    """
    turnover = _turnover_rate(row)

    if np.isnan(turnover):
        return 1.5

    # 中性区：around 8%
    # 偏离越大，冲击越高
    gap = abs(turnover - IMPACT_TURNOVER_REF)

    # 低换手更危险一点
    if turnover < 4:
        bp = 6.0 + (4.0 - turnover) * 1.5
    # 超高换手也加冲击
    elif turnover > 18:
        bp = 4.0 + (turnover - 18.0) * 0.5
    else:
        bp = gap * 0.35

    return _clip(bp, IMPACT_MIN_BP, IMPACT_MAX_BP)


def _cost_from_row(row: pd.Series) -> float:
    """
    返回单只股票的总成本（小数收益口径）
    EV 中按：
        EV = P_fill * E_ret - Cost - RiskPenalty
    所以这里直接返回小数，例如 0.0008 = 8bp
    """
    base_fee_bp = BASE_FEE_BP
    slippage_bp = _slippage_bp_by_board(row)
    impact_bp = _impact_cost_bp(row)

    total_bp = base_fee_bp + slippage_bp + impact_bp
    return _normalize_bp(total_bp)


def cost_estimate_rule(
    data: pd.DataFrame | pd.Series | None = None,
) -> pd.Series | float:
    """
    向后兼容：
    1) 老调用：cost_estimate_rule() -> float
    2) 新调用：cost_estimate_rule(row: Series) -> float
    3) 新调用：cost_estimate_rule(df: DataFrame) -> Series
    """
    if data is None:
        return _normalize_bp(COST_BP_DEFAULT)

    if isinstance(data, pd.Series):
        return float(_cost_from_row(data))

    if isinstance(data, pd.DataFrame):
        if data.empty:
            return pd.Series(dtype=float)
        return data.apply(_cost_from_row, axis=1).astype(float)

    return _normalize_bp(COST_BP_DEFAULT)


# ============================================================
# RiskPenalty 模块
# ============================================================

def _risk_penalty_from_row(row: pd.Series, regime: str = "RISK_ON") -> float:
    penalty = 0.0

    # 1) 大盘/制度层风险
    regime_u = _safe_str(regime, default="RISK_ON").upper()
    if regime_u in ("RISK_OFF", "OFF", "DEFENSE"):
        penalty += W_REGIME_DEFENSE

    # 2) 开板/炸板风险
    if _is_limit_open_risk(row):
        penalty += W_LIMIT_OPEN

    # 3) 炸板次数高
    ot = _open_times(row)
    if not np.isnan(ot) and ot >= 2:
        penalty += min(W_HIGH_OPEN_TIMES, 0.003 * ot)

    # 4) 波动过大
    amp = _amp_pct(row)
    if not np.isnan(amp) and amp >= 8:
        penalty += min(W_VOLATILITY, 0.001 * (amp - 8) + 0.003)

    # 5) 流动性差 / 低换手风险
    turnover = _turnover_rate(row)
    if not np.isnan(turnover):
        if turnover < 3:
            penalty += W_LOW_LIQUIDITY
        elif turnover < 6:
            penalty += 0.005

        # 极高换手也加一点拥挤惩罚
        if turnover > 25:
            penalty += W_EXTREME_TURNOVER

    # 6) 题材过热 / 拥挤风险
    theme_heat = _theme_heat_score(row)
    if not np.isnan(theme_heat):
        if theme_heat >= 8:
            penalty += W_HOT_THEME
        elif theme_heat >= 6:
            penalty += 0.003

    # 7) 科创/创业板波动额外惩罚
    bucket = _board_bucket(row)
    if bucket in ("STAR", "GROWTH"):
        penalty += W_GEM_STAR

    # 8) 封单弱 -> 额外惩罚
    seal_strength = _seal_strength(row)
    if not np.isnan(seal_strength):
        if seal_strength <= 2_000_000:
            penalty += 0.004
        elif seal_strength <= 5_000_000:
            penalty += 0.002

    return _clip(penalty, 0.0, RISK_PENALTY_CAP)


def risk_penalty_rule(
    regime: str,
    data: pd.DataFrame | pd.Series | None = None,
) -> pd.Series | float:
    """
    向后兼容：
    1) 老调用：risk_penalty_rule(regime) -> float
    2) 新调用：risk_penalty_rule(regime, row) -> float
    3) 新调用：risk_penalty_rule(regime, df) -> Series
    """
    regime_u = _safe_str(regime, default="RISK_ON").upper()

    if data is None:
        if regime_u in ("RISK_OFF", "OFF", "DEFENSE"):
            return float(RISK_PENALTY_ON)
        return float(RISK_PENALTY_OFF)

    if isinstance(data, pd.Series):
        return float(_risk_penalty_from_row(data, regime=regime_u))

    if isinstance(data, pd.DataFrame):
        if data.empty:
            return pd.Series(dtype=float)
        return data.apply(
            lambda r: _risk_penalty_from_row(r, regime=regime_u), axis=1
        ).astype(float)

    if regime_u in ("RISK_OFF", "OFF", "DEFENSE"):
        return float(RISK_PENALTY_ON)
    return float(RISK_PENALTY_OFF)


# ============================================================
# EV 辅助函数（为下一步 run_v2 接线预留）
# ============================================================

def ev_rule(
    p_fill: pd.Series | float,
    e_ret: pd.Series | float,
    cost: pd.Series | float,
    risk_penalty: pd.Series | float,
) -> pd.Series | float:
    """
    统一 EV 计算口径：
        EV = P_fill * E_ret - Cost - RiskPenalty
    """
    return (p_fill * e_ret) - cost - risk_penalty
