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
# - 本文件已支持“分项归因”，下一步只需 run_v2.py 接线即可落到候选表
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

# 额外成本参数（bp）
OPEN_COST_WEAK_BP = 2.0
OPEN_COST_BROKEN_BP = 4.0
OPEN_COST_MULTI_OPEN_MAX_BP = 6.0

SEAL_WEAK_BP_LOW = 1.5
SEAL_WEAK_BP_VERY_LOW = 3.0

AMOUNT_SMALL_BP = 2.0
AMOUNT_VERY_SMALL_BP = 4.0

LOW_PRICE_BP = 1.0
PENNY_PRICE_BP = 2.0

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
W_WEAK_SEAL_LOW = 0.002
W_WEAK_SEAL_VERY_LOW = 0.004

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
    return _safe_float(
        _pick_first(row, "seal_amount", "封单额", default=np.nan),
        default=np.nan,
    )


def _trade_amount(row: pd.Series) -> float:
    """
    成交额代理，尽量兼容不同字段名。
    常见单位可能是元；如果未来源表单位变化，再统一修。
    """
    return _safe_float(
        _pick_first(
            row,
            "amount",
            "成交额",
            "turnover_amount",
            "deal_amount",
            default=np.nan,
        ),
        default=np.nan,
    )


def _close_price(row: pd.Series) -> float:
    return _safe_float(
        _pick_first(
            row,
            "close",
            "收盘价",
            "last_price",
            default=np.nan,
        ),
        default=np.nan,
    )


def _limit_status(row: pd.Series) -> str:
    return _safe_str(
        _pick_first(row, "limit_status", "涨停状态", "封板状态", default="")
    ).upper()


def _is_limit_open_risk(row: pd.Series) -> bool:
    """
    开板/炸板风险代理：
    - open_times > 0
    - 或 limit_status 表示 OPEN / BROKEN / WEAK
    """
    ot = _open_times(row)
    if ot > 0:
        return True

    status = _limit_status(row)
    if status in {"OPEN", "BROKEN", "WEAK"}:
        return True
    return False


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
    用换手率做一个轻量但有分化能力的近似。
    """
    turnover = _turnover_rate(row)

    if np.isnan(turnover):
        return 1.5

    gap = abs(turnover - IMPACT_TURNOVER_REF)

    if turnover < 4:
        bp = 6.0 + (4.0 - turnover) * 1.5
    elif turnover > 18:
        bp = 4.0 + (turnover - 18.0) * 0.5
    else:
        bp = gap * 0.35

    return _clip(bp, IMPACT_MIN_BP, IMPACT_MAX_BP)


def _open_cost_bp(row: pd.Series) -> float:
    """
    开板/炸板导致的额外交易摩擦成本。
    """
    status = _limit_status(row)
    ot = _open_times(row)

    bp = 0.0
    if status == "WEAK":
        bp += OPEN_COST_WEAK_BP
    elif status in {"OPEN", "BROKEN"}:
        bp += OPEN_COST_BROKEN_BP

    if not np.isnan(ot) and ot > 0:
        bp += min(OPEN_COST_MULTI_OPEN_MAX_BP, ot * 1.2)

    return float(bp)


def _seal_cost_bp(row: pd.Series) -> float:
    """
    封单弱，意味着买入实现成本更高。
    """
    seal = _seal_strength(row)
    if np.isnan(seal):
        return 0.0
    if seal <= 2_000_000:
        return SEAL_WEAK_BP_VERY_LOW
    if seal <= 5_000_000:
        return SEAL_WEAK_BP_LOW
    return 0.0


def _amount_cost_bp(row: pd.Series) -> float:
    """
    小成交额流动性惩罚。
    """
    amount = _trade_amount(row)
    if np.isnan(amount):
        return 0.0
    if amount <= 30_000_000:
        return AMOUNT_VERY_SMALL_BP
    if amount <= 80_000_000:
        return AMOUNT_SMALL_BP
    return 0.0


def _price_cost_bp(row: pd.Series) -> float:
    """
    低价股微观结构摩擦成本。
    """
    px = _close_price(row)
    if np.isnan(px):
        return 0.0
    if px < 3:
        return PENNY_PRICE_BP
    if px < 5:
        return LOW_PRICE_BP
    return 0.0


def _cost_components_bp(row: pd.Series) -> dict[str, float]:
    base_fee_bp = BASE_FEE_BP
    slippage_bp = _slippage_bp_by_board(row)
    impact_bp = _impact_cost_bp(row)
    open_bp = _open_cost_bp(row)
    seal_bp = _seal_cost_bp(row)
    amount_bp = _amount_cost_bp(row)
    price_bp = _price_cost_bp(row)

    total_bp = (
        base_fee_bp
        + slippage_bp
        + impact_bp
        + open_bp
        + seal_bp
        + amount_bp
        + price_bp
    )

    return {
        "cost_base_bp": float(base_fee_bp),
        "cost_slippage_bp": float(slippage_bp),
        "cost_impact_bp": float(impact_bp),
        "cost_open_bp": float(open_bp),
        "cost_seal_bp": float(seal_bp),
        "cost_amount_bp": float(amount_bp),
        "cost_price_bp": float(price_bp),
        "cost_total_bp": float(total_bp),
    }


def _cost_from_row(row: pd.Series) -> float:
    """
    返回单只股票的总成本（小数收益口径）
    EV 中按：
        EV = P_fill * E_ret - Cost - RiskPenalty
    所以这里直接返回小数，例如 0.0008 = 8bp
    """
    parts = _cost_components_bp(row)
    return _normalize_bp(parts["cost_total_bp"])


def cost_breakdown_row(row: pd.Series) -> dict[str, float]:
    """
    返回单行成本分项（小数收益口径，便于直接落表）。
    """
    parts_bp = _cost_components_bp(row)
    return {k: _normalize_bp(v) for k, v in parts_bp.items()}


def cost_breakdown_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    返回 DataFrame 级别的成本分项表。
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "cost_base_bp",
                "cost_slippage_bp",
                "cost_impact_bp",
                "cost_open_bp",
                "cost_seal_bp",
                "cost_amount_bp",
                "cost_price_bp",
                "cost_total_bp",
            ]
        )

    out = df.apply(lambda r: pd.Series(_cost_components_bp(r)), axis=1)
    # 统一转成小数收益口径
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float) / 10000.0
    return out


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

def _risk_regime_penalty(row: pd.Series, regime: str) -> float:
    regime_u = _safe_str(regime, default="RISK_ON").upper()
    if regime_u in ("RISK_OFF", "OFF", "DEFENSE"):
        return W_REGIME_DEFENSE
    return 0.0


def _risk_open_penalty(row: pd.Series) -> float:
    if _is_limit_open_risk(row):
        return W_LIMIT_OPEN
    return 0.0


def _risk_open_times_penalty(row: pd.Series) -> float:
    ot = _open_times(row)
    if not np.isnan(ot) and ot >= 2:
        return min(W_HIGH_OPEN_TIMES, 0.003 * ot)
    return 0.0


def _risk_volatility_penalty(row: pd.Series) -> float:
    amp = _amp_pct(row)
    if not np.isnan(amp) and amp >= 8:
        return min(W_VOLATILITY, 0.001 * (amp - 8) + 0.003)
    return 0.0


def _risk_liquidity_penalty(row: pd.Series) -> float:
    turnover = _turnover_rate(row)
    if np.isnan(turnover):
        return 0.0
    if turnover < 3:
        return W_LOW_LIQUIDITY
    if turnover < 6:
        return 0.005
    return 0.0


def _risk_extreme_turnover_penalty(row: pd.Series) -> float:
    turnover = _turnover_rate(row)
    if not np.isnan(turnover) and turnover > 25:
        return W_EXTREME_TURNOVER
    return 0.0


def _risk_theme_penalty(row: pd.Series) -> float:
    theme_heat = _theme_heat_score(row)
    if np.isnan(theme_heat):
        return 0.0
    if theme_heat >= 8:
        return W_HOT_THEME
    if theme_heat >= 6:
        return 0.003
    return 0.0


def _risk_board_penalty(row: pd.Series) -> float:
    bucket = _board_bucket(row)
    if bucket in ("STAR", "GROWTH"):
        return W_GEM_STAR
    return 0.0


def _risk_seal_penalty(row: pd.Series) -> float:
    seal = _seal_strength(row)
    if np.isnan(seal):
        return 0.0
    if seal <= 2_000_000:
        return W_WEAK_SEAL_VERY_LOW
    if seal <= 5_000_000:
        return W_WEAK_SEAL_LOW
    return 0.0


def _risk_components(row: pd.Series, regime: str = "RISK_ON") -> dict[str, float]:
    parts = {
        "risk_regime_penalty": _risk_regime_penalty(row, regime),
        "risk_open_penalty": _risk_open_penalty(row),
        "risk_open_times_penalty": _risk_open_times_penalty(row),
        "risk_volatility_penalty": _risk_volatility_penalty(row),
        "risk_liquidity_penalty": _risk_liquidity_penalty(row),
        "risk_extreme_turnover_penalty": _risk_extreme_turnover_penalty(row),
        "risk_theme_penalty": _risk_theme_penalty(row),
        "risk_board_penalty": _risk_board_penalty(row),
        "risk_seal_penalty": _risk_seal_penalty(row),
    }
    total = float(sum(parts.values()))
    parts["risk_total_penalty"] = _clip(total, 0.0, RISK_PENALTY_CAP)
    return parts


def _risk_penalty_from_row(row: pd.Series, regime: str = "RISK_ON") -> float:
    parts = _risk_components(row, regime=regime)
    return float(parts["risk_total_penalty"])


def risk_breakdown_row(row: pd.Series, regime: str = "RISK_ON") -> dict[str, float]:
    return _risk_components(row, regime=regime)


def risk_breakdown_df(regime: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    返回 DataFrame 级别的风险惩罚分项表。
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "risk_regime_penalty",
                "risk_open_penalty",
                "risk_open_times_penalty",
                "risk_volatility_penalty",
                "risk_liquidity_penalty",
                "risk_extreme_turnover_penalty",
                "risk_theme_penalty",
                "risk_board_penalty",
                "risk_seal_penalty",
                "risk_total_penalty",
            ]
        )

    return df.apply(lambda r: pd.Series(_risk_components(r, regime=regime)), axis=1)


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
# EV 辅助函数
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
