#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium — Factor Packs Registry (一次性写死机制)

职责：
- 定义 Pack0/1/2 的启用条件（文件存在 + 字段齐）
- 自动选择可用 packs（Auto-select）
- 自动降级（Auto-degrade）：缺数据/缺字段时跳过增强包，不中断主流程
- 输出审计字段：packs_used / packs_missing / degrade_mode / missing_fields
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import PremiumConfig


@dataclass
class PackStatus:
    packs_used: List[str]
    packs_missing: List[str]
    degrade_mode: bool
    missing_fields: List[str]
    notes: Dict[str, str]


# ---- 内部：文件/字段检测 ----

def _safe_read_head(path: Path, n: int = 5) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=n)
    except Exception:
        try:
            return pd.read_csv(path, nrows=n, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path, nrows=n, encoding="gbk")
            except Exception:
                return None


def _cols_lower(df: pd.DataFrame) -> set:
    return set([str(c).strip().lower() for c in df.columns])


def _has_fields(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    cols = _cols_lower(df)
    missing = [f for f in required if f.lower() not in cols]
    return (len(missing) == 0), missing


# ---- Pack 启用判定（一次性写死） ----

def _pack0_ok(cfg: PremiumConfig, trade_date: str) -> Tuple[bool, str, List[str]]:
    # Pack0 只依赖本仓库落盘：pred_source_latest + market daily
    pred = cfg.pred_source_latest_path()
    market = cfg.market_daily_path(trade_date)

    missing_fields: List[str] = []
    if not pred.exists():
        return False, f"missing_file:{pred}", ["pred_source_latest"]
    if not market.exists():
        # 允许当日真值尚未缓存，但 pack0 仍算得出“先验因子”
        # 这里判定为 OK（由主流程 pending 处理 close_T）
        return True, f"market_daily_missing_allow_pending:{market}", []

    # 字段最低要求：pred_source 要有 ts_code/trade_date；market daily 要有 close
    dfp = _safe_read_head(pred)
    if dfp is None:
        return False, f"cannot_read:{pred}", ["pred_source_latest_read"]
    ok, miss = _has_fields(dfp, ["ts_code", "trade_date"])
    if not ok:
        missing_fields += [f"pred_source:{m}" for m in miss]

    dfm = _safe_read_head(market)
    if dfm is None:
        return True, f"cannot_read_market_allow_pending:{market}", missing_fields
    ok2, miss2 = _has_fields(dfm, ["ts_code", "trade_date", "close"])
    if not ok2:
        missing_fields += [f"market_daily:{m}" for m in miss2]

    # 只要 pred_source 合格就算 pack0 可用；market 缺字段由 pending 兜底
    if missing_fields and any(x.startswith("pred_source:") for x in missing_fields):
        return False, "pred_source_fields_missing", missing_fields
    return True, "ok", missing_fields


def _pack1_ok(cfg: PremiumConfig, trade_date: str) -> Tuple[bool, str, List[str]]:
    # Pack1：优先读本地落盘 daily_basic（避免每次在线拉取）
    # 约定路径：data/market_basic/daily_basic_YYYYMMDD.csv
    p = cfg.market_basic_path(trade_date)
    if not p.exists():
        return False, f"missing_file:{p}", ["market_basic_file"]
    df = _safe_read_head(p)
    if df is None:
        return False, f"cannot_read:{p}", ["market_basic_read"]
    ok, miss = _has_fields(df, ["ts_code", "trade_date", "turnover_rate"])
    # circ_mv/total_mv 二选一；两者都没有就算缺
    cols = _cols_lower(df)
    has_mv = ("circ_mv" in cols) or ("total_mv" in cols)
    if not has_mv:
        miss.append("circ_mv|total_mv")
    return ok and has_mv, ("ok" if (ok and has_mv) else "fields_missing"), [f"market_basic:{m}" for m in miss]


def _pack2_ok(cfg: PremiumConfig, trade_date: str) -> Tuple[bool, str, List[str]]:
    # Pack2：涨停结构微观因子（未来你修好数据源后启用）
    # 约定路径：data/limit/limit_micro_YYYYMMDD.csv
    p = cfg.limit_micro_path(trade_date)
    if not p.exists():
        return False, f"missing_file:{p}", ["limit_micro_file"]
    df = _safe_read_head(p)
    if df is None:
        return False, f"cannot_read:{p}", ["limit_micro_read"]
    ok, miss = _has_fields(df, ["ts_code", "trade_date", "open_times", "seal_amount"])
    return ok, ("ok" if ok else "fields_missing"), [f"limit_micro:{m}" for m in miss]


# ---- 对外：自动选择 packs ----

def detect_factor_packs(cfg: PremiumConfig, trade_date: str) -> PackStatus:
    packs_used: List[str] = []
    packs_missing: List[str] = []
    missing_fields: List[str] = []
    notes: Dict[str, str] = {}

    ok0, r0, miss0 = _pack0_ok(cfg, trade_date)
    notes["Pack0_base"] = r0
    if ok0:
        packs_used.append("Pack0_base")
    else:
        packs_missing.append("Pack0_base")
        missing_fields += miss0
        return PackStatus(packs_used, packs_missing, True, missing_fields, notes)  # 没 pack0 直接降级失败

    ok1, r1, miss1 = _pack1_ok(cfg, trade_date)
    notes["Pack1_tushare_basic"] = r1
    if ok1:
        packs_used.append("Pack1_tushare_basic")
    else:
        packs_missing.append("Pack1_tushare_basic")
        missing_fields += miss1

    ok2, r2, miss2 = _pack2_ok(cfg, trade_date)
    notes["Pack2_limit_micro"] = r2
    if ok2:
        packs_used.append("Pack2_limit_micro")
    else:
        packs_missing.append("Pack2_limit_micro")
        missing_fields += miss2

    degrade_mode = (len(packs_missing) > 0)
    return PackStatus(packs_used, packs_missing, degrade_mode, sorted(set(missing_fields)), notes)
