#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium — Factor Packs Registry / Detection

锁死要点（V2）：
- Pack0 永久保底：永远可用（不允许因 Pack0 检测失败导致主流程退出）
- Pack1：可用则启用；不可用则自动降级并记录审计字段
- Pack2：✅ 固定软启动（soft mode）—— 永远启用、永远不作为降级原因
- Pack3：✅ 分钟/竞价软启动（soft mode）—— 优先吃 pred_source 中的分钟衍生字段
- 检测阶段绝不抛异常（最多返回 missing_fields / missing_files / notes）
- 兼容 cfg.market_daily_cache_path / cfg.market_daily_tpl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class PackStatus:
    packs_used: List[str]
    packs_missing: List[str]
    degrade_mode: str
    missing_fields: List[str]
    missing_files: List[str]
    notes: List[str]  # predict.py 需要 pack_status.notes


# --------------------------
# helpers
# --------------------------

def _safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _market_daily_path(cfg, trade_date: str) -> Path:
    """
    兼容 PremiumConfig 的两种实现：
    A) cfg.market_daily_cache_path(trade_date)
    B) repo_root / market_cache_dir / market_daily_tpl.format(trade_date=...)
    """
    fn = _safe_getattr(cfg, "market_daily_cache_path", None)
    if callable(fn):
        try:
            return Path(fn(trade_date))
        except Exception:
            pass

    repo_root_fn = _safe_getattr(cfg, "repo_root", None)
    repo_root = Path(repo_root_fn()).resolve() if callable(repo_root_fn) else Path.cwd().resolve()

    cache_dir = _safe_getattr(cfg, "market_cache_dir", "data/market")
    tpl = _safe_getattr(cfg, "market_daily_tpl", "daily_{trade_date}.csv")
    return (repo_root / cache_dir / tpl.format(trade_date=trade_date)).resolve()


def _file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


def _repo_root(cfg) -> Path:
    fn = _safe_getattr(cfg, "repo_root", None)
    if callable(fn):
        try:
            return Path(fn()).resolve()
        except Exception:
            pass
    return Path.cwd().resolve()


def _first_available(paths: List[Path]) -> Path | None:
    for path in paths:
        if _file_exists(path):
            return path
    return None


def _dedup(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# --------------------------
# Pack checks (detection only)
# --------------------------

def _pack0_ok(cfg, trade_date: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Pack0：永久保底，不做任何外部依赖检测。
    """
    return True, [], [], ["Pack0 baseline always on"]


def _pack1_ok(cfg, trade_date: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Pack1：依赖市场缓存（close_T 等）。
    规则：文件存在则 ok；不存在则 missing_files 触发降级。
    """
    miss_fields: List[str] = []
    miss_files: List[str] = []
    notes: List[str] = []

    td = str(trade_date)[:8]
    root = _repo_root(cfg)
    candidates = [
        root / "data" / "market" / f"features_base_{td}.csv",
        root / "data" / "market" / "raw" / td[:4] / td / "daily_basic.csv",
        _market_daily_path(cfg, td),
    ]
    p = _first_available(candidates)
    if p is None:
        miss_files.extend(str(x) for x in candidates)
        notes.append("Pack1 disabled: no base/daily-basic source")
        return False, miss_fields, miss_files, notes

    notes.append(f"Pack1 enabled: market cache found -> {p}")
    return True, miss_fields, miss_files, notes


def _pack2_soft(cfg, trade_date: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Pack2：✅ 固定软启动（soft mode）
    - 永远启用
    - 不检查依赖
    - 不会导致 degrade
    """
    td = str(trade_date)[:8]
    root = _repo_root(cfg)
    candidates = [
        root / "data" / "market" / f"features_limit_{td}.csv",
        root / "data" / "market" / "raw" / td[:4] / td / "limit_list_d.csv",
        root / "data" / "limit" / f"limit_micro_{td}.csv",
    ]
    hit = _first_available(candidates)
    if hit is None:
        return False, [], [str(x) for x in candidates], ["Pack2 unavailable: no limit-structure source"]
    return True, [], [], [f"Pack2 enabled: {hit}"]


def _pack3_intraday_soft(cfg, trade_date: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Pack3：分钟级 / 竞价结构因子。

    软启动原因：
    - a-top10 的 pred_source_latest 已经可透传 intraday_* / auction_* 字段；
    - 原始 intraday_features / stk_auction 文件可能按日期或 latest 存在；
    - 缺失时应降为中性因子，而不是阻断 Premium 主流程。
    """
    td = str(trade_date)[:8]
    root = _repo_root(cfg)
    candidates = [
        root / "data" / "market" / f"features_limit_{td}.csv",
        root / "data" / "market" / "raw" / td[:4] / td / "intraday_features.csv",
        root / "data" / "market" / "raw" / td[:4] / td / "stk_auction.csv",
    ]
    hit = _first_available(candidates)
    if hit is None:
        return False, [], [str(x) for x in candidates], ["Pack3 unavailable: no intraday/auction source"]
    return True, [], [], [f"Pack3 enabled: {hit}"]


# --------------------------
# public API
# --------------------------

def detect_factor_packs(cfg, trade_date: str) -> PackStatus:
    """
    主入口：返回 pack 使用情况与降级信息。
    绝不抛异常。
    """
    packs_used: List[str] = []
    packs_missing: List[str] = []
    missing_fields: List[str] = []
    missing_files: List[str] = []
    notes: List[str] = []

    # Pack0 永久保底
    ok0, mf0, mfile0, n0 = _pack0_ok(cfg, trade_date)
    packs_used.append("Pack0")
    missing_fields.extend(mf0)
    missing_files.extend(mfile0)
    notes.extend(n0)

    # Pack1（唯一可能触发降级的 pack）
    ok1, mf1, mfile1, n1 = _pack1_ok(cfg, trade_date)
    notes.extend(n1)
    if ok1:
        packs_used.append("Pack1")
    else:
        packs_missing.append("Pack1")
        missing_fields.extend(mf1)
        missing_files.extend(mfile1)

    # Pack2: soft at scoring time, but registry reports real availability.
    ok2, mf2, mfile2, n2 = _pack2_soft(cfg, trade_date)
    notes.extend(n2)
    if ok2:
        packs_used.append("Pack2")
    else:
        packs_missing.append("Pack2")
        missing_fields.extend(mf2)
        missing_files.extend(mfile2)

    # Pack3: soft at scoring time, but never claim full mode without a source.
    ok3, mf3, mfile3, n3 = _pack3_intraday_soft(cfg, trade_date)
    notes.extend(n3)
    if ok3:
        packs_used.append("Pack3")
    else:
        packs_missing.append("Pack3")
        missing_fields.extend(mf3)
        missing_files.extend(mfile3)

    # Any unavailable pack is visible in audit output; Pack0 still guarantees a run.
    degrade_mode = "degraded" if packs_missing else "full"

    return PackStatus(
        packs_used=_dedup(packs_used),
        packs_missing=_dedup(packs_missing),
        degrade_mode=degrade_mode,
        missing_fields=_dedup(missing_fields),
        missing_files=_dedup(missing_files),
        notes=_dedup(notes),
    )


__all__ = ["PackStatus", "detect_factor_packs"]
