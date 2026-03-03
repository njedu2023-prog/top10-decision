#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium — Factor Packs Registry / Detection

锁死要点：
- Pack0 永久保底：永远可用（不允许因 Pack0 检测失败导致主流程退出）
- Pack1/2：可用则启用；不可用则自动降级并记录审计字段
- 检测阶段绝不抛异常（最多返回 missing_fields / missing_files）
- 不绑定某个 config 方法名（兼容 cfg.market_daily_cache_path / cfg.market_daily_tpl）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PackStatus:
    packs_used: List[str]
    packs_missing: List[str]
    degrade_mode: str
    missing_fields: List[str]
    missing_files: List[str]


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
    A) cfg.market_daily_cache_path(trade_date)  ✅ 当前仓库是这个
    B) cfg.market_cache_dir + cfg.market_daily_tpl.format(trade_date=...)
    """
    fn = _safe_getattr(cfg, "market_daily_cache_path", None)
    if callable(fn):
        try:
            p = fn(trade_date)
            return Path(p)
        except Exception:
            pass

    repo_root = Path(_safe_getattr(cfg, "repo_root")()).resolve() if callable(_safe_getattr(cfg, "repo_root", None)) else Path.cwd().resolve()
    cache_dir = _safe_getattr(cfg, "market_cache_dir", "data/market")
    tpl = _safe_getattr(cfg, "market_daily_tpl", "daily_{trade_date}.csv")
    return (repo_root / cache_dir / tpl.format(trade_date=trade_date)).resolve()


def _file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


# --------------------------
# Pack checks (detection only)
# --------------------------

def _pack0_ok(cfg, trade_date: str) -> Tuple[bool, Dict[str, object], List[str], List[str]]:
    """
    Pack0：永久保底，不做任何外部依赖检测。
    只要主输入 pred_source_latest 能读到，Pack0 的 builder 就应该能跑。
    这里检测阶段永远 ok，避免因为检测导致系统退出。
    """
    report = {
        "pack": "Pack0",
        "level": 0,
        "ok": True,
        "reason": "baseline_always_on",
    }
    return True, report, [], []


def _pack1_ok(cfg, trade_date: str) -> Tuple[bool, Dict[str, object], List[str], List[str]]:
    """
    Pack1：典型依赖 = 市场缓存（close_T 等）。
    规则：文件存在则 ok；不存在则 missing_files 触发降级。
    """
    miss_fields: List[str] = []
    miss_files: List[str] = []

    p = _market_daily_path(cfg, trade_date)
    if not _file_exists(p):
        miss_files.append(str(p))

    ok = (len(miss_files) == 0)
    report = {
        "pack": "Pack1",
        "level": 1,
        "ok": ok,
        "required_file": str(p),
        "reason": "ok" if ok else "market_daily_cache_missing",
    }
    return ok, report, miss_fields, miss_files


def _pack2_ok(cfg, trade_date: str) -> Tuple[bool, Dict[str, object], List[str], List[str]]:
    """
    Pack2：更高阶（例如：需要更多标签/决策字段/审计历史等）。
    在你当前主线里，Pack2 先做“软检测”：默认不强制依赖文件，
    只要未来你在 builder 里声明了硬依赖，再补充到这里。
    """
    # 先不强依赖任何外部文件，避免误杀主线
    ok = True
    report = {
        "pack": "Pack2",
        "level": 2,
        "ok": ok,
        "reason": "soft_on_by_default",
    }
    return ok, report, [], []


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

    # Pack0 永久保底
    ok0, r0, mf0, mfile0 = _pack0_ok(cfg, trade_date)
    packs_used.append("Pack0")
    missing_fields.extend(mf0)
    missing_files.extend(mfile0)

    # Pack1
    ok1, r1, mf1, mfile1 = _pack1_ok(cfg, trade_date)
    if ok1:
        packs_used.append("Pack1")
    else:
        packs_missing.append("Pack1")
        missing_fields.extend(mf1)
        missing_files.extend(mfile1)

    # Pack2（软启用：你后续可以改成硬依赖）
    ok2, r2, mf2, mfile2 = _pack2_ok(cfg, trade_date)
    if ok2:
        packs_used.append("Pack2")
    else:
        packs_missing.append("Pack2")
        missing_fields.extend(mf2)
        missing_files.extend(mfile2)

    # degrade_mode
    if packs_missing:
        degrade_mode = "degraded"
    else:
        degrade_mode = "full"

    # 去重保持顺序
    def _dedup(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return PackStatus(
        packs_used=_dedup(packs_used),
        packs_missing=_dedup(packs_missing),
        degrade_mode=degrade_mode,
        missing_fields=_dedup([x for x in missing_fields if x]),
        missing_files=_dedup([x for x in missing_files if x]),
    )


__all__ = ["PackStatus", "detect_factor_packs"]
