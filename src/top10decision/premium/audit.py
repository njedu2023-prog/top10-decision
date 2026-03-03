#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium — Audit (Factor Packs / Degrade)

工程目标（锁死）：
- audit 只负责记录/渲染，不允许因接口不匹配把主流程跑挂
- notes 兼容 dict / list / str / None
- make_audit_kv() 必须向后兼容（允许额外 kwargs，例如 extra_prefix）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_lines(notes: Any) -> List[str]:
    if notes is None:
        return []
    if isinstance(notes, dict):
        return [f"- **{k}**: {notes.get(k)}" for k in sorted(notes.keys())]
    if isinstance(notes, (list, tuple, set)):
        xs = list(notes)
        return [f"- {i:02d}. {x}" for i, x in enumerate(xs, start=1)]
    return [f"- {str(notes)}"]


def make_audit_kv(
    packs_used: Optional[List[str]] = None,
    packs_missing: Optional[List[str]] = None,
    degrade_mode: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    missing_files: Optional[List[str]] = None,
    notes: Any = None,
    extra_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    输出结构化审计 KV（predict.py 依赖）。

    ✅ 向后兼容策略：
    - 接受 extra_prefix（predict.py 可能会传）
    - 接受任何未知 kwargs（未来扩展不再炸）
    """
    packs_used = packs_used or []
    packs_missing = packs_missing or []
    degrade_mode = degrade_mode or ("degraded" if packs_missing else "full")
    missing_fields = missing_fields or []
    missing_files = missing_files or []

    # notes 统一成 list[str]
    if notes is None:
        notes_norm: List[str] = []
    elif isinstance(notes, dict):
        notes_norm = [f"{k}: {notes.get(k)}" for k in sorted(notes.keys())]
    elif isinstance(notes, (list, tuple, set)):
        notes_norm = [str(x) for x in list(notes)]
    else:
        notes_norm = [str(notes)]

    kv = {
        "degrade_mode": degrade_mode,
        "packs_used": list(packs_used),
        "packs_missing": list(packs_missing),
        "missing_fields": list(missing_fields),
        "missing_files": list(missing_files),
        "notes": notes_norm,
    }

    # 可选：把 extra_prefix 记下来（不影响逻辑）
    if extra_prefix:
        kv["extra_prefix"] = str(extra_prefix)

    # 把未知 kwargs 也落进去（可追溯，但不必强依赖）
    for k, v in (kwargs or {}).items():
        if k not in kv:
            kv[k] = v

    return kv


def make_audit_block_md(
    packs_used: Optional[List[str]] = None,
    packs_missing: Optional[List[str]] = None,
    degrade_mode: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    missing_files: Optional[List[str]] = None,
    notes: Any = None,
    **kwargs,
) -> str:
    """
    输出 Markdown 审计块。
    ✅ 同样接受 **kwargs，避免上游未来加参导致崩溃。
    """
    packs_used = packs_used or []
    packs_missing = packs_missing or []
    degrade_mode = degrade_mode or ("degraded" if packs_missing else "full")
    missing_fields = missing_fields or []
    missing_files = missing_files or []

    md: List[str] = []
    md.append("\n---\n")
    md.append("## 审计（Factor Packs / Degrade）\n")
    md.append(f"- degrade_mode: **{degrade_mode}**\n")
    md.append(f"- packs_used: `{', '.join(packs_used) if packs_used else '-'}`\n")
    md.append(f"- packs_missing: `{', '.join(packs_missing) if packs_missing else '-'}`\n")

    if missing_fields:
        md.append("\n### missing_fields\n")
        for x in missing_fields:
            md.append(f"- {x}\n")

    if missing_files:
        md.append("\n### missing_files\n")
        for x in missing_files:
            md.append(f"- {x}\n")

    lines = _to_lines(notes)
    if lines:
        md.append("\n### notes\n")
        md.extend([l + "\n" for l in lines])

    md.append("\n")
    return "".join(md)


__all__ = ["make_audit_block_md", "make_audit_kv"]
