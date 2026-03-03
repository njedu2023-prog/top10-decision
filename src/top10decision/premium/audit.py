#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium — Audit (Factor Packs / Degrade)

锁死要点：
- audit 只负责“记录/渲染”，不允许把主流程跑挂
- notes 兼容 dict / list / str / None
- 提供 make_audit_kv() 供 predict.py 结构化写入（向后兼容）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_lines(notes: Any) -> List[str]:
    """
    将 notes 统一转成 Markdown bullet lines。
    支持：
    - dict: {k: v}
    - list/tuple/set: ["a", "b"] 或 [{"k":1}, ...]
    - str: "..."
    - None: []
    """
    if notes is None:
        return []

    if isinstance(notes, dict):
        lines: List[str] = []
        for k in sorted(notes.keys()):
            v = notes.get(k)
            lines.append(f"- **{k}**: {v}")
        return lines

    if isinstance(notes, (list, tuple, set)):
        lines: List[str] = []
        xs = list(notes)
        for i, x in enumerate(xs, start=1):
            lines.append(f"- {i:02d}. {x}")
        return lines

    return [f"- {str(notes)}"]


def make_audit_kv(
    packs_used: Optional[List[str]] = None,
    packs_missing: Optional[List[str]] = None,
    degrade_mode: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    missing_files: Optional[List[str]] = None,
    notes: Any = None,
) -> Dict[str, Any]:
    """
    输出结构化审计 KV，给上游写入 json/csv/last_run 等（predict.py 依赖）。
    """
    packs_used = packs_used or []
    packs_missing = packs_missing or []
    degrade_mode = degrade_mode or ("degraded" if packs_missing else "full")
    missing_fields = missing_fields or []
    missing_files = missing_files or []

    # notes 统一为 list[str]，方便落盘与审计
    if notes is None:
        notes_norm: List[str] = []
    elif isinstance(notes, dict):
        notes_norm = [f"{k}: {notes.get(k)}" for k in sorted(notes.keys())]
    elif isinstance(notes, (list, tuple, set)):
        notes_norm = [str(x) for x in list(notes)]
    else:
        notes_norm = [str(notes)]

    return {
        "degrade_mode": degrade_mode,
        "packs_used": list(packs_used),
        "packs_missing": list(packs_missing),
        "missing_fields": list(missing_fields),
        "missing_files": list(missing_files),
        "notes": notes_norm,
    }


def make_audit_block_md(
    packs_used: Optional[List[str]] = None,
    packs_missing: Optional[List[str]] = None,
    degrade_mode: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    missing_files: Optional[List[str]] = None,
    notes: Any = None,
) -> str:
    """
    输出一段 Markdown 审计块，供 premium_latest.md / report_md 拼接用。
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
