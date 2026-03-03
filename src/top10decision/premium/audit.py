#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium — Audit (Factor Packs / Degrade)

目标（锁死）：
- 不管 notes 是 dict 还是 list，都能安全渲染成 Markdown
- 不允许 audit 渲染阶段把主流程跑挂
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union


def _to_lines(notes: Any) -> List[str]:
    """
    将 notes 统一转成 Markdown 的 bullet lines。
    支持：
    - dict: {k: v}
    - list/tuple/set: ["a", "b"] 或 [{"k":1}, ...]
    - str: "..."
    - None: []
    """
    if notes is None:
        return []

    # dict
    if isinstance(notes, dict):
        lines: List[str] = []
        for k in sorted(notes.keys()):
            v = notes.get(k)
            lines.append(f"- **{k}**: {v}")
        return lines

    # list/tuple/set
    if isinstance(notes, (list, tuple, set)):
        lines = []
        for i, x in enumerate(list(notes), start=1):
            # 每条都转字符串，避免复杂对象报错
            lines.append(f"- {i:02d}. {x}")
        return lines

    # str / other
    return [f"- {str(notes)}"]


def make_audit_block_md(
    packs_used: Optional[List[str]] = None,
    packs_missing: Optional[List[str]] = None,
    degrade_mode: Optional[str] = None,
    missing_fields: Optional[List[str]] = None,
    missing_files: Optional[List[str]] = None,
    notes: Any = None,
) -> str:
    """
    输出一段 Markdown 审计块，给 premium_latest.md / report_md 拼接用。
    """
    packs_used = packs_used or []
    packs_missing = packs_missing or []
    degrade_mode = degrade_mode or ("degraded" if packs_missing else "full")
    missing_fields = missing_fields or []
    missing_files = missing_files or []

    md = []
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
