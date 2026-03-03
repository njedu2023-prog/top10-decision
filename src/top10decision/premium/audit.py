#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium — Audit helpers

职责：
- 统一生成 audit dict（可写入 _last_run.txt）
- 统一把 audit 信息插入到报告 Markdown（不改 report_md.py 也能落地）
"""

from __future__ import annotations

from typing import Dict, List


def make_audit_block_md(
    packs_used: List[str],
    packs_missing: List[str],
    degrade_mode: bool,
    missing_fields: List[str],
    notes: Dict[str, str] | None = None,
) -> str:
    notes = notes or {}
    lines = []
    lines.append("\n---\n")
    lines.append("## Factor Packs（本次因子包审计）\n")
    lines.append(f"- packs_used: {packs_used}\n")
    lines.append(f"- packs_missing: {packs_missing}\n")
    lines.append(f"- degrade_mode: {bool(degrade_mode)}\n")
    if missing_fields:
        lines.append(f"- missing_fields: {missing_fields}\n")
    if notes:
        lines.append("- pack_notes:\n")
        for k in sorted(notes.keys()):
            lines.append(f"  - {k}: {notes[k]}\n")
    return "".join(lines)


def make_audit_kv(extra_prefix: str, packs_used, packs_missing, degrade_mode, missing_fields, notes) -> Dict[str, object]:
    """
    给 _last_run.txt 用：key-value 平铺（避免嵌套 JSON 解析麻烦）
    """
    extra = {
        f"{extra_prefix}_packs_used": ",".join(packs_used or []),
        f"{extra_prefix}_packs_missing": ",".join(packs_missing or []),
        f"{extra_prefix}_degrade_mode": bool(degrade_mode),
        f"{extra_prefix}_missing_fields": ",".join(missing_fields or []),
    }
    if notes:
        for k, v in notes.items():
            extra[f"{extra_prefix}_note_{k}"] = str(v)
    return extra
