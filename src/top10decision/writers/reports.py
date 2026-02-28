#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

from top10decision.writers.io_contract import norm_ymd, REPORT_FMT, EVAL_FMT


def write_decision_report(exec_date: str, report_md: str) -> str:
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    ed = norm_ymd(exec_date) or "unknown"
    path = REPORT_FMT.format(yyyymmdd=ed)
    Path(path).write_text(report_md, encoding="utf-8")
    return path


def write_eval_json(exec_date: str, payload: dict) -> str:
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    ed = norm_ymd(exec_date) or "unknown"
    path = EVAL_FMT.format(yyyymmdd=ed)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
