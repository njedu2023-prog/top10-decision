#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally train Premium only when enough new T+1 truth is available."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig
from top10decision.premium.train import train_models


def _read_json(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _date_text(value: object) -> str:
    raw = str(value or "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return "".join(ch for ch in raw if ch.isdigit())[:8]


def _matured_days(out_root: Path) -> Set[str]:
    days: Set[str] = set()
    for path in sorted(out_root.glob("premium_verify_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if df.empty or "t1_verify_ready" not in df.columns:
            continue
        ready = pd.to_numeric(df["t1_verify_ready"], errors="coerce").fillna(0).eq(1)
        if not ready.any():
            continue
        d_col = "trade_date" if "trade_date" in df.columns else "d_analysis_trade_date"
        t_col = "buy_date" if "buy_date" in df.columns else "t_trade_date"
        t1_col = "target_date" if "target_date" in df.columns else "t1_trade_date"
        if not all(c in df.columns for c in (d_col, t_col, t1_col)):
            continue
        d = df[d_col].map(_date_text)
        t = df[t_col].map(_date_text)
        t1 = df[t1_col].map(_date_text)
        valid = ready & d.lt(t) & t.lt(t1)
        days.update(d.loc[valid & d.str.len().eq(8)].tolist())
    return days


def _write_state(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental Premium model training gate")
    parser.add_argument("--min-new-days", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PremiumConfig.load()
    out_root = cfg.out_root()
    model_dir = out_root / "models"
    meta = _read_json(model_dir / "limitup_probability_engine_meta.json")
    state_path = model_dir / "auto_train_state.json"
    state = _read_json(state_path)
    days = _matured_days(out_root)
    current_days = len(days)
    previous_days = max(int(meta.get("n_days", 0) or 0), int(state.get("attempted_n_days", 0) or 0))
    new_days = max(0, current_days - previous_days)
    has_candidate = (model_dir / "limitup_probability_engine_candidate.joblib").exists()
    should_train = bool(
        args.force
        or (current_days >= 4 and not has_candidate)
        or new_days >= max(1, int(args.min_new_days))
    )

    if args.verbose:
        print(
            f"[premium][auto-train] matured_days={current_days} previous_days={previous_days} "
            f"new_days={new_days} threshold={args.min_new_days} should_train={should_train}"
        )

    now = datetime.now(timezone.utc).isoformat()
    if not should_train:
        _write_state(state_path, {
            **state,
            "checked_at_utc": now,
            "matured_n_days": current_days,
            "previous_n_days": previous_days,
            "new_n_days": new_days,
            "decision": "skip_no_increment",
        })
        return 0

    result = train_models(cfg)
    _write_state(state_path, {
        "checked_at_utc": now,
        "attempted_n_days": current_days,
        "matured_n_days": current_days,
        "new_n_days": new_days,
        "decision": "trained",
        "trained": bool(getattr(result, "trained", False)),
        "reason": str(getattr(result, "reason", "")),
        "n_samples": int(getattr(result, "n_samples", 0) or 0),
        "n_days": int(getattr(result, "n_days", 0) or 0),
    })
    if args.verbose:
        print(
            f"[premium][auto-train] result trained={getattr(result, 'trained', False)} "
            f"reason={getattr(result, 'reason', '')}"
        )
    # A challenger failing promotion is a valid, safe outcome; prediction continues.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
