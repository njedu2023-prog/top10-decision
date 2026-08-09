#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally train Premium when new or corrected T/T+1 truth is available."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig
from top10decision.premium.limitup_probability_engine import GATE_VERSION
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


_T1_TRUTH_COLS = [
    "t1_up_hit",
    "t1_high_profit_hit",
    "t1_accept_hit",
    "t1_fail_hit",
    "t1_big_drawdown_hit",
    "t1_close_ret",
    "t1_high_ret",
]

_TRUTH_FINGERPRINT_COLS = [
    "trade_date",
    "d_analysis_trade_date",
    "buy_date",
    "t_trade_date",
    "target_date",
    "t1_trade_date",
    "ts_code",
    "t_up_actual",
    "t_limitup_actual",
    "t_touch_limitup_actual",
    "t_high_profit_hit",
    "t_open_ret",
    "t_intraday_ret",
    "t_close_ret",
    *_T1_TRUTH_COLS,
]


def _ready_mask(df: pd.DataFrame, names: list[str]) -> pd.Series:
    ready = pd.Series(False, index=df.index, dtype=bool)
    for name in names:
        if name not in df.columns:
            continue
        raw = df[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        ready = ready | numeric.eq(1) | raw.astype(str).str.strip().str.lower().isin(
            {"true", "yes", "y", "ok", "ready"}
        )
    return ready


def _truth_snapshot(out_root: Path) -> Dict[str, object]:
    """Return matured T/T+1 day sets plus a deterministic truth-only digest."""
    t_days: Set[str] = set()
    t1_days: Set[str] = set()
    digest = hashlib.sha256()
    fingerprint_files = 0
    fingerprint_rows = 0
    for path in sorted(out_root.glob("premium_verify_*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if df.empty:
            continue
        t_ready = _ready_mask(df, ["t_limitup_verify_ready"])
        if not t_ready.any():
            t_ready = pd.to_numeric(
                df.get("t_limitup_actual", pd.Series(np.nan, index=df.index)),
                errors="coerce",
            ).notna()
        t1_ready = _ready_mask(df, ["t1_verify_ready", "label_matured"])
        if not t1_ready.any():
            t1_ready = pd.Series(False, index=df.index, dtype=bool)
            for col in _T1_TRUTH_COLS:
                if col in df.columns:
                    t1_ready = t1_ready | pd.to_numeric(df[col], errors="coerce").notna()
        if not t_ready.any() and not t1_ready.any():
            continue
        d_col = "trade_date" if "trade_date" in df.columns else "d_analysis_trade_date"
        t_col = "buy_date" if "buy_date" in df.columns else "t_trade_date"
        if not all(c in df.columns for c in (d_col, t_col)):
            continue
        d = df[d_col].map(_date_text)
        t = df[t_col].map(_date_text)
        valid_calendar = d.lt(t) & d.str.len().eq(8) & t.str.len().eq(8)
        t_valid = t_ready & valid_calendar
        t1_valid = t1_ready & valid_calendar
        t_days.update(d.loc[t_valid].tolist())
        t1_days.update(d.loc[t1_valid].tolist())

        truth_valid = (t_valid | t1_valid)
        if truth_valid.any():
            cols = [c for c in _TRUTH_FINGERPRINT_COLS if c in df.columns]
            canonical = df.loc[truth_valid, cols].copy()
            for col in canonical.columns:
                canonical[col] = canonical[col].astype("string").fillna("<NA>")
            sort_cols = [c for c in (d_col, "ts_code") if c in canonical.columns]
            if sort_cols:
                canonical = canonical.sort_values(sort_cols, kind="stable")
            payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
            fingerprint_files += 1
            fingerprint_rows += int(len(canonical))
    return {
        "t_days": t_days,
        "t1_days": t1_days,
        "truth_fingerprint": digest.hexdigest(),
        "fingerprint_files": fingerprint_files,
        "fingerprint_rows": fingerprint_rows,
    }


def _matured_days(out_root: Path) -> Set[str]:
    return set(_truth_snapshot(out_root)["t_days"])


def _training_decision(
    *,
    force: bool,
    current_t_days: int,
    previous_t_days: int,
    current_t1_days: int,
    previous_t1_days: int,
    min_new_days: int,
    has_candidate: bool,
    contract_upgrade: bool,
    truth_fingerprint: str,
    previous_fingerprint: str,
) -> tuple[bool, bool, int, int]:
    new_t_days = max(0, int(current_t_days) - int(previous_t_days))
    new_t1_days = max(0, int(current_t1_days) - int(previous_t1_days))
    truth_revision = bool(
        previous_fingerprint
        and previous_fingerprint != truth_fingerprint
        and new_t_days == 0
        and new_t1_days == 0
    )
    threshold = max(1, int(min_new_days))
    should_train = bool(
        force
        or (current_t_days >= 4 and contract_upgrade)
        or (current_t_days >= 4 and not has_candidate)
        or new_t_days >= threshold
        or new_t1_days >= threshold
        or truth_revision
    )
    return should_train, truth_revision, new_t_days, new_t1_days


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
    snapshot = _truth_snapshot(out_root)
    current_days = len(snapshot["t_days"])
    current_t1_days = len(snapshot["t1_days"])
    previous_days = max(
        int(meta.get("n_days", 0) or 0),
        int(state.get("attempted_t_days", state.get("attempted_n_days", 0)) or 0),
    )
    previous_t1_days = int(
        state.get(
            "attempted_t1_days",
            state.get("training_baseline_t1_days", current_t1_days),
        )
        or 0
    )
    truth_fingerprint = str(snapshot["truth_fingerprint"])
    previous_fingerprint = str(state.get("truth_fingerprint", "") or "")
    has_candidate = (model_dir / "limitup_probability_engine_candidate.joblib").exists()
    contract_upgrade = str(meta.get("gate_version", "")) != GATE_VERSION
    should_train, truth_revision, new_days, new_t1_days = _training_decision(
        force=bool(args.force),
        current_t_days=current_days,
        previous_t_days=previous_days,
        current_t1_days=current_t1_days,
        previous_t1_days=previous_t1_days,
        min_new_days=int(args.min_new_days),
        has_candidate=has_candidate,
        contract_upgrade=contract_upgrade,
        truth_fingerprint=truth_fingerprint,
        previous_fingerprint=previous_fingerprint,
    )

    if args.verbose:
        print(
            f"[premium][auto-train] t_days={current_days} previous_t_days={previous_days} "
            f"new_t_days={new_days} t1_days={current_t1_days} "
            f"previous_t1_days={previous_t1_days} new_t1_days={new_t1_days} "
            f"threshold={args.min_new_days} truth_revision={truth_revision} "
            f"contract_upgrade={contract_upgrade} should_train={should_train}"
        )

    now = datetime.now(timezone.utc).isoformat()
    if not should_train:
        _write_state(state_path, {
            **state,
            "checked_at_utc": now,
            "matured_n_days": current_days,
            "matured_t_days": current_days,
            "matured_t1_days": current_t1_days,
            "training_baseline_t1_days": previous_t1_days,
            "previous_n_days": previous_days,
            "new_n_days": new_days,
            "new_t_days": new_days,
            "new_t1_days": new_t1_days,
            "truth_fingerprint": truth_fingerprint,
            "truth_fingerprint_files": int(snapshot["fingerprint_files"]),
            "truth_fingerprint_rows": int(snapshot["fingerprint_rows"]),
            "truth_revision": truth_revision,
            "gate_version": GATE_VERSION,
            "contract_upgrade": contract_upgrade,
            "decision": "skip_no_increment",
        })
        return 0

    result = train_models(cfg)
    _write_state(state_path, {
        "checked_at_utc": now,
        "attempted_n_days": current_days,
        "attempted_t_days": current_days,
        "attempted_t1_days": current_t1_days,
        "training_baseline_t1_days": current_t1_days,
        "matured_n_days": current_days,
        "matured_t_days": current_days,
        "matured_t1_days": current_t1_days,
        "new_n_days": new_days,
        "new_t_days": new_days,
        "new_t1_days": new_t1_days,
        "truth_fingerprint": truth_fingerprint,
        "truth_fingerprint_files": int(snapshot["fingerprint_files"]),
        "truth_fingerprint_rows": int(snapshot["fingerprint_rows"]),
        "truth_revision": truth_revision,
        "gate_version": GATE_VERSION,
        "contract_upgrade": contract_upgrade,
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
