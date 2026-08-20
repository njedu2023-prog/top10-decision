#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.backfill_decision_v11_history import (  # noqa: E402
    _read_csv,
    _sha256_frame,
)
from top10decision.auction_v3.config import (  # noqa: E402
    TARGET_HISTORY_DATES,
    TARGET_INDEPENDENT_OOS_DATES,
)
from top10decision.writers.io_contract import (  # noqa: E402
    is_a_share_trading_day,
    next_a_share_trading_day,
)


def _date(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def validate_history_backfill(root: Path) -> dict[str, object]:
    history_root = (
        root
        / "data"
        / "auction_v3"
        / "history"
        / "tplus1_open_0930_v1"
    )
    manifest_path = history_root / "manifest_latest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"history manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "decision_v11_history_manifest_v1":
        raise RuntimeError("unsupported Decision history manifest")
    if payload.get("strict_calendar") is not True:
        raise RuntimeError("Decision history manifest is not strict-calendar data")

    target_window = int(payload.get("target_window_open_sessions") or 0)
    compact_dates = int(payload.get("total_compact_signal_dates") or 0)
    independent_target = int(payload.get("target_independent_dates") or 0)
    if target_window < TARGET_HISTORY_DATES:
        raise RuntimeError(
            f"history window {target_window} < required {TARGET_HISTORY_DATES}"
        )
    if compact_dates < TARGET_HISTORY_DATES:
        raise RuntimeError(
            f"compact history dates {compact_dates} < required {TARGET_HISTORY_DATES}"
        )
    if independent_target != TARGET_INDEPENDENT_OOS_DATES:
        raise RuntimeError(
            "independent OOS target drift: "
            f"{independent_target} != {TARGET_INDEPENDENT_OOS_DATES}"
        )

    relative_output = Path(str(payload.get("output_file") or ""))
    if (
        not str(relative_output)
        or relative_output.is_absolute()
        or ".." in relative_output.parts
    ):
        raise RuntimeError("history manifest output_file is unsafe")
    output_path = (root / relative_output).resolve()
    try:
        output_path.relative_to(history_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "history manifest output_file must stay inside the immutable history root"
        ) from exc
    if not output_path.name.startswith("training_") or output_path.suffix != ".csv":
        raise RuntimeError("history manifest output_file is not a training CSV")
    history = _read_csv(output_path)
    if history.empty:
        raise RuntimeError(f"history output is empty: {output_path}")
    actual_sha = _sha256_frame(history)
    expected_sha = str(payload.get("output_sha256") or "")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"history output fingerprint drift: expected={expected_sha} actual={actual_sha}"
        )

    signal_dates = sorted(
        {
            _date(value)
            for value in history.get("signal_date", [])
            if _date(value)
        }
    )
    produced_dates = int(payload.get("produced_signal_dates") or 0)
    if len(signal_dates) != produced_dates:
        raise RuntimeError(
            f"produced signal-date count drift: {len(signal_dates)} != {produced_dates}"
        )
    closed_dates = [
        value for value in signal_dates if not is_a_share_trading_day(value)
    ]
    if closed_dates:
        raise RuntimeError(
            "non-trading signal dates in history output: " + ", ".join(closed_dates)
        )

    all_signal_dates: set[str] = set()
    history_files = sorted(history_root.glob("training_*.csv"))
    if not history_files:
        raise RuntimeError("Decision compact history has no training files")
    for path in history_files:
        frame = _read_csv(path)
        required = {"signal_date", "buy_date", "target_exit_date"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(
                f"history file {path.name} is missing strict date columns: {missing}"
            )
        chains = {
            (_date(row.signal_date), _date(row.buy_date), _date(row.target_exit_date))
            for row in frame[["signal_date", "buy_date", "target_exit_date"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        for signal_date, buy_date, exit_date in chains:
            if not all((signal_date, buy_date, exit_date)):
                raise RuntimeError(f"history file {path.name} has an incomplete date chain")
            if not all(
                is_a_share_trading_day(value)
                for value in (signal_date, buy_date, exit_date)
            ):
                raise RuntimeError(
                    f"history file {path.name} has a non-trading D/T/T+1 chain: "
                    f"{signal_date}/{buy_date}/{exit_date}"
                )
            if next_a_share_trading_day(signal_date) != buy_date:
                raise RuntimeError(
                    f"history file {path.name} has non-adjacent D/T dates: "
                    f"{signal_date}/{buy_date}"
                )
            if next_a_share_trading_day(buy_date) != exit_date:
                raise RuntimeError(
                    f"history file {path.name} has non-adjacent T/T+1 dates: "
                    f"{buy_date}/{exit_date}"
                )
            all_signal_dates.add(signal_date)
    if len(all_signal_dates) != compact_dates:
        raise RuntimeError(
            "compact history signal-date count drift: "
            f"{len(all_signal_dates)} != {compact_dates}"
        )
    return {
        "validated": True,
        "strict_calendar": True,
        "target_history_dates": TARGET_HISTORY_DATES,
        "target_independent_oos_dates": TARGET_INDEPENDENT_OOS_DATES,
        "total_compact_signal_dates": compact_dates,
        "validated_output_signal_dates": len(signal_dates),
        "validated_compact_signal_dates": len(all_signal_dates),
        "validated_history_files": len(history_files),
        "output_file": str(relative_output),
        "output_sha256": actual_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Decision history backfill without publishing current signals"
    )
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    result = validate_history_backfill(Path(args.root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
