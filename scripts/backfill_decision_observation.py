#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def candidate_dates(root: Path) -> list[str]:
    archive = root / "data" / "pred" / "archive"
    dates: list[str] = []
    for path in archive.glob("pred_source_20*.csv"):
        match = re.fullmatch(r"pred_source_(20\d{6})\.csv", path.name)
        if match:
            dates.append(match.group(1))
    return sorted(set(dates))


def _run_signal(root: Path, signal_date: str, order_amount: float) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_auction_v3.py",
            "--root",
            str(root),
            "--signal-date",
            signal_date,
            "--order-amount",
            str(order_amount),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/publish_decision_action.py"],
        cwd=root,
        check=True,
    )


def backfill_observation(
    root: Path,
    signal_date: str,
    *,
    order_amount: float = 100_000.0,
) -> dict[str, object]:
    root = root.resolve()
    signal_date = str(signal_date or "").strip()
    if not re.fullmatch(r"20\d{6}", signal_date):
        raise ValueError("signal_date must be YYYYMMDD")

    available = candidate_dates(root)
    if signal_date not in available:
        raise RuntimeError(
            f"frozen candidate snapshot is missing for D={signal_date}"
        )

    latest_signal_date = available[-1]
    run_dates = [signal_date]
    if signal_date != latest_signal_date:
        run_dates.append(latest_signal_date)

    for current in run_dates:
        action = "historical_backfill" if current == signal_date else "restore_latest"
        print(
            json.dumps(
                {"decision_observation_action": action, "signal_date": current},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        _run_signal(root, current, max(0.0, float(order_amount)))

    return {
        "backfilled_signal_date": signal_date,
        "latest_signal_date": latest_signal_date,
        "latest_restored": signal_date != latest_signal_date,
        "run_dates": run_dates,
        "statistics_policy": "retrospective_rows_excluded_from_forward_metrics",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill one dated Decision observation and restore current latest "
            "aliases before publication"
        )
    )
    parser.add_argument("--signal-date", required=True, help="D date YYYYMMDD")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--order-amount", type=float, default=100_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = backfill_observation(
        Path(args.root),
        args.signal_date,
        order_amount=args.order_amount,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
