#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.writers.io_contract import next_a_share_trading_day


ROOT = Path(__file__).resolve().parents[1]


def candidate_dates(root: Path) -> list[str]:
    archive = root / "data" / "pred" / "archive"
    dates: list[str] = []
    for path in archive.glob("pred_source_20*.csv"):
        match = re.fullmatch(r"pred_source_(20\d{6})\.csv", path.name)
        if match:
            dates.append(match.group(1))
    return sorted(set(dates))


def _validate_published_action(
    root: Path,
    signal_date: str,
    report_date: str,
) -> Path:
    expected_exit_date = next_a_share_trading_day(report_date)
    path = root / "outputs" / "decision" / f"action_plan_{report_date}.json"
    if not path.is_file():
        raise RuntimeError(f"dated Decision action plan was not published: {path}")

    plan = json.loads(path.read_text(encoding="utf-8"))
    actual_dates = {
        "signal_date": str(plan.get("signal_date") or ""),
        "exec_date": str(plan.get("exec_date") or ""),
        "exit_date": str(plan.get("exit_date") or ""),
        "report_date": str(plan.get("report_date") or ""),
    }
    expected_dates = {
        "signal_date": signal_date,
        "exec_date": report_date,
        "exit_date": expected_exit_date,
        "report_date": report_date,
    }
    if actual_dates != expected_dates:
        raise RuntimeError(
            "dated Decision action plan has the wrong strict-calendar chain: "
            f"actual={actual_dates} expected={expected_dates}"
        )
    if (plan.get("model") or {}).get("prediction_matches_report") is not True:
        raise RuntimeError(
            f"dated Decision action plan does not match D={signal_date} prediction"
        )
    if plan.get("status_code") == "PENDING_AUCTION_MODEL":
        raise RuntimeError(
            f"dated Decision action plan remained pending after D={signal_date} backfill"
        )
    return path


def _run_signal(root: Path, signal_date: str, order_amount: float) -> None:
    report_date = next_a_share_trading_day(signal_date)
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
        [
            sys.executable,
            "scripts/publish_decision_action.py",
            "--root",
            str(root),
            "--report-date",
            report_date,
        ],
        cwd=root,
        check=True,
    )
    _validate_published_action(root, signal_date, report_date)


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
