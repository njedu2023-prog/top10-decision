#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Backfill Premium TOP10 T-auction to T+1 11:00 execution truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig  # noqa: E402
from top10decision.premium.execution_truth import (  # noqa: E402
    DEFAULT_FETCH_BUDGET,
    DEFAULT_SELL_TIME,
    DEFAULT_START_DATE,
    build_and_write_execution_truth,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--sell-time", default=DEFAULT_SELL_TIME)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--fetch-budget", type=int, default=DEFAULT_FETCH_BUDGET)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = PremiumConfig.load()
    result, paths = build_and_write_execution_truth(
        cfg.out_root(),
        cfg.market_cache_root(),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        cost_bps=args.cost_bps,
        sell_time=str(args.sell_time),
        fetch_budget=max(0, int(args.fetch_budget)),
    )
    summary = result.summary
    print(
        "[premium-execution-truth] "
        f"records={summary.get('records', 0)} "
        f"ready={summary.get('ready_records', 0)} "
        f"eligible={summary.get('model_eligible_records', 0)} "
        f"complete_days={summary.get('truth_complete_days', 0)} "
        f"pending={summary.get('pending_records', 0)} "
        f"missing={summary.get('missing_records', 0)} "
        f"api_requests={summary.get('api_requests', 0)} "
        f"fetch_errors={summary.get('fetch_error_count', 0)}"
    )
    if args.verbose:
        for name, path in paths.items():
            print(f"[premium-execution-truth] {name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
