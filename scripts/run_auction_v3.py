#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.auction_v3 import AuctionV3Config, AuctionV3Engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Decision V8 manual auction guidance with strict probability "
            "calibration, conformal return bounds and auction truth"
        )
    )
    parser.add_argument("--signal-date", default="", help="D-day signal date, YYYYMMDD; default latest frozen pred source")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--force-prediction", action="store_true", help="Replace a dated prediction snapshot; disabled by default")
    parser.add_argument("--order-amount", type=float, default=100_000.0, help="Reference amount used only by auction-capacity simulation; no order is sent")
    parser.add_argument("--round-trip-cost-bps", type=float, default=35.0, help="Commission, taxes and fees excluding modeled slippage")
    parser.add_argument("--slippage-bps-each-side", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AuctionV3Config(
        root=Path(args.root).resolve(),
        order_amount_cny=max(0.0, args.order_amount),
        round_trip_cost_bps=max(0.0, args.round_trip_cost_bps),
        slippage_bps_each_side=max(0.0, args.slippage_bps_each_side),
    )
    result = AuctionV3Engine(config).run(args.signal_date, force_prediction=args.force_prediction)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
