#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.decision.action_plan import publish_action_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the unified Decision action-plan and latest-report index")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--report-date", default="", help="Execution/report date YYYYMMDD; defaults to latest report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dated, latest, index, plan = publish_action_plan(Path(args.root), args.report_date)
    print(
        json.dumps(
            {
                "dated": str(dated),
                "latest": str(latest),
                "index": str(index),
                "status_code": plan.get("status_code"),
                "formal_buy_count": plan.get("formal_buy_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
