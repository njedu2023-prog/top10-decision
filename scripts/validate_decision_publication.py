#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.writers.io_contract import (  # noqa: E402
    is_a_share_trading_day,
    next_a_share_trading_day,
)


def _date(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _single_date(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    values = sorted({_date(value) for value in frame[column] if _date(value)})
    return values[0] if len(values) == 1 else ""


def validate_publication(root: Path) -> dict[str, object]:
    action_path = root / "outputs" / "decision" / "action_plan_latest.json"
    index_path = root / "outputs" / "decision" / "report_index.json"
    prediction_path = (
        root / "outputs" / "auction_v3" / "predictions" / "pred_latest.csv"
    )
    for path in (action_path, index_path, prediction_path):
        if not path.is_file():
            raise RuntimeError(f"required Decision publication missing: {path}")

    action = json.loads(action_path.read_text(encoding="utf-8"))
    report_index = json.loads(index_path.read_text(encoding="utf-8"))
    prediction = pd.read_csv(prediction_path, low_memory=False)
    if prediction.empty:
        raise RuntimeError("latest Decision prediction is empty")

    signal_date = _date(action.get("signal_date"))
    exec_date = _date(action.get("exec_date"))
    exit_date = _date(action.get("exit_date"))
    report_date = _date(action.get("report_date"))
    if not all((signal_date, exec_date, exit_date, report_date)):
        raise RuntimeError("latest Decision action plan has incomplete D/T/T+1 dates")
    if report_date != exec_date:
        raise RuntimeError(
            f"latest report_date={report_date} does not match exec_date={exec_date}"
        )
    if _date(report_index.get("latest_report_date")) != report_date:
        raise RuntimeError("report index does not point to the latest action plan")

    for label, value in (
        ("D", signal_date),
        ("T", exec_date),
        ("T+1", exit_date),
    ):
        if not is_a_share_trading_day(value):
            raise RuntimeError(f"{label}={value} is not an A-share trading day")
    if next_a_share_trading_day(signal_date) != exec_date:
        raise RuntimeError("T is not the strict next A-share trading day after D")
    if next_a_share_trading_day(exec_date) != exit_date:
        raise RuntimeError("T+1 is not the strict next A-share trading day after T")

    prediction_dates = {
        "signal_date": _single_date(prediction, "signal_date"),
        "exec_date": _single_date(prediction, "expected_buy_date"),
        "exit_date": _single_date(prediction, "expected_exit_date"),
    }
    expected_dates = {
        "signal_date": signal_date,
        "exec_date": exec_date,
        "exit_date": exit_date,
    }
    if prediction_dates != expected_dates:
        raise RuntimeError(
            "latest prediction/action dates disagree: "
            f"prediction={prediction_dates} action={expected_dates}"
        )

    model = action.get("model") or {}
    if model.get("prediction_matches_report") is not True:
        raise RuntimeError("latest action plan is not backed by the matching prediction")
    if model.get("artifact_versions_match") is not True:
        raise RuntimeError("latest action plan model version is inconsistent")
    if model.get("artifact_fingerprints_match") is not True:
        raise RuntimeError("latest action plan model fingerprint is inconsistent")
    if model.get("trade_selector_artifacts_match") is not True:
        raise RuntimeError("latest action plan trade-selector fingerprint is inconsistent")
    status_code = str(action.get("status_code") or "")
    if not status_code or status_code.startswith("PENDING_"):
        raise RuntimeError(f"latest action plan is incomplete: {status_code or 'missing'}")

    return {
        "validated": True,
        "calendar": "strict_A_share_exchange_calendar",
        "signal_date": signal_date,
        "exec_date": exec_date,
        "exit_date": exit_date,
        "report_date": report_date,
        "status_code": status_code,
        "prediction_rows": int(len(prediction)),
        "model_artifact_sha256": str(model.get("artifact_sha256") or ""),
        "trade_selector_artifact_sha256": str(
            model.get("trade_selector_artifact_sha256") or ""
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when the latest Decision page data is partial or stale"
    )
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    result = validate_publication(Path(args.root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
