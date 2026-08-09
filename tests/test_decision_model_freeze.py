from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from top10decision.decision.model_freeze import (
    DecisionModelFreezeError,
    apply_frozen_history_cutoff,
    load_model_freeze,
    validate_pinned_files,
    validate_runtime_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DecisionModelFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {
            "schema_version": "decision_model_freeze_v1",
            "active": True,
            "freeze_id": "test-freeze",
            "training_cutoff_signal_date": "20260807",
            "production": {
                "model_version": "auction_v12",
                "model_artifact_sha256": "main-sha",
                "promoted": False,
                "trade_selector_version": "selector-v2",
                "trade_selector_artifact_sha256": "selector-sha",
                "trade_selector_promoted": False,
                "formal_status": "NO_TRADE_MODEL_NOT_PROMOTED",
                "formal_buy_count": 0,
            },
            "pinned_files": {},
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_history_cutoff_keeps_frozen_training_window(self) -> None:
        history = pd.DataFrame(
            {
                "signal_date": ["20260806", "20260807", "20260810"],
                "ts_code": ["A", "B", "C"],
            }
        )
        frozen, audit = apply_frozen_history_cutoff(history, self.manifest)
        self.assertEqual(frozen["ts_code"].tolist(), ["A", "B"])
        self.assertEqual(audit["rows_removed"], 1)
        self.assertEqual(audit["history_end"], "20260807")

    def test_pinned_file_drift_fails_closed(self) -> None:
        target = self.root / "models/frozen.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"frozen")
        self.manifest["pinned_files"] = {
            "models/frozen.bin": hashlib.sha256(b"frozen").hexdigest()
        }
        validate_pinned_files(self.root, self.manifest)
        target.write_bytes(b"changed")
        with self.assertRaises(DecisionModelFreezeError):
            validate_pinned_files(self.root, self.manifest)

    def test_inactive_manifest_does_not_enforce_pins(self) -> None:
        manifest = {
            "active": False,
            "freeze_id": "released",
            "pinned_files": {"missing.bin": "not-used"},
        }
        audit = validate_pinned_files(self.root, manifest)
        self.assertFalse(audit["active"])
        self.assertFalse(audit["enforced"])

    def test_runtime_fingerprints_must_match(self) -> None:
        _write_json(
            self.root / "outputs/auction_v3/models/model_meta_latest.json",
            {
                "model_version": "auction_v12",
                "model_artifact_sha256": "main-sha",
                "promoted": False,
                "data_coverage": {"history_end": "20260807"},
            },
        )
        _write_json(
            self.root / "outputs/auction_v3/metrics/backtest_latest.json",
            {
                "trade_selector": {
                    "version": "selector-v2",
                    "production_artifact_sha256": "selector-sha",
                    "promoted": False,
                }
            },
        )
        audit = validate_runtime_artifacts(self.root, self.manifest)
        self.assertTrue(audit["validated"])

        self.manifest["production"]["model_artifact_sha256"] = "other"
        with self.assertRaises(DecisionModelFreezeError):
            validate_runtime_artifacts(self.root, self.manifest)

    def test_action_plan_must_preserve_frozen_no_trade_state(self) -> None:
        _write_json(
            self.root / "outputs/auction_v3/models/model_meta_latest.json",
            {
                "model_version": "auction_v12",
                "model_artifact_sha256": "main-sha",
                "promoted": False,
                "data_coverage": {"history_end": "20260805"},
            },
        )
        _write_json(
            self.root / "outputs/auction_v3/metrics/backtest_latest.json",
            {
                "trade_selector": {
                    "version": "selector-v2",
                    "production_artifact_sha256": "selector-sha",
                    "promoted": False,
                }
            },
        )
        _write_json(
            self.root / "outputs/decision/action_plan_latest.json",
            {
                "status_code": "NO_TRADE_MODEL_NOT_PROMOTED",
                "formal_buy_count": 0,
                "model": {
                    "version": "auction_v12",
                    "artifact_sha256": "main-sha",
                    "promoted": False,
                    "trade_selector_artifact_sha256": "selector-sha",
                    "trade_selector": {
                        "version": "selector-v2",
                        "promoted": False,
                    },
                },
            },
        )
        audit = validate_runtime_artifacts(self.root, self.manifest)
        self.assertTrue(audit["action_plan_checks"]["formal_status"])

        action_plan = json.loads(
            (self.root / "outputs/decision/action_plan_latest.json").read_text()
        )
        action_plan["formal_buy_count"] = 1
        _write_json(
            self.root / "outputs/decision/action_plan_latest.json",
            action_plan,
        )
        with self.assertRaises(DecisionModelFreezeError):
            validate_runtime_artifacts(self.root, self.manifest)

    def test_manifest_requires_valid_cutoff(self) -> None:
        self.manifest["training_cutoff_signal_date"] = "2026-08-07"
        _write_json(
            self.root / "models/decision_model_freeze.json",
            self.manifest,
        )
        with self.assertRaises(DecisionModelFreezeError):
            load_model_freeze(self.root, required=True)


if __name__ == "__main__":
    unittest.main()
