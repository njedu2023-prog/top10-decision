from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.backfill_decision_v11_history import (  # noqa: E402
    _read_csv,
    _sha256_frame,
    _write_csv,
)
from scripts.backfill_decision_observation import (  # noqa: E402
    backfill_observation,
)
from scripts.validate_decision_history_backfill import (  # noqa: E402
    validate_history_backfill,
)
from scripts.validate_decision_publication import validate_publication  # noqa: E402


class DecisionAutonomyTests(unittest.TestCase):
    def test_historical_observation_backfill_restores_latest_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            archive = root / "data" / "pred" / "archive"
            archive.mkdir(parents=True)
            for signal_date in ("20260818", "20260820"):
                (archive / f"pred_source_{signal_date}.csv").write_text(
                    "trade_date,ts_code\n",
                    encoding="utf-8",
                )

            with mock.patch(
                "scripts.backfill_decision_observation.subprocess.run"
            ) as run:
                result = backfill_observation(root, "20260818")

            selector_dates = [
                call.args[0][call.args[0].index("--signal-date") + 1]
                for call in run.call_args_list
                if "scripts/run_auction_v3.py" in call.args[0]
            ]
            publish_calls = [
                call
                for call in run.call_args_list
                if "scripts/publish_decision_action.py" in call.args[0]
            ]
            self.assertEqual(selector_dates, ["20260818", "20260820"])
            self.assertEqual(len(publish_calls), 2)
            self.assertTrue(result["latest_restored"])
            self.assertEqual(result["latest_signal_date"], "20260820")
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["cwd"], root)
                self.assertTrue(call.kwargs["check"])

    def test_observation_backfill_requires_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "snapshot is missing"):
                backfill_observation(Path(temp), "20260818")

    def test_history_fingerprint_survives_csv_float_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.csv"
            before = pd.DataFrame(
                {
                    "signal_date": ["20260819"],
                    "ts_code": ["000001.SZ"],
                    "net_return": [0.1 + 0.2],
                    "optional": [None],
                }
            )
            _write_csv(before, path)
            after = _read_csv(path)
            self.assertEqual(_sha256_frame(before), _sha256_frame(after))

    def test_publication_requires_one_matching_strict_calendar_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decision_root = root / "outputs" / "decision"
            prediction_root = root / "outputs" / "auction_v3" / "predictions"
            decision_root.mkdir(parents=True)
            prediction_root.mkdir(parents=True)
            action = {
                "report_date": "20260820",
                "signal_date": "20260819",
                "exec_date": "20260820",
                "exit_date": "20260821",
                "status_code": "NO_TRADE_MODEL_NOT_PROMOTED",
                "model": {
                    "prediction_matches_report": True,
                    "artifact_versions_match": True,
                    "artifact_fingerprints_match": True,
                    "trade_selector_artifacts_match": True,
                    "artifact_sha256": "model",
                    "trade_selector_artifact_sha256": "selector",
                },
            }
            (decision_root / "action_plan_latest.json").write_text(
                json.dumps(action), encoding="utf-8"
            )
            (decision_root / "report_index.json").write_text(
                json.dumps({"latest_report_date": "20260820"}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "signal_date": "20260819",
                        "expected_buy_date": "20260820",
                        "expected_exit_date": "20260821",
                    }
                ]
            ).to_csv(prediction_root / "pred_latest.csv", index=False)

            next_dates = {
                "20260819": "20260820",
                "20260820": "20260821",
            }
            with mock.patch(
                "scripts.validate_decision_publication.is_a_share_trading_day",
                return_value=True,
            ), mock.patch(
                "scripts.validate_decision_publication.next_a_share_trading_day",
                side_effect=lambda value: next_dates[value],
            ):
                result = validate_publication(root)
            self.assertTrue(result["validated"])

            action["model"]["prediction_matches_report"] = False
            (decision_root / "action_plan_latest.json").write_text(
                json.dumps(action), encoding="utf-8"
            )
            with mock.patch(
                "scripts.validate_decision_publication.is_a_share_trading_day",
                return_value=True,
            ), mock.patch(
                "scripts.validate_decision_publication.next_a_share_trading_day",
                side_effect=lambda value: next_dates[value],
            ):
                with self.assertRaises(RuntimeError):
                    validate_publication(root)

    def test_history_validation_does_not_publish_current_action_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history_root = (
                root
                / "data"
                / "auction_v3"
                / "history"
                / "tplus1_open_0930_v1"
            )
            history_root.mkdir(parents=True)
            history = pd.DataFrame(
                [
                    {
                        "signal_date": "20260819",
                        "buy_date": "20260820",
                        "target_exit_date": "20260821",
                        "ts_code": "000001.SZ",
                    }
                ]
            )
            output_path = history_root / "training_20260819_20260819.csv"
            history.to_csv(output_path, index=False)
            manifest = {
                "schema_version": "decision_v11_history_manifest_v1",
                "strict_calendar": True,
                "target_window_open_sessions": 1,
                "total_compact_signal_dates": 1,
                "target_independent_dates": 1,
                "output_file": str(output_path.relative_to(root)),
                "output_sha256": _sha256_frame(history),
                "produced_signal_dates": 1,
            }
            (history_root / "manifest_latest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch(
                "scripts.validate_decision_history_backfill.is_a_share_trading_day",
                return_value=True,
            ), mock.patch(
                "scripts.validate_decision_history_backfill.next_a_share_trading_day",
                side_effect={
                    "20260819": "20260820",
                    "20260820": "20260821",
                }.get,
            ), mock.patch(
                "scripts.validate_decision_history_backfill.TARGET_HISTORY_DATES",
                1,
            ), mock.patch(
                "scripts.validate_decision_history_backfill.TARGET_INDEPENDENT_OOS_DATES",
                1,
            ):
                result = validate_history_backfill(root)
            self.assertTrue(result["validated"])
            self.assertFalse((root / "outputs" / "decision").exists())


if __name__ == "__main__":
    unittest.main()
