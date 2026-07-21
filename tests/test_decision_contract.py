from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_v2 import _apply_ev_upgrade_v1  # noqa: E402
from scripts.build_eret_truth import infer_eret_label  # noqa: E402
from scripts.build_fill_truth import infer_fill_label  # noqa: E402
from scripts.resolve_sample_maturity import (  # noqa: E402
    resolve_sample_maturity_rows,
    resolve_trade_calendar,
)
from top10decision.data.tushare_minute import opening_auction_price_from_snapshot  # noqa: E402
from top10decision.decision.action_plan import build_action_plan  # noqa: E402
from top10decision.decision.eligibility import filter_standard_limit_universe  # noqa: E402
from top10decision.decision.exit_policy import simulate_tplus1_exit  # noqa: E402
from top10decision.writers import io_contract  # noqa: E402


class DecisionCalendarContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_calendar_path = io_contract.TRADE_CALENDAR_PATH
        io_contract.TRADE_CALENDAR_PATH = Path(self.temp.name) / "missing_calendar.csv"
        io_contract._load_exchange_calendar.cache_clear()

    def tearDown(self) -> None:
        io_contract.TRADE_CALENDAR_PATH = self.original_calendar_path
        io_contract._load_exchange_calendar.cache_clear()
        self.temp.cleanup()

    def test_official_2026_holidays_are_skipped(self) -> None:
        self.assertEqual(io_contract.choose_exec_date("20260430", "20260501"), "20260506")
        self.assertEqual(io_contract.choose_exit_date("20260506"), "20260507")
        self.assertEqual(io_contract.choose_exec_date("20260618", "20260619"), "20260622")
        self.assertEqual(io_contract.choose_exit_date("20260622"), "20260623")

    def test_synced_calendar_is_authoritative_and_fails_closed(self) -> None:
        calendar_path = Path(self.temp.name) / "trade_cal_sse.csv"
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20260720", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20260721", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20260722", "is_open": 1},
            ]
        ).to_csv(calendar_path, index=False)
        io_contract.TRADE_CALENDAR_PATH = calendar_path
        io_contract._load_exchange_calendar.cache_clear()

        self.assertEqual(io_contract.choose_exec_date("20260720", "20260721"), "20260722")
        with self.assertRaises(RuntimeError):
            io_contract.is_a_share_trading_day("20260723")

    def test_maturity_uses_only_explicit_exchange_calendar(self) -> None:
        calendar_path = Path(self.temp.name) / "strict_calendar.csv"
        days = [
            ("20260430", 1),
            ("20260501", 0),
            ("20260502", 0),
            ("20260503", 0),
            ("20260504", 0),
            ("20260505", 0),
            ("20260506", 1),
            ("20260507", 1),
        ]
        pd.DataFrame(
            [{"exchange": "SSE", "cal_date": day, "is_open": flag} for day, flag in days]
        ).to_csv(calendar_path, index=False)

        calendar = resolve_trade_calendar(
            raw_trade_dates=["20260430", "20260502", "20260506", "20260507"],
            candidate_trade_dates=["20260430"],
            current_run_date="20260507",
            trade_calendar_file=calendar_path,
        )
        rows = resolve_sample_maturity_rows(
            current_run_date="20260507",
            all_trade_dates_from_raw=["20260430", "20260502", "20260506", "20260507"],
            candidate_trade_dates=["20260430"],
            trade_calendar_dates=calendar,
        )
        self.assertEqual(rows[0].exec_date, "20260506")
        self.assertEqual(rows[0].target_date, "20260507")
        self.assertEqual(rows[0].FULLY_READY, 1)
        self.assertNotIn("20260502", calendar)

    def test_maturity_calendar_gap_fails_closed(self) -> None:
        calendar_path = Path(self.temp.name) / "calendar_with_gap.csv"
        pd.DataFrame(
            [
                {"cal_date": "20260430", "is_open": 1},
                {"cal_date": "20260501", "is_open": 0},
                {"cal_date": "20260502", "is_open": 0},
                {"cal_date": "20260503", "is_open": 0},
                # 20260504 intentionally absent.
                {"cal_date": "20260505", "is_open": 0},
                {"cal_date": "20260506", "is_open": 1},
                {"cal_date": "20260507", "is_open": 1},
            ]
        ).to_csv(calendar_path, index=False)
        with self.assertRaisesRegex(RuntimeError, "存在缺口"):
            resolve_trade_calendar(
                raw_trade_dates=["20260430", "20260506", "20260507"],
                candidate_trade_dates=["20260430"],
                current_run_date="20260507",
                trade_calendar_file=calendar_path,
            )


class DecisionExecutionTruthTests(unittest.TestCase):
    def test_later_intraday_break_does_not_count_as_auction_fill(self) -> None:
        label = infer_fill_label(
            pd.Series(
                {
                    "open_t1": 11.0,
                    "up_limit_t1": 11.0,
                    "open_times_t1": 3,
                    "break_open_times_t1": 3,
                }
            )
        )
        self.assertEqual(label[0], 0)
        self.assertEqual(label[1], "strong_opening_auction_limit_up_unconfirmed")

    def test_eret_gap_up_uses_predeclared_take_profit_not_hindsight_close(self) -> None:
        label = infer_eret_label(
            pd.Series(
                {
                    "y_fill": 1,
                    "entry_price_proxy_t1": 10.0,
                    "open_t2": 11.0,
                    "high_t2": 20.0,
                    "low_t2": 10.5,
                    "close_t2": 20.0,
                }
            )
        )
        self.assertAlmostEqual(float(label[0]), 0.03)
        self.assertEqual(label[2], 1)
        self.assertEqual(label[3], 10.3)
        self.assertEqual(label[6], "take_profit_gap_conservative")

    def test_daily_bar_both_hits_are_counted_stop_first(self) -> None:
        result = simulate_tplus1_exit(
            entry_price=10.0,
            open_price=10.0,
            high_price=10.5,
            low_price=9.7,
            close_price=10.2,
        )
        self.assertTrue(result.executable)
        self.assertEqual(result.exit_price, 9.75)
        self.assertEqual(result.reason, "both_hit_stop_first_conservative")

    def test_minute_first_touch_can_confirm_take_profit_before_stop(self) -> None:
        minute = pd.DataFrame(
            [
                {"time": "2026-07-22 09:30:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1},
                {"time": "2026-07-22 09:31:00", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.3},
                {"time": "2026-07-22 09:32:00", "open": 10.3, "high": 10.3, "low": 9.7, "close": 9.8},
            ]
        )
        result = simulate_tplus1_exit(
            entry_price=10.0,
            open_price=10.0,
            high_price=10.4,
            low_price=9.7,
            close_price=9.8,
            minute_frame=minute,
        )
        self.assertEqual(result.exit_price, 10.3)
        self.assertEqual(result.reason, "take_profit_first_touch")

    def test_no_threshold_hit_uses_time_exit_proxy(self) -> None:
        result = simulate_tplus1_exit(
            entry_price=10.0,
            open_price=10.0,
            high_price=10.2,
            low_price=9.9,
            close_price=10.1,
        )
        self.assertEqual(result.exit_price, 10.1)
        self.assertEqual(result.reason, "time_exit_close_proxy")

    def test_one_price_limit_down_is_not_an_executable_tplus1_exit(self) -> None:
        label = infer_eret_label(
            pd.Series(
                {
                    "y_fill": 1,
                    "entry_price_proxy_t1": 10.0,
                    "open_t2": 9.0,
                    "high_t2": 9.0,
                    "low_t2": 9.0,
                    "close_t2": 9.0,
                    "down_limit_t2": 9.0,
                }
            )
        )
        self.assertIsNone(label[0])
        self.assertEqual(label[1], "blocked_one_price_limit_down")
        self.assertEqual(label[2], 0)
        self.assertEqual(label[5], 0)

    def test_minute_snapshot_uses_first_0930_open(self) -> None:
        frame = pd.DataFrame(
            [
                {"time": "2026-07-21 09:30:00", "open": 10.2, "close": 10.3},
                {"time": "2026-07-21 09:31:00", "open": 10.4, "close": 10.5},
            ]
        )
        self.assertEqual(opening_auction_price_from_snapshot(frame), 10.2)


class DecisionUniverseAndEvTests(unittest.TestCase):
    def test_only_price_limit_mechanisms_at_or_below_ten_percent_survive(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_code": "600001.SH", "name": "主板A"},
                {"ts_code": "000002.SZ", "name": "主板B"},
                {"ts_code": "002003.SZ", "name": "ST样本"},
                {"ts_code": "300001.SZ", "name": "创业板"},
                {"ts_code": "688001.SH", "name": "科创板"},
                {"ts_code": "920001.BJ", "name": "北交所"},
                {"ts_code": "600010.SH", "name": "新股", "limit_type": "no_limit"},
                {"ts_code": "600011.SH", "name": "价格取整", "pre_close": 3.79, "up_limit": 4.17},
                {"ts_code": "600012.SH", "name": "上市初期", "trade_date": "20260720", "list_date": "20260715"},
            ]
        )
        eligible, audit = filter_standard_limit_universe(frame)
        self.assertEqual(set(eligible["ts_code"]), {"600001.SH", "000002.SZ", "002003.SZ", "600011.SH"})
        self.assertEqual(audit["rejected_rows"], 5)
        self.assertTrue((eligible["decision_limit_pct"] <= 10.0).all())

    def test_ev_cost_and_risk_are_subtracted_once(self) -> None:
        frame = pd.DataFrame(
            [{"p_fill_pred": 0.8, "e_ret_pred": 0.05, "cost_est": 0.004, "risk_penalty": 0.006}]
        )
        result = _apply_ev_upgrade_v1(frame).iloc[0]
        self.assertAlmostEqual(float(result["ev_pred"]), 0.03, places=12)
        self.assertEqual(float(result["ev_penalty_total_extra"]), 0.0)


class DecisionActionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "outputs" / "decision").mkdir(parents=True)
        (self.root / "outputs" / "auction_v3" / "predictions").mkdir(parents=True)
        (self.root / "outputs" / "auction_v3" / "metrics").mkdir(parents=True)
        (self.root / "outputs" / "auction_v3" / "models").mkdir(parents=True)
        (self.root / "data" / "decision").mkdir(parents=True)
        (self.root / "outputs" / "decision" / "decision_report_20260721.md").write_text("# report\n", encoding="utf-8")
        (self.root / "outputs" / "decision" / "eval_20260721.json").write_text(
            json.dumps(
                {
                    "signal_date": "20260720",
                    "exec_date": "20260721",
                    "exit_date": "20260722",
                    "risk_budget": 0.6,
                    "stop_trading": False,
                    "paths": {"candidates": "data/decision/decision_candidates_20260720.csv"},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"ts_code": "600001.SH", "name": "主板", "industry": "银行", "p_fill_pred": 0.8, "e_ret_pred": 0.02, "ev_pred": 0.01},
                {"ts_code": "300001.SZ", "name": "创业板", "industry": "软件", "p_fill_pred": 0.9, "e_ret_pred": 0.03, "ev_pred": 0.02},
            ]
        ).to_csv(self.root / "data" / "decision" / "decision_candidates_20260720.csv", index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_model_artifacts(self, *, promoted: bool, version: str = "auction_v5_manual_guidance_2") -> None:
        pd.DataFrame(
            [
                {
                    "signal_date": "20260720",
                    "expected_buy_date": "20260721",
                    "expected_exit_date": "20260722",
                    "ts_code": "600001.SH",
                    "name": "主板",
                    "industry": "银行",
                    "selected": 1,
                    "action": "BUY",
                    "recommended_max_price": 10.5,
                    "mechanism_limit_pct": 10.036,
                    "predicted_fill_probability": 0.8,
                    "predicted_exit_probability": 0.95,
                    "predicted_big_loss_probability": 0.05,
                    "predicted_continuation_limit_up_probability": 0.60,
                    "predicted_net_return": 0.02,
                    "predicted_return_lcb": 0.005,
                    "conservative_ev": 0.004,
                    "risk_gate_pass": 1,
                    "stage": "2→3",
                    "stage_focus": 1,
                    "order_type": "LIMIT_ONLY_MANUAL",
                    "market_order_allowed": 0,
                    "model_ready": 1,
                    "model_promoted": int(promoted),
                    "model_version": version,
                },
                {
                    "signal_date": "20260720",
                    "expected_buy_date": "20260721",
                    "expected_exit_date": "20260722",
                    "ts_code": "300001.SZ",
                    "name": "创业板",
                    "selected": 1,
                    "action": "BUY",
                    "model_ready": 1,
                    "model_promoted": int(promoted),
                    "model_version": version,
                },
            ]
        ).to_csv(self.root / "outputs" / "auction_v3" / "predictions" / "pred_latest.csv", index=False)
        (self.root / "outputs" / "auction_v3" / "metrics" / "backtest_latest.json").write_text(
            json.dumps({"model_version": version, "promoted": promoted, "promotion_failures": []}), encoding="utf-8"
        )
        (self.root / "outputs" / "auction_v3" / "models" / "model_meta_latest.json").write_text(
            json.dumps({"model_version": version, "ready": True, "promoted": promoted}), encoding="utf-8"
        )

    def test_unpromoted_model_can_never_emit_formal_buy(self) -> None:
        self._write_model_artifacts(promoted=False)
        plan = build_action_plan(self.root)
        self.assertEqual(plan["status_code"], "NO_TRADE_MODEL_NOT_PROMOTED")
        self.assertEqual(plan["formal_buy_count"], 0)
        self.assertFalse(any(row["action"] == "BUY" for row in plan["candidates"]))
        self.assertEqual(plan["stage_watch_count"], 1)
        self.assertEqual(plan["stage_watchlist"][0]["watch_label"], "仅观察")

    def test_promoted_model_still_rejects_above_ten_percent_board(self) -> None:
        self._write_model_artifacts(promoted=True)
        plan = build_action_plan(self.root)
        actions = {row["ts_code"]: row["action"] for row in plan["candidates"]}
        self.assertEqual(actions["600001.SH"], "BUY")
        self.assertEqual(actions["300001.SZ"], "REJECT")
        self.assertEqual(plan["formal_buy_count"], 1)
        main_board = next(row for row in plan["candidates"] if row["ts_code"] == "600001.SH")
        self.assertEqual(main_board["mechanism_limit_pct"], 10.0)
        self.assertEqual(main_board["stage_transition"], "2→3")
        self.assertEqual(main_board["market_order_allowed"], 0)
        self.assertFalse(plan["broker_connected"])
        self.assertEqual(plan["stage_watchlist"][0]["watch_label"], "正式买入")

    def test_artifact_version_mismatch_fails_closed(self) -> None:
        self._write_model_artifacts(promoted=True)
        meta_path = self.root / "outputs" / "auction_v3" / "models" / "model_meta_latest.json"
        meta_path.write_text(json.dumps({"model_version": "stale", "ready": True, "promoted": True}), encoding="utf-8")
        plan = build_action_plan(self.root)
        self.assertFalse(plan["model"]["promoted"])
        self.assertFalse(plan["model"]["artifact_versions_match"])
        self.assertEqual(plan["formal_buy_count"], 0)


if __name__ == "__main__":
    unittest.main()
