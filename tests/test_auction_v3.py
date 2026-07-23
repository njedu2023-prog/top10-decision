from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.auction_v3 import AuctionV3Config, AuctionV3Engine  # noqa: E402


class AuctionV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.codes = [f"6000{i:02d}.SH" for i in range(6)]
        dates = pd.bdate_range("2026-01-05", periods=40)
        self.dates = [d.strftime("%Y%m%d") for d in dates]
        self._write_market()
        self._write_candidates()
        self.config = AuctionV3Config(
            root=self.root,
            min_train_dates=8,
            min_train_rows=40,
            promotion_min_dates=12,
            promotion_min_oos_dates=5,
            backtest_block_dates=4,
            order_amount_cny=10_000,
            max_auction_participation=0.02,
            observation_validation_start_date=self.dates[0],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_market(self) -> None:
        previous = {code: 10.0 + idx for idx, code in enumerate(self.codes)}
        for day_index, trade_date in enumerate(self.dates):
            rows = []
            limits = []
            auctions = []
            for code_index, code in enumerate(self.codes):
                pre_close = previous[code]
                open_price = round(pre_close * (1.0 + 0.005 * code_index), 2)
                close_price = round(pre_close * 1.10, 2)
                high = close_price
                low = round(min(open_price, close_price) * 0.99, 2)
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close_price,
                        "pre_close": pre_close,
                        "vol": 1_000_000,
                        "amount": 50_000_000,
                        "pct_chg": 10.0,
                    }
                )
                limits.append({"ts_code": code, "trade_date": trade_date, "up_limit": round(pre_close * 1.10, 2), "down_limit": round(pre_close * 0.90, 2)})
                auctions.append({"ts_code": code, "trade_date": trade_date, "vol": 500_000, "price": open_price, "amount": 10_000_000})
                previous[code] = close_price
            path = self.root / "data" / "market" / "raw" / trade_date[:4] / trade_date
            path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path / "daily.csv", index=False)
            pd.DataFrame(limits).to_csv(path / "stk_limit.csv", index=False)
            pd.DataFrame(auctions).to_csv(path / "stk_auction.csv", index=False)

    def _write_candidates(self) -> None:
        archive = self.root / "data" / "pred" / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        for day_index, trade_date in enumerate(self.dates[:-1]):
            verify_date = self.dates[day_index + 1]
            rows = []
            for rank, code in enumerate(self.codes, start=1):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "verify_date": verify_date,
                        "rank": rank,
                        "ts_code": code,
                        "name": f"测试{rank}",
                        "晋阶": f"{min(rank, 6)}→{min(rank + 1, 7)}",
                        "Probability": 0.8 - rank * 0.07,
                        "StrengthScore": 90 - rank * 5,
                        "ThemeBoost": 0.9 - rank * 0.08,
                        "final_score_v2": 0.9 - rank * 0.07,
                        "intraday_quality_score": 0.9 - rank * 0.05,
                        "intraday_risk_score": rank * 0.04,
                        "intraday_hard_risk_flag": 0,
                        "auction_strength_score": 0.85 - rank * 0.04,
                        "stage_quality_weight": 1.0,
                        "stage_risk_weight": 0.01,
                        "limit_times": min(rank, 6),
                    }
                )
            pd.DataFrame(rows).to_csv(archive / f"pred_source_{trade_date}.csv", index=False)
        latest = archive / f"pred_source_{self.dates[-2]}.csv"
        latest_frame = pd.read_csv(latest)
        pred_root = self.root / "data" / "pred"
        latest_frame.to_csv(pred_root / "pred_source_latest.csv", index=False)

    def test_history_and_walkforward_are_dated(self) -> None:
        legacy_path = self.root / "data" / "market" / "raw" / self.dates[0][:4] / self.dates[0] / "daily.csv"
        legacy_daily = pd.read_csv(legacy_path).drop(columns=["pre_close"])
        legacy_daily.to_csv(legacy_path, index=False)
        engine = AuctionV3Engine(self.config)
        history = engine.build_history()
        self.assertIn(self.dates[0], set(history["signal_date"]))
        self.assertTrue((history["exit_reason"] == "take_profit_gap_conservative").all())
        self.assertTrue(np.allclose(history["gross_return"], self.config.take_profit_pct, atol=0.0015))
        self.assertGreaterEqual(history["signal_date"].nunique(), 30)
        oos, metrics = engine.run_backtest(history)
        self.assertFalse(oos.empty)
        self.assertTrue((oos["oos_train_end"] < oos["signal_date"]).all())
        self.assertGreater(metrics["oos_dates"], 0)

    def test_prediction_is_frozen_and_has_actionable_price(self) -> None:
        engine = AuctionV3Engine(self.config)
        history = engine.build_history()
        _, metrics = engine.run_backtest(history)
        bundle = engine.fit_models(history)
        self.assertIsNotNone(bundle)
        self.assertIsNotNone(bundle.fill_model)
        signal_date = self.dates[-2]
        candidates = engine.load_candidates(signal_date)
        first = engine.build_prediction(signal_date, candidates, bundle, metrics)
        self.assertIn("recommended_max_price", first.columns)
        self.assertIn("observation_max_price", first.columns)
        self.assertIn("observation_rank", first.columns)
        self.assertIn("predicted_continuation_limit_up_probability", first.columns)
        self.assertIn("market_order_allowed", first.columns)
        self.assertIn("feature_contract", first.columns)
        self.assertTrue(first["recommended_max_price"].notna().any())
        priced = first[first["recommended_max_price"].notna()]
        self.assertTrue((priced["recommended_max_price"] < priced["estimated_up_limit"]).all())
        observed = first[first["observation_selected"].eq(1)]
        self.assertLessEqual(len(observed), self.config.max_observation_candidates)
        self.assertTrue(observed["observation_max_price"].notna().all())
        self.assertTrue((observed["observation_max_price"] < observed["estimated_up_limit"]).all())
        self.assertTrue((first["market_order_allowed"] == 0).all())
        dated = self.config.prediction_root / f"pred_{signal_date}.csv"
        before = dated.read_bytes()
        second = engine.build_prediction(signal_date, candidates.iloc[::-1], bundle, metrics)
        self.assertEqual(before, dated.read_bytes())
        self.assertEqual(len(first), len(second))
        legacy = second.copy()
        legacy["model_version"] = "legacy_test"
        legacy.to_csv(dated, index=False)
        migrated = engine.build_prediction(signal_date, candidates, bundle, metrics)
        self.assertTrue((self.config.prediction_root / f"pred_{signal_date}_legacy_test.csv").exists())
        self.assertEqual({self.config.model_version}, set(migrated["model_version"]))
        ledger, _ = engine.settle_predictions()
        self.assertEqual(len(migrated), len(ledger))

    def test_opening_limit_up_that_breaks_later_is_not_a_confirmed_fill(self) -> None:
        trade_date = self.dates[5]
        code = self.codes[0]
        market_root = self.root / "data" / "market" / "raw" / trade_date[:4] / trade_date
        daily = pd.read_csv(market_root / "daily.csv")
        limits = pd.read_csv(market_root / "stk_limit.csv")
        auctions = pd.read_csv(market_root / "stk_auction.csv")
        up_limit = float(limits.loc[limits["ts_code"].eq(code), "up_limit"].iloc[0])
        daily.loc[daily["ts_code"].eq(code), ["open", "high", "low", "close"]] = [
            up_limit,
            up_limit,
            round(up_limit * 0.98, 2),
            round(up_limit * 0.99, 2),
        ]
        auctions.loc[auctions["ts_code"].eq(code), "price"] = up_limit
        daily.to_csv(market_root / "daily.csv", index=False)
        auctions.to_csv(market_root / "stk_auction.csv", index=False)

        fill, reason = AuctionV3Engine(self.config)._market_buyable(trade_date, code)
        self.assertEqual(fill, 0)
        self.assertEqual(reason, "opening_auction_limit_up_unconfirmed")

    def test_big_loss_probability_is_a_hard_veto(self) -> None:
        engine = AuctionV3Engine(self.config)
        history = engine.build_history()
        bundle = engine.fit_models(history)
        self.assertIsNotNone(bundle)
        bundle.loss_model = None
        bundle.loss_constant = self.config.max_big_loss_probability + 0.01
        score = engine._score_candidate_at_gaps(history.iloc[-1], bundle)
        self.assertIsNotNone(score)
        self.assertEqual(score["risk_gate_pass"], 0)
        self.assertEqual(score["model_reason"], "big_loss_probability_exceeds_cap")
        self.assertTrue(np.isnan(score["recommended_max_gap"]))

    def test_stage_is_derived_from_consecutive_limit_up_closes(self) -> None:
        engine = AuctionV3Engine(self.config)
        signal_date = self.dates[1]
        self.assertEqual(
            engine._consecutive_limit_up_count(signal_date, self.codes[0]),
            2,
        )
        current = engine._current_base(signal_date, engine.load_candidates(signal_date))
        row = current.loc[current["ts_code"].eq(self.codes[0])].iloc[0]
        self.assertEqual(row["stage"], "2→3")
        self.assertEqual(int(row["limit_times"]), 2)

    def test_authoritative_limit_list_restores_candidates_missing_from_old_ranking(self) -> None:
        signal_date = self.dates[10]
        market_root = self.root / "data" / "market" / "raw" / signal_date[:4] / signal_date
        daily = pd.read_csv(market_root / "daily.csv")
        limits = pd.read_csv(market_root / "stk_limit.csv")
        detail = daily[["ts_code", "trade_date", "close"]].merge(
            limits[["ts_code", "up_limit"]],
            on="ts_code",
            how="inner",
        )
        detail["name"] = [f"权威{i}" for i in range(len(detail))]
        detail["limit_type"] = "U"
        detail["open_times"] = 0
        detail["fd_amount"] = 1_000_000
        detail["first_time"] = 100000
        detail["last_time"] = 100000
        detail["seal_amount"] = 1_000_000
        detail.to_csv(market_root / "limit_list_d.csv", index=False)

        source_path = self.root / "data" / "pred" / "archive" / f"pred_source_{signal_date}.csv"
        source = pd.read_csv(source_path)
        missing_code = source.iloc[-1]["ts_code"]
        source.iloc[:-1].to_csv(source_path, index=False)

        candidates = AuctionV3Engine(self.config).load_candidates(signal_date)
        self.assertEqual(set(candidates["ts_code"]), set(self.codes))
        self.assertIn(missing_code, set(candidates["ts_code"]))

    def test_true_price_verification_and_reports(self) -> None:
        engine = AuctionV3Engine(self.config)
        signal_date = self.dates[1]
        history = engine.build_history()
        oos, metrics = engine.run_backtest(history)
        prediction = engine.build_prediction(signal_date, engine.load_candidates(signal_date), engine.fit_models(history), metrics)
        ledger, cumulative = engine.settle_predictions()
        self.assertFalse(ledger.empty)
        self.assertIn("actual_buy_price", ledger.columns)
        self.assertIn("actual_exit_price", ledger.columns)
        self.assertIn("truth_source", ledger.columns)
        self.assertGreaterEqual(cumulative.get("selected_predictions", 0), 1)
        observation, observation_metrics = engine.settle_observations()
        self.assertFalse(observation.empty)
        self.assertIn("market_daily_return", observation.columns)
        self.assertIn("continuation_limit_up_hit", observation.columns)
        self.assertIn("actual_net_return", observation.columns)
        self.assertIn("prediction_timing_status", observation.columns)
        self.assertLessEqual(
            int(observation.groupby("expected_buy_date").size().max()),
            self.config.max_observation_candidates,
        )
        self.assertGreater(observation_metrics.get("t_validated_rows", 0), 0)
        self.assertEqual(
            observation_metrics.get("performance_scope"),
            "premarket_valid_predictions_only",
        )
        self.assertGreater(
            observation_metrics.get("retrospective_truth_rows", 0),
            0,
        )
        from top10decision.auction_v3.reporting import write_reports

        paths = write_reports(
            self.config,
            prediction=prediction,
            ledger=ledger,
            backtest_trades=oos,
            backtest_metrics=metrics,
            cumulative_metrics=cumulative,
        )
        for path in paths.values():
            self.assertTrue(Path(path).exists())
            self.assertIn("Decision", Path(path).read_text(encoding="utf-8"))
        self.assertIn("竞价", Path(paths["current"]).read_text(encoding="utf-8"))

    def test_observation_timing_audit_uses_t_0925_deadline(self) -> None:
        valid = AuctionV3Engine._prediction_timing_status(
            "2026-07-22T13:12:08+00:00",
            "20260723",
        )
        late = AuctionV3Engine._prediction_timing_status(
            "2026-07-23T11:13:00+00:00",
            "20260723",
        )
        self.assertEqual(valid[0:2], ("PREMARKET_VALID", 1))
        self.assertEqual(late[0:2], ("RETROSPECTIVE_LATE_GENERATION", 0))
        self.assertEqual(valid[2], "2026-07-23T01:25:00+00:00")


if __name__ == "__main__":
    unittest.main()
