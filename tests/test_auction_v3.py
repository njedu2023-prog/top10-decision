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
                score_edge = (5 - code_index) * 0.003
                market_wave = 0.002 * np.sin(day_index / 3.0)
                open_price = round(pre_close * (1.0 + 0.005 * code_index), 2)
                close_price = round(open_price * (1.0 + score_edge + market_wave), 2)
                high = round(max(open_price, close_price) * 1.01, 2)
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
                        "pct_chg": (close_price / pre_close - 1.0) * 100.0,
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
        self.assertIn("feature_contract", first.columns)
        self.assertTrue(first["recommended_max_price"].notna().any())
        dated = self.config.prediction_root / f"pred_{signal_date}.csv"
        before = dated.read_bytes()
        second = engine.build_prediction(signal_date, candidates.iloc[::-1], bundle, metrics)
        self.assertEqual(before, dated.read_bytes())
        self.assertEqual(len(first), len(second))

    def test_true_price_verification_and_reports(self) -> None:
        engine = AuctionV3Engine(self.config)
        signal_date = self.dates[-4]
        history = engine.build_history()
        oos, metrics = engine.run_backtest(history)
        prediction = engine.build_prediction(signal_date, engine.load_candidates(signal_date), engine.fit_models(history), metrics)
        ledger, cumulative = engine.settle_predictions()
        self.assertFalse(ledger.empty)
        self.assertIn("actual_buy_price", ledger.columns)
        self.assertIn("actual_exit_price", ledger.columns)
        self.assertIn("truth_source", ledger.columns)
        self.assertGreaterEqual(cumulative.get("selected_predictions", 0), 1)
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
            self.assertIn("竞价", Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
