from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.decision.trade_selector import (  # noqa: E402
    FORBIDDEN_TRADE_SELECTOR_FEATURES,
    TRADE_SELECTOR_FEATURES,
    TradeProbabilityCalibrator,
    TradeSelectorBundle,
    TradeSelectorConfig,
    fit_trade_selector,
    prepare_observation_top10,
    score_trade_selector,
    walkforward_trade_selector,
)


def _constant_bundle(
    *,
    return_value: float = 0.03,
    fill_probability: float = 0.80,
    big_loss_probability: float = 0.10,
    min_score: float = 0.0,
) -> TradeSelectorBundle:
    return TradeSelectorBundle(
        return_model=None,
        return_constant=return_value,
        fill_model=None,
        fill_constant=fill_probability,
        big_loss_model=None,
        big_loss_constant=big_loss_probability,
        fill_calibrator=TradeProbabilityCalibrator(
            "constant",
            fill_probability,
        ),
        big_loss_calibrator=TradeProbabilityCalibrator(
            "constant",
            big_loss_probability,
        ),
        mean_return_margin=0.002,
        residual_q10=-0.05,
        policy={
            "ready": True,
            "max_positions": 2,
            "thresholds": {
                "min_trade_score": min_score,
                "min_mean_return_lcb": -0.10,
                "min_fill_probability": 0.0,
                "max_big_loss_probability": 0.50,
            },
        },
        train_rows=100,
        train_dates=50,
        return_training_rows=40,
        calibration_rows=30,
        policy_rows=30,
        return_selection={"selected": "constant", "passed": True},
        probability_selection={},
        artifact_sha256="a" * 64,
    )


class DecisionTradeSelectorTest(unittest.TestCase):
    def test_feature_contract_excludes_every_t_and_tplus1_truth(self) -> None:
        self.assertFalse(
            set(TRADE_SELECTOR_FEATURES)
            & FORBIDDEN_TRADE_SELECTOR_FEATURES
        )

    def test_observation_top10_is_exact_and_never_padded(self) -> None:
        rows = []
        for date, count in (("20260105", 12), ("20260106", 4)):
            for rank in range(1, count + 1):
                rows.append(
                    {
                        "signal_date": date,
                        "ts_code": f"600{rank:03d}.SH",
                        "stage": "2→3" if rank % 2 else "3→4",
                        "mechanism_limit_pct": 10.0,
                        "source_rank": rank,
                        "rank": rank,
                        "predicted_big_loss_probability": rank / 100.0,
                        "predicted_return_lcb": 0.02 - rank / 1000.0,
                        "predicted_exit_probability": 0.95,
                        "conservative_ev": 0.03 - rank / 1000.0,
                        "predicted_continuation_limit_up_probability": 0.60,
                        "d_close": 10.0,
                    }
                )
        top10 = prepare_observation_top10(pd.DataFrame(rows))
        counts = top10.groupby("signal_date").size().to_dict()
        self.assertEqual(counts, {"20260105": 10, "20260106": 4})
        self.assertEqual(int(top10["observation_rank"].max()), 10)
        self.assertEqual(
            top10.loc[
                top10["signal_date"].eq("20260106"),
                "observation_rank",
            ].tolist(),
            [1, 2, 3, 4],
        )

    def test_trade_selector_chooses_at_most_two_and_allows_zero(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "signal_date": "20260105",
                    "ts_code": f"60000{rank}.SH",
                    "observation_rank": rank,
                }
                for rank in range(1, 6)
            ]
        )
        selected = score_trade_selector(
            frame,
            _constant_bundle(),
            globally_promoted=True,
        )
        self.assertEqual(int(selected["trade_selected"].sum()), 2)
        self.assertEqual(
            selected.loc[
                selected["trade_selected"].eq(1),
                "trade_rank",
            ].tolist(),
            [1.0, 2.0],
        )
        no_trade = score_trade_selector(
            frame,
            _constant_bundle(min_score=0.50),
            globally_promoted=True,
        )
        self.assertEqual(int(no_trade["trade_selected"].sum()), 0)

    def test_conditional_return_fit_uses_only_market_buyable_rows(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=120)
        rows = []
        for day_index, date in enumerate(dates):
            signal_date = date.strftime("%Y%m%d")
            for rank in range(1, 5):
                market_fill = int(rank % 2 == 0)
                rows.append(
                    {
                        "signal_date": signal_date,
                        "ts_code": f"600{rank:03d}.SH",
                        "observation_rank": rank,
                        "source_rank": rank,
                        "stage": "2→3" if rank <= 2 else "3→4",
                        "market_fill": market_fill,
                        "net_return": (
                            0.015
                            + 0.002 * np.sin(day_index / 5.0)
                            if market_fill
                            else 0.50
                        ),
                        "big_loss_hit": 0,
                        "continuation_limit_up_hit": int(rank == 1),
                        "d_return": 0.10,
                        "path_strength_delta": rank / 100.0,
                        "predicted_net_return": 0.01 - rank / 1000.0,
                        "predicted_fill_probability": 0.50,
                        "predicted_big_loss_probability": 0.10,
                    }
                )
        history = pd.DataFrame(rows)
        config = TradeSelectorConfig(
            min_fit_dates=30,
            min_fit_buyable_rows=40,
            min_policy_dates=12,
            calibration_fraction=0.15,
            policy_fraction=0.15,
        )
        bundle = fit_trade_selector(
            history,
            cost_rate=0.0045,
            config=config,
        )
        self.assertIsNotNone(bundle)
        assert bundle is not None
        fit_dates = sorted(history["signal_date"].unique())[: bundle.train_dates]
        expected_buyable = history[
            history["signal_date"].isin(fit_dates)
            & history["market_fill"].eq(1)
        ]
        self.assertEqual(
            bundle.return_training_rows,
            len(expected_buyable),
        )
        self.assertEqual(
            bundle.return_selection["training_scope"],
            "market_fill_eq_1_only",
        )
        self.assertLess(bundle.return_training_rows, bundle.train_rows)

    def test_zero_trade_model_cannot_be_promoted(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=100)
        rows = []
        for day_index, date in enumerate(dates):
            for rank in range(1, 4):
                rows.append(
                    {
                        "signal_date": date.strftime("%Y%m%d"),
                        "ts_code": f"600{rank:03d}.SH",
                        "observation_rank": rank,
                        "source_rank": rank,
                        "stage": "2→3",
                        "market_fill": 1,
                        "net_return": -0.02 - rank / 1000.0,
                        "big_loss_hit": int(rank == 3),
                        "continuation_limit_up_hit": 0,
                        "d_return": 0.10,
                        "predicted_net_return": -0.01,
                        "predicted_fill_probability": 0.80,
                        "predicted_big_loss_probability": 0.40,
                    }
                )
        config = TradeSelectorConfig(
            warmup_dates=65,
            block_dates=15,
            min_fit_dates=30,
            min_fit_buyable_rows=40,
            min_policy_dates=10,
            min_policy_trades=6,
            min_policy_buyable_trades=4,
            promotion_min_oos_dates=20,
            promotion_min_trades=10,
            promotion_min_buyable_trades=10,
        )
        _, _, metrics = walkforward_trade_selector(
            pd.DataFrame(rows),
            cost_rate=0.0045,
            config=config,
        )
        self.assertFalse(metrics["promoted"])
        self.assertTrue(
            metrics["no_trade_guard"]["zero_trades_cannot_promote"]
        )
        self.assertIn(
            "production_policy_ready",
            metrics["promotion_failures"],
        )


if __name__ == "__main__":
    unittest.main()
