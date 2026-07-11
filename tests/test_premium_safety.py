from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load Premium submodules without executing the package's eager convenience
# imports. This keeps the safety suite focused and independent of report code.
top_pkg = types.ModuleType("top10decision")
top_pkg.__path__ = [str(SRC / "top10decision")]
premium_pkg = types.ModuleType("top10decision.premium")
premium_pkg.__path__ = [str(SRC / "top10decision" / "premium")]
report_stub = types.ModuleType("top10decision.premium.report_md")
report_stub.render_premium_report_md = lambda *args, **kwargs: ""
sys.modules.setdefault("top10decision", top_pkg)
sys.modules.setdefault("top10decision.premium", premium_pkg)
sys.modules.setdefault("top10decision.premium.report_md", report_stub)

from top10decision.premium.factor_builders import build_pack1_tushare_basic, build_pack2_limit_micro
from top10decision.premium.limitup_probability_engine import (
    _model_gate,
    fit_limitup_probability_engine,
    infer_feature_cols,
    time_split,
)
from top10decision.premium.predict import (
    _apply_adaptive_rank_score,
    _apply_professional_premium_scores,
    _historical_limitup_stats_from_df,
    _load_validated_platt_calibrator,
    _recent_limitup_model_health,
)
from top10decision.premium.premium_views import render_premium_report_html
from top10decision.premium.train import _build_ehx_feature_cols, _fit_validated_ehx


def _load_backfill_module():
    path = ROOT / "scripts" / "backfill_premium_limitup_truth.py"
    spec = importlib.util.spec_from_file_location("premium_truth_backfill", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_apple_style_module():
    path = ROOT / "scripts" / "apple_style_premium_reports.py"
    spec = importlib.util.spec_from_file_location("premium_apple_style", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_rebuild_module():
    path = ROOT / "scripts" / "rebuild_premium_reports.py"
    spec = importlib.util.spec_from_file_location("premium_report_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class PremiumSafetyTests(unittest.TestCase):
    def test_time_split_has_two_day_embargo(self):
        dates = [f"202601{x:02d}" for x in range(1, 21)]
        df = pd.DataFrame({"d_trade_date": np.repeat(dates, 2), "x": np.arange(40)})
        train, valid, train_end, valid_start = time_split(df, valid_ratio=0.30, embargo_days=2)
        self.assertLess(train_end, valid_start)
        ordered = sorted(df["d_trade_date"].unique())
        self.assertGreaterEqual(ordered.index(valid_start) - ordered.index(train_end), 3)
        self.assertFalse(set(train["d_trade_date"]).intersection(set(valid["d_trade_date"])))

    def test_future_fields_are_excluded(self):
        n = 100
        df = pd.DataFrame({
            "d_trade_date": ["20260101"] * n,
            "d_factor": np.arange(n),
            "in_p10": np.ones(n),
            "in_p50": np.ones(n),
            "t1_close_actual": np.arange(n),
        })
        cols = infer_feature_cols(df)
        self.assertIn("d_factor", cols)
        self.assertNotIn("in_p10", cols)
        self.assertNotIn("in_p50", cols)
        self.assertNotIn("t1_close_actual", cols)

        ehx_cols = _build_ehx_feature_cols(pd.DataFrame({
            "eret_pred_raw": [0.0],
            "rank_eret_plus": [1],
            "eret_plus_conf_score": [0.9],
            "r_p50": [0.1],
            "in_p10": [1],
            "in_p50": [1],
        }))
        self.assertEqual(ehx_cols, ["eret_pred_raw"])

    def test_primary_model_gate_requires_real_lift(self):
        rows = [
            {"target": "t_limitup_hit", "auc": 0.60, "brier_skill": 0.10, "daily_top10_lift": 0.0, "daily_spearman_ic": 0.08},
            {"target": "t1_accept_hit", "auc": 0.55, "brier_skill": 0.03},
            {"target": "t1_close_ret", "daily_spearman_ic": 0.05},
        ]
        ok, reason = _model_gate(pd.DataFrame(rows), validation_days=20, validation_samples=600)
        self.assertFalse(ok)
        self.assertIn("t_limit_top10_lift=0.0", reason)

        rows[0]["daily_top10_lift"] = 0.03
        ok, _ = _model_gate(pd.DataFrame(rows), validation_days=20, validation_samples=600)
        self.assertTrue(ok)

    def test_unvalidated_model_has_zero_production_weight(self):
        base = pd.DataFrame({
            "t_limitup_prob": [0.40],
            "t_touch_limitup_prob_model": [0.50],
            "t1_continue_up_rate": [0.45],
            "model_can_rank": [0],
            "t_up_prob_model": [0.99],
            "t_high_profit_prob_model": [0.99],
            "t_limitup_prob_model": [0.99],
            "t1_up_prob_model": [0.99],
            "t1_high_profit_prob_model": [0.99],
            "t1_accept_prob_model": [0.99],
            "t1_fail_prob_model": [0.01],
            "t1_big_drawdown_prob_model": [0.01],
            "t_limitup_strength": [60.0],
        })
        out, _ = _apply_professional_premium_scores(base)
        expected_rule_t_up = 0.40
        self.assertAlmostEqual(float(out.loc[0, "t_up_prob_model_blend"]), expected_rule_t_up, places=6)
        self.assertEqual(out.loc[0, "premium_rank_mode"], "professional_score_rule_guarded")

        high_model, _ = _apply_adaptive_rank_score(base, {})
        high_model, _ = _apply_professional_premium_scores(high_model)
        low_input = base.copy()
        for col in [c for c in low_input.columns if c.endswith("_model")]:
            low_input[col] = 0.01
        low_input["t1_close_ret_pred"] = -0.50
        low_input["t1_high_ret_pred"] = -0.40
        low_model, _ = _apply_adaptive_rank_score(low_input, {})
        low_model, _ = _apply_professional_premium_scores(low_model)
        self.assertAlmostEqual(
            float(high_model.loc[0, "premium_rank_score"]),
            float(low_model.loc[0, "premium_rank_score"]),
            places=6,
        )

    def test_truth_uses_official_limit_and_t_open_proxy(self):
        module = _load_backfill_module()
        verify = pd.DataFrame({
            "trade_date": ["20260105"],
            "buy_date": ["20260106"],
            "target_date": ["20260107"],
            "ts_code": ["600000.SH"],
            "close_T": [10.0],
            "t_max_buy_price": [10.5],
        })
        daily_t = pd.DataFrame({
            "ts_code": ["600000.SH"], "open": [10.8], "high": [11.0], "low": [10.7], "close": [11.0]
        })
        daily_t1 = pd.DataFrame({
            "ts_code": ["600000.SH"], "open": [11.1], "high": [12.2], "low": [10.6], "close": [11.88]
        })
        limits = pd.DataFrame({"ts_code": ["600000.SH"], "up_limit": [11.0]})
        out = module._attach_truth(
            verify, daily_t, daily_t1, limits, "20260105", "20260106", "20260107"
        )
        self.assertEqual(int(out.loc[0, "t_limitup_actual"]), 1)
        self.assertEqual(out.loc[0, "entry_price_proxy_type"], "t_open_daily_proxy")
        self.assertAlmostEqual(float(out.loc[0, "entry_price_proxy"]), 10.8, places=6)
        self.assertAlmostEqual(float(out.loc[0, "t1_close_ret"]), 11.88 / 10.8 - 1.0, places=6)
        self.assertEqual(int(out.loc[0, "t1_verify_ready"]), 1)

        invalid = module._attach_truth(
            verify, daily_t, daily_t1, limits, "20260105", "20260106", "20260106"
        )
        self.assertEqual(int(invalid.loc[0, "label_matured"]), 0)
        self.assertEqual(invalid.loc[0, "t1_verify_reason"], "invalid_D_T_T1_order")

    def test_actual_feature_files_are_consumed(self):
        class Cfg:
            def __init__(self, root: Path):
                self._root = root

            def repo_root(self):
                return self._root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market = root / "data" / "market"
            market.mkdir(parents=True)
            pd.DataFrame({
                "trade_date": ["20260105"], "ts_code": ["600000.SH"],
                "turnover_rate": [8.8], "float_mv": [123.0],
            }).to_csv(market / "features_base_20260105.csv", index=False)
            pd.DataFrame({
                "trade_date": ["20260105"], "ts_code": ["600000.SH"],
                "open_times": [2], "up_limit": [11.0],
            }).to_csv(market / "features_limit_20260105.csv", index=False)
            cfg = Cfg(root)
            pack1 = build_pack1_tushare_basic(cfg, "20260105")
            pack2 = build_pack2_limit_micro(cfg, "20260105")
            self.assertEqual(float(pack1.loc[0, "turnover_rate"]), 8.8)
            self.assertEqual(float(pack1.loc[0, "circ_mv"]), 123.0)
            self.assertEqual(int(pack2.loc[0, "open_times"]), 2)
            self.assertEqual(float(pack2.loc[0, "up_limit"]), 11.0)

    def test_training_paths_use_out_of_time_validation(self):
        rng = np.random.default_rng(42)
        dates = [f"202602{x:02d}" for x in range(1, 21)]
        rows = []
        for date in dates:
            for _ in range(15):
                x = float(rng.normal())
                limit_hit = int(x + rng.normal(scale=0.4) > 0)
                ret = 0.02 * x + float(rng.normal(scale=0.004))
                rows.append({
                    "d_trade_date": date,
                    "d_signal": x,
                    "label_matured": 1,
                    "t_up_hit": limit_hit,
                    "t_high_profit_hit": limit_hit,
                    "t_limitup_hit": limit_hit,
                    "t_touch_limitup": limit_hit,
                    "t1_up_hit": int(ret > 0),
                    "t1_high_profit_hit": int(ret > 0.01),
                    "t1_accept_hit": int(ret > -0.005),
                    "t1_fail_hit": int(ret < -0.005),
                    "t1_big_drawdown_hit": int(ret < -0.02),
                    "t_close_ret": ret,
                    "t_intraday_ret": ret + 0.01,
                    "t1_close_ret": ret,
                    "t1_high_ret": ret + 0.01,
                })
        bundle = fit_limitup_probability_engine(
            pd.DataFrame(rows), feature_cols=["d_signal"], valid_ratio=0.30, min_samples=20
        )
        self.assertGreaterEqual(bundle.validation_days, 5)
        self.assertIn("daily_top10_lift", bundle.metrics.columns)
        self.assertIn("brier_skill", bundle.metrics.columns)

        ehx_rows = []
        for date in dates:
            for _ in range(12):
                x = float(rng.normal())
                real = 0.03 * x + float(rng.normal(scale=0.003))
                ehx_rows.append({
                    "trade_date": date,
                    "eret_pred_raw": 0.0,
                    "real_premium_ret": real,
                    "delta_ret": real,
                    "signal": x,
                })
        promoted, _, metrics = _fit_validated_ehx(pd.DataFrame(ehx_rows), ["signal", "eret_pred_raw"])
        self.assertTrue(metrics["validation_pass"])
        self.assertIsNotNone(promoted)
        self.assertLess(metrics["plus_mae"], metrics["raw_mae"])

    def test_metrics_panel_is_collapsed_with_round_toggle(self):
        rows = pd.DataFrame({
            "rank": [1], "ts_code": ["600000.SH"], "name": ["示例"], "close_T": [10.0],
            "t_limitup_prob": [0.4], "t_limitup_strength": [55.0],
            "t1_continue_up_rate": [0.5], "limitup_continuation_score": [52.0],
            "premium_rank_score": [51.0],
        })
        html = render_premium_report_html(
            "20260105", "20260106", "20260107", rows, rows,
            True, "pending", "now", "test",
        )
        self.assertIn('<details class="metrics-details">', html)
        self.assertNotIn('<details class="metrics-details" open', html)
        self.assertIn('class="metrics-toggle"', html)

        styled = _load_apple_style_module().restyle_html(html)
        self.assertIn(".metrics-toggle::before{content:\"+\"", styled)
        self.assertIn('<details class="metrics-details">', styled)
        self.assertNotIn("验证口径：T日收盘涨停=命中", styled)

    def test_probability_calibrator_requires_holdout_brier_improvement(self):
        class Cfg:
            def __init__(self, root: Path):
                self.root = root

            def out_learning_dir(self):
                return self.root

        rng = np.random.default_rng(7)
        rows = []
        dates = [f"202603{x:02d}" for x in range(1, 21)]
        for date in dates:
            for raw_prob in np.linspace(0.35, 0.90, 60):
                true_prob = 1.0 / (1.0 + np.exp(-(np.log(raw_prob / (1.0 - raw_prob)) - 2.0)))
                rows.append({
                    "d_trade_date": date,
                    "t_limitup_prob": raw_prob,
                    "t_limitup_hit": int(rng.random() < true_prob),
                    "t_limitup_verify_ready": 1,
                    "model_can_rank": 0,
                })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            pd.DataFrame(rows).to_csv(path / "limitup_probability_training_samples.csv", index=False)
            calibrator, reason, metrics = _load_validated_platt_calibrator(Cfg(path), False)
            self.assertEqual(reason, "ok")
            self.assertIsNotNone(calibrator)
            self.assertLess(metrics["calibrated_brier"], metrics["raw_brier"])
            self.assertGreater(calibrator[0], 0)

    def test_recent_health_kills_a_degraded_active_model(self):
        class Cfg:
            def __init__(self, root: Path):
                self.root = root

            def out_root(self):
                return self.root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day in range(1, 9):
                actual = np.array([1] * 4 + [0] * 16)
                # The model ranks misses first; the rule ranks true hits first.
                model_prob = np.linspace(0.95, 0.10, 20)[::-1]
                rule_prob = np.linspace(0.95, 0.10, 20)
                pd.DataFrame({
                    "trade_date": [f"202604{day:02d}"] * 20,
                    "t_limitup_actual": actual,
                    "t_limitup_verify_ready": 1,
                    "t_limitup_prob_model": model_prob,
                    "t_limitup_prob_rule": rule_prob,
                    "model_can_rank": 1,
                }).to_csv(root / f"premium_verify_202604{day:02d}.csv", index=False)
            ok, reason, metrics = _recent_limitup_model_health(Cfg(root))
            self.assertFalse(ok)
            self.assertEqual(reason, "recent_health_fail")
            self.assertLess(metrics["top10_lift"], 0)

    def test_history_stats_coalesce_training_and_verify_aliases(self):
        history = pd.DataFrame({
            "_history_date": ["20260701", "20260702"],
            "rank": [1, 2],
            "label_matured": [1, np.nan],
            "t_limitup_verify_ready": [np.nan, 1],
            "t1_verify_ready": [np.nan, 1],
            "t_limitup_hit": [1, np.nan],
            "t_limitup_actual": [np.nan, 0],
            "t_up_hit": [1, np.nan],
            "t_up_actual": [np.nan, 1],
            "t_limitup_prob_rule": [0.8, np.nan],
            "t_limitup_prob": [np.nan, 0.2],
            "t1_continue_up_rate_rule": [0.6, np.nan],
            "t1_continue_up_rate": [np.nan, 0.4],
            "t1_close_ret": [0.03, -0.01],
        })
        stats = _historical_limitup_stats_from_df(history, "mixed_test")
        self.assertTrue(stats["ready"])
        self.assertEqual(stats["n_days"], 2)
        self.assertEqual(stats["top10_total"], 2)
        self.assertEqual(stats["top10_hits"], 1)
        self.assertEqual(stats["top1_up_total"], 1)
        self.assertEqual(stats["calibration_rows"], 2)

    def test_report_rebuild_merges_newer_verify_truth(self):
        module = _load_rebuild_module()

        class Cfg:
            def __init__(self, root: Path):
                self.root = root

            def out_root(self):
                return self.root

            def out_learning_dir(self):
                return self.root / "learning"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "learning").mkdir(parents=True)
            pd.DataFrame({
                "d_trade_date": ["20260701"], "ts_code": ["600000.SH"], "rank": [1],
                "label_matured": [1], "t_limitup_hit": [1], "t_up_hit": [1],
                "t_limitup_prob_rule": [0.8], "t1_close_ret": [0.03],
            }).to_csv(root / "learning" / "limitup_probability_training_samples.csv", index=False)
            pd.DataFrame({
                "trade_date": ["20260702"], "ts_code": ["000001.SZ"], "rank": [1],
                "t_limitup_verify_ready": [1], "t_limitup_actual": [0], "t_up_actual": [1],
                "t_limitup_prob": [0.2], "t1_close_ret": [-0.01],
            }).to_csv(root / "premium_verify_20260702.csv", index=False)
            stats = module._collect_historical_limitup_stats(Cfg(root))
            self.assertTrue(stats["ready"])
            self.assertEqual(stats["n_days"], 2)
            self.assertEqual(stats["top10_total"], 2)
            self.assertEqual(stats["top10_hits"], 1)


if __name__ == "__main__":
    unittest.main()
