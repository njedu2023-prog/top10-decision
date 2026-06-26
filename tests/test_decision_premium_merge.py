#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd

from top10decision.premium.predict import _decision_merge_trace, _load_decision_merge


class DummyPremiumConfig:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.decision_glob = "outputs/decision/*.csv"

    def repo_root(self) -> Path:
        return self._root


def test_premium_loads_decision_labels_without_report_schema_change(tmp_path) -> None:
    root = tmp_path
    out_dir = root / "outputs" / "decision"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_date": "20260624",
                "ts_code": "600001.SH",
                "name": "A",
                "sector": "化工",
                "rank": 1,
                "weight": 0.12,
                "can_buy": 1,
                "p_fill_pred": 0.73,
                "reason": "pass",
            },
            {
                "trade_date": "20260623",
                "ts_code": "600002.SH",
                "name": "old",
                "rank": 2,
                "p_fill_pred": 0.91,
            },
        ]
    ).to_csv(out_dir / "decision_candidates_20260624.csv", index=False)

    cfg = DummyPremiumConfig(root)
    dec = _load_decision_merge(cfg, "20260624")

    assert list(dec["ts_code"]) == ["600001.SH"]
    assert float(dec.loc[0, "dec_p_fill"]) == 0.73
    assert float(dec.loc[0, "dec_can_buy"]) == 1.0
    assert float(dec.loc[0, "dec_weight"]) == 0.12
    assert str(dec.loc[0, "dec_reason"]) == "pass"

    premium_input = pd.DataFrame(
        [
            {"trade_date": "20260624", "ts_code": "600001.SH"},
            {"trade_date": "20260624", "ts_code": "600099.SH"},
        ]
    )
    merged = premium_input.merge(dec, on=["trade_date", "ts_code"], how="left")
    trace = _decision_merge_trace(dec, merged)

    assert trace["decision_merge_reason"] == "ok"
    assert trace["decision_merge_rows"] == 1
    assert trace["decision_merge_coverage"] == 0.5
    assert trace["decision_merge_dec_p_fill_coverage"] == 0.5
