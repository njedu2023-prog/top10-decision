"""Top10 Decision - shared config constants.

These constants are intended to be stable and reflect MD contracts.
They must not rely on any third-party execution results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillTruthConfig:
    buy_window_start: str = "09:30:00"
    buy_window_end: str = "10:30:00"

    # expected output path (template). It must not change existing downstream paths.
    output_path_template: str = "data/market/fill_truth_{trade_date}.csv"

    # label quality categories (keep stable for audit)
    label_quality_default: str = "strong"
    label_quality_missing_truth: str = "weak_missing_truth"


fill_truth_config = FillTruthConfig()

@dataclass(frozen=True)
class EntryPriceProxyConfig:
    mode_default: str = "t1_open"
    mode_fallback: str = "t1_open_degrade_missing"

entry_price_proxy_config = EntryPriceProxyConfig()


@dataclass(frozen=True)
class ERetTruthConfig:
    entry_price_proxy_mode: str = "t1_open"
    sell_price_col: str = "close_t2"
    output_path_template: str = "data/market/eret_truth_{trade_date}.csv"

    label_quality_default: str = "strong"
    label_quality_missing_truth: str = "weak_missing_truth"


eret_truth_config = ERetTruthConfig()
