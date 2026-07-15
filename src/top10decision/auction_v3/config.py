from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuctionV3Config:
    """Runtime and governance settings for the auction-overnight strategy."""

    root: Path = Path(".")
    max_candidates: int = 50
    max_positions: int = 3
    round_trip_cost_bps: float = 35.0
    slippage_bps_each_side: float = 5.0
    order_amount_cny: float = 100_000.0
    max_auction_participation: float = 0.01
    min_edge: float = 0.001
    min_fill_probability: float = 0.55
    min_profit_probability: float = 0.52
    max_big_loss_probability: float = 0.25
    big_loss_threshold: float = -0.03
    tail_risk_aversion: float = 0.50
    min_train_dates: int = 20
    min_train_rows: int = 300
    promotion_min_dates: int = 250
    promotion_min_oos_dates: int = 80
    embargo_dates: int = 2
    backtest_block_dates: int = 10
    gap_grid_min: float = -0.05
    gap_grid_max: float = 0.10
    gap_grid_step: float = 0.005
    lower_confidence_quantile: float = 0.10
    prediction_interval_upper_quantile: float = 0.90
    model_version: str = "auction_v3_walkforward_3"

    @property
    def output_root(self) -> Path:
        return self.root / "outputs" / "auction_v3"

    @property
    def prediction_root(self) -> Path:
        return self.output_root / "predictions"

    @property
    def truth_root(self) -> Path:
        return self.output_root / "truth"

    @property
    def verification_root(self) -> Path:
        return self.output_root / "verification"

    @property
    def metrics_root(self) -> Path:
        return self.output_root / "metrics"

    @property
    def model_root(self) -> Path:
        return self.output_root / "models"

    @property
    def report_root(self) -> Path:
        return self.root / "docs" / "reports"

    @property
    def broker_fills_path(self) -> Path:
        return self.root / "data" / "auction_v3" / "broker_fills.csv"

    @property
    def cost_rate(self) -> float:
        return (self.round_trip_cost_bps + 2.0 * self.slippage_bps_each_side) / 10_000.0

    def ensure_directories(self) -> None:
        for path in (
            self.prediction_root,
            self.truth_root,
            self.verification_root,
            self.metrics_root,
            self.model_root,
            self.report_root,
            self.broker_fills_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
