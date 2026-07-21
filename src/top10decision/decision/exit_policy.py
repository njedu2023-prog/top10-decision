from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .contracts import (
    EXIT_LATEST_TIME,
    EXIT_POLICY_VERSION,
    EXIT_STOP_LOSS_PCT,
    EXIT_TAKE_PROFIT_PCT,
)


@dataclass(frozen=True)
class TimedExitResult:
    exit_price: float | None
    executable: bool
    reason: str
    source: str
    take_profit_price: float | None
    stop_loss_price: float | None
    latest_exit_time: str = EXIT_LATEST_TIME
    policy_version: str = EXIT_POLICY_VERSION


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _price(value: float) -> float:
    return round(float(value) + 1e-12, 2)


def adjusted_entry_reference(
    entry_price: Any,
    buy_close: Any = None,
    target_pre_close: Any = None,
) -> float:
    entry = _finite(entry_price)
    buy_close_value = _finite(buy_close)
    target_pre_close_value = _finite(target_pre_close)
    if entry <= 0:
        return float("nan")
    if buy_close_value > 0 and target_pre_close_value > 0:
        return entry * target_pre_close_value / buy_close_value
    return entry


def corporate_action_safe_return(
    entry_price: Any,
    exit_price: Any,
    buy_close: Any = None,
    target_pre_close: Any = None,
) -> float:
    entry = _finite(entry_price)
    exit_value = _finite(exit_price)
    buy_close_value = _finite(buy_close)
    target_pre_close_value = _finite(target_pre_close)
    if entry <= 0 or exit_value <= 0:
        return float("nan")
    if buy_close_value > 0 and target_pre_close_value > 0:
        return (buy_close_value / entry) * (exit_value / target_pre_close_value) - 1.0
    return exit_value / entry - 1.0


def _is_one_price_limit_down(
    *,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    down_limit: Any,
) -> bool:
    limit_value = _finite(down_limit)
    prices = [_finite(value) for value in (open_price, high_price, low_price, close_price)]
    return bool(
        limit_value > 0
        and all(value > 0 and abs(value - limit_value) <= 0.011 for value in prices)
    )


def _minute_window(frame: pd.DataFrame | None, latest_exit_time: str) -> pd.DataFrame:
    if frame is None or frame.empty or "time" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    times = out["time"].astype(str).str.extract(r"(\d{2}:\d{2})(?::\d{2})?", expand=False)
    out["_hhmm"] = times
    out = out[out["_hhmm"].between("09:30", latest_exit_time, inclusive="both")].copy()
    for column in ("open", "high", "low", "close"):
        if column not in out.columns:
            return pd.DataFrame()
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"]).sort_values("_hhmm")


def simulate_tplus1_exit(
    *,
    entry_price: Any,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    buy_close: Any = None,
    target_pre_close: Any = None,
    down_limit: Any = None,
    minute_frame: pd.DataFrame | None = None,
    take_profit_pct: float = EXIT_TAKE_PROFIT_PCT,
    stop_loss_pct: float = EXIT_STOP_LOSS_PCT,
    latest_exit_time: str = EXIT_LATEST_TIME,
) -> TimedExitResult:
    reference = adjusted_entry_reference(entry_price, buy_close, target_pre_close)
    if not math.isfinite(reference) or reference <= 0:
        return TimedExitResult(None, False, "missing_entry_reference", "", None, None, latest_exit_time)

    take_profit = _price(reference * (1.0 + float(take_profit_pct)))
    stop_loss = _price(reference * (1.0 + float(stop_loss_pct)))
    daily_open = _finite(open_price)
    daily_high = _finite(high_price)
    daily_low = _finite(low_price)
    daily_close = _finite(close_price)

    if _is_one_price_limit_down(
        open_price=daily_open,
        high_price=daily_high,
        low_price=daily_low,
        close_price=daily_close,
        down_limit=down_limit,
    ):
        return TimedExitResult(
            None,
            False,
            "blocked_one_price_limit_down",
            "daily_limit_truth",
            take_profit,
            stop_loss,
            latest_exit_time,
        )

    minute = _minute_window(minute_frame, latest_exit_time)
    if not minute.empty:
        first_open = float(minute.iloc[0]["open"])
        if first_open <= stop_loss:
            return TimedExitResult(first_open, True, "stop_gap_at_open", "minute_1m_first_touch", take_profit, stop_loss, latest_exit_time)
        if first_open >= take_profit:
            return TimedExitResult(take_profit, True, "take_profit_gap_conservative", "minute_1m_first_touch", take_profit, stop_loss, latest_exit_time)
        for _, bar in minute.iterrows():
            low_value = float(bar["low"])
            high_value = float(bar["high"])
            if low_value <= stop_loss:
                reason = "both_hit_stop_first_conservative" if high_value >= take_profit else "stop_loss_first_touch"
                return TimedExitResult(stop_loss, True, reason, "minute_1m_first_touch", take_profit, stop_loss, latest_exit_time)
            if high_value >= take_profit:
                return TimedExitResult(take_profit, True, "take_profit_first_touch", "minute_1m_first_touch", take_profit, stop_loss, latest_exit_time)
        final_close = float(minute.iloc[-1]["close"])
        return TimedExitResult(final_close, True, "time_exit", "minute_1m_time_exit", take_profit, stop_loss, latest_exit_time)

    if not all(math.isfinite(value) and value > 0 for value in (daily_open, daily_high, daily_low, daily_close)):
        return TimedExitResult(None, False, "missing_tplus1_ohlc", "", take_profit, stop_loss, latest_exit_time)
    if daily_open <= stop_loss:
        return TimedExitResult(daily_open, True, "stop_gap_at_open", "daily_ohlc_conservative_proxy", take_profit, stop_loss, latest_exit_time)
    if daily_open >= take_profit:
        return TimedExitResult(take_profit, True, "take_profit_gap_conservative", "daily_ohlc_conservative_proxy", take_profit, stop_loss, latest_exit_time)
    if daily_low <= stop_loss:
        reason = "both_hit_stop_first_conservative" if daily_high >= take_profit else "stop_loss_daily_proxy"
        return TimedExitResult(stop_loss, True, reason, "daily_ohlc_conservative_proxy", take_profit, stop_loss, latest_exit_time)
    if daily_high >= take_profit:
        return TimedExitResult(take_profit, True, "take_profit_daily_proxy", "daily_ohlc_conservative_proxy", take_profit, stop_loss, latest_exit_time)
    return TimedExitResult(daily_close, True, "time_exit_close_proxy", "daily_ohlc_conservative_proxy", take_profit, stop_loss, latest_exit_time)


__all__ = [
    "TimedExitResult",
    "adjusted_entry_reference",
    "corporate_action_safe_return",
    "simulate_tplus1_exit",
]
