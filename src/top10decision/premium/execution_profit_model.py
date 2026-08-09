#!/usr/bin/env python3
"""Guarded T-auction to T+1 11:00 execution-profit model.

The model is intentionally isolated from Decision.  It reconstructs labels
from the immutable Premium execution-truth ledger, validates them with purged
expanding walk-forward folds, and permits a buy only when both the model-level
and candidate-level conservative net-return gates pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "premium_exec_profit_v2_actual_auction_1100_purged"
TRUTH_SCHEMA_VERSION = "premium_execution_truth_v1"
MODEL_RELEASE_D_DATE = "20260807"
EMBARGO_DAYS = 2
VALIDATION_MODE = "purged_expanding_walk_forward_2d_embargo"
DATE_RE = re.compile(r"(20\d{6})")

PREMIUM_SCORE_FEATURES: Tuple[str, ...] = (
    "t_limitup_prob",
    "t_limitup_strength",
    "t1_continue_up_rate",
    "limitup_continuation_score",
    "premium_adaptive_score",
    "premium_final_score",
    "premium_rank_score",
    "t_up_attack_score",
    "t1_accept_score",
    "execution_safety_score",
    "risk_penalty_score",
    "market_score",
    "eret_plus_score",
    "t_close_ret_pred",
    "t1_close_ret_pred",
    "intraday_quality_score",
    "intraday_risk_score",
    "auction_strength_score",
    "t_high_profit_prob_model",
    "t_touch_limitup_prob_model",
    "t1_high_profit_prob_model",
    "t1_fail_prob_model",
    "t1_big_drawdown_prob_model",
    "t_intraday_ret_pred",
    "t1_high_ret_pred",
    "intraday_attack_edge",
    "intraday_execution_edge",
    "intraday_risk_penalty",
    "mkt_emotion_score",
    "turnover_rate",
    "volume_ratio",
    "circ_mv",
    "volatility_5d",
    "volatility_10d",
    "max_drawdown_20d",
    "tail_risk_score",
    "hot_boards_score",
    "board_crowding_rank",
    "open_times",
    "fd_amount",
    "seal_amount",
)

CROSS_SECTIONAL_BASES: Tuple[str, ...] = (
    "t_limitup_prob",
    "t_limitup_strength",
    "t1_continue_up_rate",
    "premium_adaptive_score",
    "premium_final_score",
    "premium_rank_score",
    "t_up_attack_score",
    "t1_accept_score",
    "execution_safety_score",
    "risk_penalty_score",
    "eret_plus_score",
    "t_close_ret_pred",
    "t1_close_ret_pred",
    "t1_high_ret_pred",
    "intraday_quality_score",
    "intraday_risk_score",
    "auction_strength_score",
    "turnover_rate",
    "volume_ratio",
    "open_times",
)
CROSS_SECTIONAL_FEATURES: Tuple[str, ...] = tuple(
    f"xs_{feature}" for feature in CROSS_SECTIONAL_BASES
)

# Production v2 deliberately uses frozen D-day candidate facts. Reconstructed
# market panels remain optional enrichment, never a prerequisite for scoring.
FEATURES: Tuple[str, ...] = (
    "rank_pct",
    "cap_buffer",
    "d_limit_streak",
    "log_circ_mv",
    "log_fd_amount",
    "log_seal_amount",
    *PREMIUM_SCORE_FEATURES,
    *CROSS_SECTIONAL_FEATURES,
)

LIMIT_FEATURES: Tuple[str, ...] = (
    "up_limit",
    "limit_up_strength",
    "open_times",
    "fd_amount",
    "seal_amount",
    "break_count_proxy",
    "is_hot_board",
    "board_rank",
    "board_limit_up_count",
)


@dataclass(frozen=True)
class ExecutionProfitDiagnostics:
    trade_date: str
    model_version: str = MODEL_VERSION
    ready: bool = False
    reason: str = "not_run"
    cost_bps: float = 35.0
    history_days: int = 0
    fill_rows: int = 0
    return_rows: int = 0
    limit_rows: int = 0
    feature_count: int = 0
    fill_auc: float = float("nan")
    fill_brier: float = float("nan")
    fill_baseline_brier: float = float("nan")
    profit_rank_ic: float = float("nan")
    top_quintile_mean_net_return: float = float("nan")
    top_quintile_lift: float = float("nan")
    policy_days: int = 0
    policy_trade_days: int = 0
    policy_filled_days: int = 0
    policy_winning_filled_days: int = 0
    policy_filled_win_rate: float = float("nan")
    policy_mean_net_return: float = float("nan")
    policy_net_return_lcb: float = float("nan")
    policy_compound_return: float = float("nan")
    policy_max_drawdown: float = float("nan")
    policy_ten_percent_compound_return: float = float("nan")
    policy_ten_percent_max_drawdown: float = float("nan")
    forced_top_days: int = 0
    forced_top_filled_days: int = 0
    forced_top_mean_net_return: float = float("nan")
    forced_top_compound_return: float = float("nan")
    forced_top_max_drawdown: float = float("nan")
    post_release_days: int = 0
    post_release_trade_days: int = 0
    post_release_filled_days: int = 0
    post_release_winning_filled_days: int = 0
    post_release_mean_net_return: float = float("nan")
    post_release_net_return_lcb: float = float("nan")
    post_release_compound_return: float = float("nan")
    walk_forward_folds: int = 0
    embargo_days: int = EMBARGO_DAYS
    release_d_date: str = MODEL_RELEASE_D_DATE
    training_start_date: str = ""
    training_end_date: str = ""
    selected_fill_features: Tuple[str, ...] = ()
    selected_return_features: Tuple[str, ...] = ()
    data_fingerprint: str = ""
    feature_fingerprint: str = ""
    positive_fold_rate: float = float("nan")
    validation_mode: str = VALIDATION_MODE
    truth_source: str = "premium_execution_truth_ledger.csv"
    error_type: str = ""
    error_message: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in names:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


def _numeric(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    col = _first_col(df, names)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_code(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    col = _first_col(out, ["ts_code", "code", "symbol"])
    if col is None:
        out["ts_code"] = ""
    elif col != "ts_code":
        out = out.rename(columns={col: "ts_code"})
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    return out


def _date_from_path(path: Path) -> str:
    match = DATE_RE.search(path.name)
    return match.group(1) if match else ""


def _cost_bps() -> float:
    try:
        return float(os.getenv("PREMIUM_EXECUTION_COST_BPS", "35"))
    except Exception:
        return 35.0


def _candidate_dates(market_root: Path, trade_date: str, max_days: int) -> List[str]:
    dates = sorted(
        {
            _date_from_path(path)
            for path in market_root.glob("daily_*.csv")
            if _date_from_path(path) and _date_from_path(path) <= trade_date
        }
    )
    return dates[-max_days:]


def _load_daily_panel(market_root: Path, dates: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for date in dates:
        path = market_root / f"daily_{date}.csv"
        frame = _normalize_code(_read_csv(path))
        if frame.empty:
            continue
        frame["d_date"] = date
        keep = ["d_date", "ts_code", "open", "high", "low", "close", "vol", "amount"]
        for col in keep:
            if col not in frame.columns:
                frame[col] = np.nan
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame()

    daily = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("open", "high", "low", "close", "vol", "amount"):
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.drop_duplicates(["d_date", "ts_code"]).sort_values(["ts_code", "d_date"]).reset_index(drop=True)
    group = daily.groupby("ts_code", sort=False)
    previous_close = group["close"].shift(1)
    daily["ret_1d"] = daily["close"] / previous_close - 1.0
    daily["gap_d"] = daily["open"] / previous_close - 1.0
    daily["intraday_d"] = daily["close"] / daily["open"] - 1.0
    daily["range_d"] = (daily["high"] - daily["low"]) / previous_close
    spread = (daily["high"] - daily["low"]).replace(0.0, np.nan)
    daily["close_pos_d"] = (daily["close"] - daily["low"]) / spread
    daily["one_price_d"] = ((daily["high"] - daily["low"]).abs() <= 0.001).astype(float)
    daily["log_amount_d"] = np.log1p(daily["amount"].clip(lower=0.0))
    daily["log_vol_d"] = np.log1p(daily["vol"].clip(lower=0.0))
    for window in (2, 3, 5, 10):
        daily[f"mom_{window}d"] = daily["close"] / group["close"].shift(window) - 1.0
    for window in (3, 5, 10):
        amount_mean = group["amount"].transform(lambda s: s.rolling(window, min_periods=2).mean())
        vol_mean = group["vol"].transform(lambda s: s.rolling(window, min_periods=2).mean())
        daily[f"amount_ratio_{window}d"] = daily["amount"] / amount_mean
        daily[f"vol_ratio_{window}d"] = daily["vol"] / vol_mean
        daily[f"volatility_{window}d"] = group["ret_1d"].transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )
    rolling_high = group["close"].transform(lambda s: s.rolling(10, min_periods=2).max())
    daily["drawdown_10d"] = daily["close"] / rolling_high - 1.0

    market = daily.groupby("d_date", as_index=False).agg(
        market_up_ratio=("ret_1d", lambda s: float((s > 0).mean())),
        market_avg_ret=("ret_1d", "mean"),
        market_volatility=("ret_1d", "std"),
        market_amount=("amount", "sum"),
    )
    market["market_log_amount"] = np.log1p(market.pop("market_amount").clip(lower=0.0))
    return daily.merge(market, on="d_date", how="left")


def _load_limit_panel(market_root: Path, dates: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for date in dates:
        frame = _normalize_code(_read_csv(market_root / f"features_limit_{date}.csv"))
        if frame.empty:
            continue
        frame["d_date"] = date
        keep = ["d_date", "ts_code", *[col for col in LIMIT_FEATURES if col in frame.columns]]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=["d_date", "ts_code"])
    out = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(["d_date", "ts_code"])
    for col in LIMIT_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_frozen_history(out_root: Path, trade_date: str, allowed_dates: Iterable[str]) -> pd.DataFrame:
    allowed = set(allowed_dates)
    frames: List[pd.DataFrame] = []
    for path in sorted(out_root.glob("premium_verify_*.csv")):
        date = _date_from_path(path)
        if not date or date >= trade_date or date not in allowed:
            continue
        frame = _normalize_code(_read_csv(path))
        if frame.empty or not frame["ts_code"].ne("").any():
            continue
        frame["d_date"] = date
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(["d_date", "ts_code"], keep="last")


def _load_execution_truth(out_root: Path, trade_date: str) -> pd.DataFrame:
    path = Path(out_root) / "verification" / "premium_execution_truth_ledger.csv"
    frame = _normalize_code(_read_csv(path))
    required = {
        "schema_version",
        "d_trade_date",
        "rank",
        "ts_code",
        "fill_observed",
        "net_return",
        "model_eligible",
        "status",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    if not frame["schema_version"].dropna().astype(str).eq(TRUTH_SCHEMA_VERSION).all():
        raise RuntimeError("execution truth schema mismatch")
    frame = frame.copy()
    frame["d_date"] = (
        frame["d_trade_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    )
    frame = frame.loc[frame["d_date"].lt(str(trade_date))]
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["fill_actual"] = pd.to_numeric(frame["fill_observed"], errors="coerce")
    frame["net_ret"] = pd.to_numeric(frame["net_return"], errors="coerce")
    frame["model_eligible"] = pd.to_numeric(frame["model_eligible"], errors="coerce").fillna(0)
    frame["status"] = frame["status"].astype(str)
    settled = frame["status"].isin(
        {"READY", "NO_AUCTION_MATCH", "NO_FILL_PRICE_CAP", "NO_SELL_AT_1100"}
    )
    return (
        frame.loc[settled]
        .drop_duplicates(["d_date", "rank"], keep="last")
        .sort_values(["d_date", "rank"], kind="stable")
        .reset_index(drop=True)
    )


def _load_frozen_top10_features(out_root: Path, dates: Iterable[str]) -> pd.DataFrame:
    allowed = set(str(value) for value in dates)
    frames: List[pd.DataFrame] = []
    for path in sorted(Path(out_root).glob("premium_top10_*.csv")):
        d_date = _date_from_path(path)
        if d_date not in allowed:
            continue
        frame = _normalize_code(_read_csv(path))
        if frame.empty:
            continue
        frame = frame.copy()
        frame["d_date"] = d_date
        keep = ["d_date", "ts_code"]
        for column in (
            "close_T",
            "t_max_buy_price",
            "T日可接受买入价",
            "晋阶",
            *PREMIUM_SCORE_FEATURES,
        ):
            if column in frame.columns and column not in keep:
                keep.append(column)
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=["d_date", "ts_code"])
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(["d_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _attach_frozen_top10_features(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or panel.empty:
        return rows
    out = rows.copy()
    additions = panel.copy()
    value_columns = [column for column in additions.columns if column not in {"d_date", "ts_code"}]
    additions = additions.rename(columns={column: f"{column}__frozen" for column in value_columns})
    out = out.merge(additions, on=["d_date", "ts_code"], how="left")
    for column in value_columns:
        frozen = f"{column}__frozen"
        if column in out.columns:
            out[column] = out[column].where(out[column].notna(), out[frozen])
        else:
            out[column] = out[frozen]
        out = out.drop(columns=[frozen])
    return out


def _parse_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce",
    )


def _truth_from_market(
    history: pd.DataFrame,
    daily: pd.DataFrame,
    limit: pd.DataFrame,
    calendar: Sequence[str],
    trade_date: str,
    cost_rate: float,
) -> pd.DataFrame:
    if history.empty:
        return history
    date_pos = {date: idx for idx, date in enumerate(calendar)}
    date_map: Dict[str, Tuple[str, str]] = {}
    for date in history["d_date"].astype(str).unique():
        idx = date_pos.get(date)
        if idx is not None and idx + 2 < len(calendar) and calendar[idx + 2] <= trade_date:
            date_map[date] = (calendar[idx + 1], calendar[idx + 2])
    out = history[history["d_date"].isin(date_map)].copy()
    if out.empty:
        return out
    out["t_date"] = out["d_date"].map(lambda d: date_map[str(d)][0])
    out["t1_date"] = out["d_date"].map(lambda d: date_map[str(d)][1])

    raw_daily = daily[["d_date", "ts_code", "open", "high", "low", "close"]].copy()
    t = raw_daily.rename(
        columns={"d_date": "t_date", "open": "t_open", "high": "t_high", "low": "t_low", "close": "t_close"}
    )
    t1 = raw_daily.rename(
        columns={"d_date": "t1_date", "open": "t1_open", "high": "t1_high", "low": "t1_low", "close": "t1_close"}
    )
    out = out.merge(t, on=["t_date", "ts_code"], how="left")
    out = out.merge(t1, on=["t1_date", "ts_code"], how="left")

    out = out.drop(columns=["t_up_limit_exec"], errors="ignore")
    if "up_limit" in limit.columns:
        t_limit = limit[["d_date", "ts_code", "up_limit"]].rename(
            columns={"d_date": "t_date", "up_limit": "t_up_limit_exec"}
        )
        out = out.merge(t_limit, on=["t_date", "ts_code"], how="left")
    else:
        out["t_up_limit_exec"] = np.nan

    cap_col = _first_col(out, ["t_max_buy_price", "T日可接受买入价", "max_buy_price"])
    cap = _parse_price(out[cap_col]) if cap_col else pd.Series(np.nan, index=out.index)
    prices_ready = out[["t_open", "t_high", "t_low", "t_close", "t1_close"]].notna().all(axis=1)
    official_limit = pd.to_numeric(out["t_up_limit_exec"], errors="coerce")
    limit_hit = prices_ready & official_limit.notna() & (out["t_close"] >= official_limit * 0.9985)
    one_price_limit = limit_hit & ((out["t_high"] - out["t_low"]).abs() <= 0.001)
    under_cap = cap.isna() | (out["t_open"] <= cap * 1.0015)
    out["fill_proxy"] = np.where(prices_ready, (under_cap & ~one_price_limit).astype(int), np.nan)
    out["t_limitup_hit_exec"] = np.where(prices_ready & official_limit.notna(), limit_hit.astype(int), np.nan)
    gross = out["t1_close"] / out["t_open"] - 1.0
    out["gross_ret"] = np.where(prices_ready, gross, np.nan)
    out["net_ret"] = np.where(prices_ready, gross - cost_rate, np.nan)
    return out


def _attach_features(rows: pd.DataFrame, daily: pd.DataFrame, limit: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    if not daily.empty and {"d_date", "ts_code"}.issubset(daily.columns):
        daily_features = daily.drop(
            columns=["open", "high", "low", "close", "vol", "amount"], errors="ignore"
        )
        out = out.merge(
            daily_features,
            on=["d_date", "ts_code"],
            how="left",
            suffixes=("", "_market"),
        )
    if not limit.empty and {"d_date", "ts_code"}.issubset(limit.columns):
        d_limit = limit.drop(columns=["up_limit"], errors="ignore")
        out = out.merge(
            d_limit,
            on=["d_date", "ts_code"],
            how="left",
            suffixes=("", "_limit"),
        )
    rank = _numeric(out, ["rank", "rank_premium_final", "rank_adaptive_score"])
    count = out.groupby("d_date")["ts_code"].transform("count").clip(lower=1)
    # Frozen verification files contain TOP30 while current full output can be
    # wider.  Keep the rank scale invariant instead of leaking pool width.
    out["rank_pct"] = (rank / 30.0).clip(lower=0.0, upper=2.0)
    out["candidate_count"] = count.astype(float)
    d_close = (
        pd.to_numeric(out["close_T"], errors="coerce")
        if "close_T" in out.columns
        else pd.to_numeric(out["d_close"], errors="coerce")
        if "d_close" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    if "max_buy_price" in out.columns:
        max_buy = _parse_price(out["max_buy_price"])
    elif "t_max_buy_price" in out.columns:
        max_buy = _parse_price(out["t_max_buy_price"])
    elif "T日可接受买入价" in out.columns:
        max_buy = _parse_price(out["T日可接受买入价"])
    else:
        max_buy = pd.Series(np.nan, index=out.index)
    out["cap_buffer"] = max_buy / d_close - 1.0
    stage_col = _first_col(out, ["晋阶", "limit_stage", "board_stage"])
    if stage_col:
        out["d_limit_streak"] = pd.to_numeric(
            out[stage_col].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
    else:
        out["d_limit_streak"] = np.nan
    out["log_circ_mv"] = np.log1p(
        _numeric(out, ["circ_mv", "float_mv"]).clip(lower=0.0)
    )
    out["log_fd_amount"] = np.log1p(_numeric(out, ["fd_amount"]).clip(lower=0.0))
    out["log_seal_amount"] = np.log1p(_numeric(out, ["seal_amount"]).clip(lower=0.0))
    for base in CROSS_SECTIONAL_BASES:
        values = _numeric(out, [base])
        out[f"xs_{base}"] = values.groupby(out["d_date"]).rank(
            method="average", pct=True
        )
    for feature in FEATURES:
        if feature not in out.columns:
            out[feature] = np.nan
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out


def _date_split(frame: pd.DataFrame, fraction: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(frame["d_date"].astype(str).unique())
    if len(dates) < 2:
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy()
    cut = max(1, min(len(dates) - 1, int(len(dates) * fraction)))
    return frame[frame["d_date"].isin(dates[:cut])], frame[frame["d_date"].isin(dates[cut:])]


def _linear_classifier() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.12, max_iter=3000)),
        ]
    )


def _tree_classifier() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=120,
                    max_leaf_nodes=7,
                    min_samples_leaf=35,
                    l2_regularization=10.0,
                    random_state=31,
                ),
            ),
        ]
    )


def _linear_regressor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=12.0)),
        ]
    )


def _tree_regressor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.035,
                    max_iter=140,
                    max_leaf_nodes=7,
                    min_samples_leaf=20,
                    l2_regularization=12.0,
                    loss="squared_error",
                    random_state=47,
                ),
            ),
        ]
    )


def _date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    if "d_date" not in frame.columns or frame.empty:
        return np.ones(len(frame), dtype=float)
    counts = frame.groupby("d_date")["d_date"].transform("size").astype(float)
    weights = 1.0 / counts.clip(lower=1.0)
    mean = float(weights.mean()) if len(weights) else 1.0
    return (weights / mean).to_numpy(dtype=float) if mean > 0 else np.ones(len(frame), dtype=float)


def _fit_pair(frame: pd.DataFrame, features: Sequence[str], target: str) -> Optional[Tuple[Pipeline, Pipeline]]:
    if frame.empty or frame[target].nunique() < 2:
        return None
    linear = _linear_classifier()
    tree = _tree_classifier()
    weights = _date_equal_weights(frame)
    linear.fit(frame[list(features)], frame[target].astype(int), model__sample_weight=weights)
    tree.fit(frame[list(features)], frame[target].astype(int), model__sample_weight=weights)
    return linear, tree


def _fit_regression_pair(
    frame: pd.DataFrame, features: Sequence[str], target: str
) -> Optional[Tuple[Pipeline, Pipeline]]:
    values = pd.to_numeric(frame.get(target), errors="coerce")
    usable = frame.loc[values.notna()].copy()
    if len(usable) < 40 or values.loc[usable.index].nunique() < 5:
        return None
    target_values = values.loc[usable.index].clip(-0.20, 0.20)
    linear = _linear_regressor()
    tree = _tree_regressor()
    weights = _date_equal_weights(usable)
    linear.fit(usable[list(features)], target_values, model__sample_weight=weights)
    tree.fit(usable[list(features)], target_values, model__sample_weight=weights)
    return linear, tree


def _predict_pair(
    pair: Tuple[Pipeline, Pipeline], frame: pd.DataFrame, features: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = pair[0].predict_proba(frame[list(features)])[:, 1]
    right = pair[1].predict_proba(frame[list(features)])[:, 1]
    return 0.5 * (left + right), np.minimum(left, right), np.maximum(left, right)


def _predict_regression_pair(
    pair: Tuple[Pipeline, Pipeline], frame: pd.DataFrame, features: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(pair[0].predict(frame[list(features)]), dtype=float)
    right = np.asarray(pair[1].predict(frame[list(features)]), dtype=float)
    blend = np.clip(0.5 * (left + right), -0.20, 0.20)
    lower = np.clip(np.minimum(left, right), -0.20, 0.20)
    upper = np.clip(np.maximum(left, right), -0.20, 0.20)
    return blend, lower, upper


def _purged_date_folds(
    dates: Sequence[str], *, n_splits: int = 3, embargo_days: int = 2
) -> List[Tuple[set[str], set[str]]]:
    ordered = sorted(set(str(value) for value in dates))
    if len(ordered) < 24:
        return []
    first_valid = max(16 + embargo_days, int(len(ordered) * 0.45))
    validation_dates = ordered[first_valid:]
    if len(validation_dates) < n_splits * 3:
        return []
    blocks = [list(block) for block in np.array_split(validation_dates, n_splits) if len(block)]
    folds: List[Tuple[set[str], set[str]]] = []
    positions = {date: index for index, date in enumerate(ordered)}
    for block in blocks:
        valid_start = positions[block[0]]
        train_end = valid_start - embargo_days - 1
        if train_end < 11:
            continue
        folds.append((set(ordered[: train_end + 1]), set(block)))
    return folds


def _daily_rank_ic(frame: pd.DataFrame, prediction: str, actual: str) -> float:
    values: List[float] = []
    for _, group in frame.groupby("d_date", sort=True):
        tmp = group[[prediction, actual]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) < 3 or tmp[prediction].nunique() < 2 or tmp[actual].nunique() < 2:
            continue
        value = tmp[prediction].rank().corr(tmp[actual].rank())
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _select_training_features(
    frame: pd.DataFrame,
    current: pd.DataFrame,
    target: str,
    *,
    max_features: int,
    min_features: int = 8,
) -> List[str]:
    """Select a compact feature set using training dates only."""

    dates = sorted(frame["d_date"].astype(str).unique())
    midpoint = max(1, len(dates) // 2)
    early_dates = set(dates[:midpoint])
    late_dates = set(dates[midpoint:])
    scored: List[Tuple[float, str]] = []
    fallbacks: List[Tuple[int, str]] = []
    min_rows = 30 if target == "fill_actual" else 20
    for feature in FEATURES:
        if feature not in frame.columns or feature not in current.columns:
            continue
        present = int(frame[feature].notna().sum())
        if present < min_rows or not current[feature].notna().any():
            continue
        if pd.to_numeric(frame[feature], errors="coerce").nunique(dropna=True) < 2:
            continue
        fallbacks.append((present, feature))
        overall = _daily_rank_ic(frame, feature, target)
        early = _daily_rank_ic(frame.loc[frame["d_date"].isin(early_dates)], feature, target)
        late = _daily_rank_ic(frame.loc[frame["d_date"].isin(late_dates)], feature, target)
        if not np.isfinite(overall):
            continue
        same_sign = (
            np.isfinite(early)
            and np.isfinite(late)
            and np.sign(early) == np.sign(late)
            and np.sign(overall) == np.sign(early)
        )
        stability = min(abs(float(early)), abs(float(late))) if same_sign else 0.0
        coverage = present / max(1, len(frame))
        score = coverage * (abs(float(overall)) + 0.50 * stability)
        if not same_sign:
            score *= 0.35
        scored.append((score, feature))

    ordered = [feature for _, feature in sorted(scored, reverse=True)]
    ordered.extend(
        feature
        for _, feature in sorted(fallbacks, reverse=True)
        if feature not in ordered
    )
    preferred = ["rank_pct", "cap_buffer", "d_limit_streak"]
    ordered = [feature for feature in preferred if feature in ordered] + [
        feature for feature in ordered if feature not in preferred
    ]

    selected: List[str] = []
    numeric = frame[list(dict.fromkeys(ordered))].apply(pd.to_numeric, errors="coerce")
    for feature in ordered:
        if len(selected) >= max_features:
            break
        correlated = False
        for kept in selected:
            pair = numeric[[feature, kept]].dropna()
            if len(pair) >= min_rows:
                corr = pair[feature].corr(pair[kept], method="spearman")
                if np.isfinite(corr) and abs(float(corr)) >= 0.96:
                    correlated = True
                    break
        if not correlated:
            selected.append(feature)
    return selected if len(selected) >= min_features else []


def _fingerprint_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return ""
    stable = frame[selected].copy()
    sort_columns = [
        column for column in ("d_date", "rank", "ts_code") if column in stable.columns
    ]
    if sort_columns:
        stable = stable.sort_values(sort_columns, kind="stable")
    payload = stable.to_csv(index=False, na_rep="").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _score_components(
    p_fill_lcb: np.ndarray,
    p_profit_lcb: np.ndarray,
    p_big_loss_ucb: np.ndarray,
    expected_return_lcb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    safe_ev = np.asarray(p_fill_lcb) * np.asarray(expected_return_lcb)
    edge = safe_ev - 0.015 * np.asarray(p_big_loss_ucb)
    return np.clip(safe_ev, -0.20, 0.20), np.clip(edge, -0.25, 0.25)


def _candidate_gate(
    p_fill_lcb: np.ndarray,
    p_profit_lcb: np.ndarray,
    p_big_loss_ucb: np.ndarray,
    expected_return_lcb: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    return (
        (np.asarray(p_fill_lcb) >= 0.45)
        & (np.asarray(p_profit_lcb) >= 0.52)
        & (np.asarray(p_big_loss_ucb) <= 0.35)
        & (np.asarray(expected_return_lcb) >= 0.003)
        & (np.asarray(edge) > 0.0)
    )


def _policy_returns(frame: pd.DataFrame) -> Tuple[pd.Series, int, int, int]:
    returns: List[float] = []
    trade_days = 0
    filled_days = 0
    winning_filled_days = 0
    for _, group in frame.groupby("d_date", sort=True):
        candidates = group.loc[group["candidate_gate"].astype(bool)].sort_values(
            "execution_score_oof", ascending=False, kind="stable"
        )
        if candidates.empty:
            returns.append(0.0)
            continue
        selected = candidates.iloc[0]
        fill_value = pd.to_numeric(selected.get("fill_actual"), errors="coerce")
        actual = pd.to_numeric(selected.get("net_ret"), errors="coerce")
        if not np.isfinite(fill_value) or (int(fill_value) == 1 and not np.isfinite(actual)):
            # The selected trade cannot be evaluated yet; do not turn missing
            # truth into an artificial zero return.
            continue
        trade_days += 1
        if int(fill_value) != 1:
            returns.append(0.0)
            continue
        filled_days += 1
        if float(actual) > 0.0:
            winning_filled_days += 1
        returns.append(float(actual))
    return pd.Series(returns, dtype=float), trade_days, filled_days, winning_filled_days


def _forced_top_returns(frame: pd.DataFrame) -> Tuple[pd.Series, int, int]:
    returns: List[float] = []
    filled_days = 0
    winning_filled_days = 0
    for _, group in frame.groupby("d_date", sort=True):
        candidates = group.sort_values(
            "execution_score_oof", ascending=False, kind="stable"
        )
        if candidates.empty:
            continue
        selected = candidates.iloc[0]
        fill_value = pd.to_numeric(selected.get("fill_actual"), errors="coerce")
        actual = pd.to_numeric(selected.get("net_ret"), errors="coerce")
        if not np.isfinite(fill_value) or (int(fill_value) == 1 and not np.isfinite(actual)):
            continue
        if int(fill_value) != 1:
            returns.append(0.0)
            continue
        filled_days += 1
        winning_filled_days += int(float(actual) > 0.0)
        returns.append(float(actual))
    return pd.Series(returns, dtype=float), filled_days, winning_filled_days


def _return_summary(values: pd.Series) -> Dict[str, float]:
    returns = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if returns.empty:
        return {
            "mean": float("nan"),
            "lcb": float("nan"),
            "compound": float("nan"),
            "drawdown": float("nan"),
            "ten_percent_compound": float("nan"),
            "ten_percent_drawdown": float("nan"),
        }
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    lcb = mean - std / np.sqrt(len(returns)) if np.isfinite(std) else float("nan")
    equity = (1.0 + returns).cumprod()
    scaled_equity = (1.0 + 0.10 * returns).cumprod()
    return {
        "mean": mean,
        "lcb": lcb,
        "compound": float(equity.iloc[-1] - 1.0),
        "drawdown": float((equity / equity.cummax() - 1.0).min()),
        "ten_percent_compound": float(scaled_equity.iloc[-1] - 1.0),
        "ten_percent_drawdown": float(
            (scaled_equity / scaled_equity.cummax() - 1.0).min()
        ),
    }


def _rank_ic(prediction: np.ndarray, actual: pd.Series) -> float:
    tmp = pd.DataFrame({"prediction": prediction, "actual": pd.to_numeric(actual, errors="coerce")}).dropna()
    if len(tmp) < 3 or tmp["prediction"].nunique() < 2 or tmp["actual"].nunique() < 2:
        return float("nan")
    return float(tmp["prediction"].rank().corr(tmp["actual"].rank()))


def _failed_frame(
    candidates: pd.DataFrame, diagnostics: ExecutionProfitDiagnostics
) -> Tuple[pd.DataFrame, ExecutionProfitDiagnostics]:
    out = candidates.copy().reset_index(drop=True)
    out["exec_p_fill"] = np.nan
    out["exec_p_fill_lcb"] = np.nan
    out["exec_p_profit_lcb"] = np.nan
    out["exec_p_limitup_lcb"] = np.nan
    out["exec_p_big_loss_ucb"] = np.nan
    out["exec_expected_net_return"] = np.nan
    out["exec_conditional_net_return"] = np.nan
    out["exec_conditional_net_return_lcb"] = np.nan
    out["exec_profit_edge"] = np.nan
    out["execution_profit_score"] = np.nan
    out["exec_trade_eligible"] = 0
    out["exec_model_ready"] = 0
    out["exec_model_reason"] = diagnostics.reason
    out["exec_model_version"] = diagnostics.model_version
    out["exec_cost_bps"] = diagnostics.cost_bps
    out["exec_fill_auc"] = diagnostics.fill_auc
    out["exec_profit_rank_ic"] = diagnostics.profit_rank_ic
    out["exec_holdout_top_net"] = diagnostics.top_quintile_mean_net_return
    return out, diagnostics


def execution_backtest_payload(
    diagnostics: ExecutionProfitDiagnostics,
) -> Dict[str, object]:
    """Serialize the executable policy audit separately from Rank1 baseline."""

    return {
        "schema_version": "premium_execution_profit_walk_forward_v2",
        "model_version": diagnostics.model_version,
        "ready": diagnostics.ready,
        "reason": diagnostics.reason,
        "window_start": diagnostics.training_start_date,
        "window_end": diagnostics.training_end_date,
        "a_share_trading_days": diagnostics.history_days,
        "walk_forward_folds": diagnostics.walk_forward_folds,
        "embargo_days": diagnostics.embargo_days,
        "signals": diagnostics.policy_trade_days,
        "filled_signals": diagnostics.policy_filled_days,
        "winning_filled_signals": diagnostics.policy_winning_filled_days,
        "filled_win_rate": diagnostics.policy_filled_win_rate,
        "mean_daily_net_return": diagnostics.policy_mean_net_return,
        "mean_daily_net_return_lcb": diagnostics.policy_net_return_lcb,
        "full_position_compound_return": diagnostics.policy_compound_return,
        "full_position_max_drawdown": diagnostics.policy_max_drawdown,
        "ten_percent_position_compound_return": diagnostics.policy_ten_percent_compound_return,
        "ten_percent_position_max_drawdown": diagnostics.policy_ten_percent_max_drawdown,
        "forced_top_days": diagnostics.forced_top_days,
        "forced_top_filled_signals": diagnostics.forced_top_filled_days,
        "forced_top_mean_daily_net_return": diagnostics.forced_top_mean_net_return,
        "forced_top_compound_return": diagnostics.forced_top_compound_return,
        "forced_top_max_drawdown": diagnostics.forced_top_max_drawdown,
        "recent_holdout_signals": diagnostics.post_release_trade_days,
        "recent_holdout_filled_signals": diagnostics.post_release_filled_days,
        "recent_holdout_winning_filled_signals": diagnostics.post_release_winning_filled_days,
        "recent_holdout_compound_return": diagnostics.post_release_compound_return,
        "post_release_days": diagnostics.post_release_days,
        "post_release_mean_net_return": diagnostics.post_release_mean_net_return,
        "post_release_net_return_lcb": diagnostics.post_release_net_return_lcb,
        "release_d_date": diagnostics.release_d_date,
        "fill_auc": diagnostics.fill_auc,
        "fill_brier": diagnostics.fill_brier,
        "fill_baseline_brier": diagnostics.fill_baseline_brier,
        "profit_rank_ic": diagnostics.profit_rank_ic,
        "positive_fold_rate": diagnostics.positive_fold_rate,
        "selected_fill_features": list(diagnostics.selected_fill_features),
        "selected_return_features": list(diagnostics.selected_return_features),
        "data_fingerprint": diagnostics.data_fingerprint,
        "feature_fingerprint": diagnostics.feature_fingerprint,
        "validation_mode": diagnostics.validation_mode,
        "truth_source": diagnostics.truth_source,
        "cost_bps": diagnostics.cost_bps,
        "entry": "T official opening-auction matched price with published price cap",
        "exit": "T+1 exact 11:00 one-minute bar open",
        "validation": (
            "Purged expanding walk-forward with a two-trading-day embargo; "
            "NO_TRADE is retained as zero return and forced TOP1 is audit-only."
        ),
        "warning": (
            "Shadow evidence only until all model and post-release gates pass; "
            "market auction matched volume is not broker order-level fill confirmation."
        ),
    }


def score_execution_candidates(
    candidates: pd.DataFrame,
    *,
    out_root: Path,
    market_root: Path,
    trade_date: str,
    max_market_days: int = 105,
) -> Tuple[pd.DataFrame, ExecutionProfitDiagnostics]:
    cost_bps = _cost_bps()
    initial = ExecutionProfitDiagnostics(trade_date=trade_date, cost_bps=cost_bps)
    if candidates is None or candidates.empty:
        return _failed_frame(pd.DataFrame(), initial)

    try:
        history = _load_execution_truth(Path(out_root), trade_date)
        if history.empty:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                trade_date=trade_date, cost_bps=cost_bps, reason="actual_execution_truth_missing"
            ))
        calendar = _candidate_dates(Path(market_root), trade_date, max_market_days)
        daily = _load_daily_panel(Path(market_root), calendar) if calendar else pd.DataFrame()
        limit = _load_limit_panel(Path(market_root), calendar) if calendar else pd.DataFrame()
        frozen_features = _load_frozen_top10_features(
            Path(out_root), history["d_date"].astype(str).unique()
        )
        history = _attach_frozen_top10_features(history, frozen_features)
        history = _attach_features(history, daily, limit)
        if history.empty:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                trade_date=trade_date, cost_bps=cost_bps, reason="actual_execution_truth_missing"
            ))

        fill = history[history["fill_actual"].notna()].copy()
        returns = history[
            history["status"].eq("READY")
            & history["model_eligible"].eq(1)
            & history["net_ret"].notna()
        ].copy()
        history_days = int(history["d_date"].nunique())
        if history_days < 24 or len(fill) < 180 or len(returns) < 120:
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=history_days,
                fill_rows=len(fill),
                return_rows=len(returns),
                limit_rows=0,
                reason="actual_execution_truth_samples_insufficient",
            )
            return _failed_frame(candidates, diagnostics)

        returns["profit_hit"] = (returns["net_ret"] > 0.0).astype(int)
        returns["big_loss_hit"] = (returns["net_ret"] <= -0.03).astype(int)

        current = candidates.copy().reset_index(drop=True)
        current["d_date"] = trade_date
        current = _normalize_code(current)
        current_features = _attach_features(current, daily, limit)
        folds = _purged_date_folds(
            sorted(fill["d_date"].astype(str).unique()),
            embargo_days=EMBARGO_DAYS,
        )
        if len(folds) < 2:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=history_days,
                fill_rows=len(fill),
                return_rows=len(returns),
                reason="actual_truth_walk_forward_folds_insufficient",
            ))

        first_train_dates = folds[0][0]
        feature_fill_train = fill.loc[fill["d_date"].isin(first_train_dates)]
        feature_return_train = returns.loc[returns["d_date"].isin(first_train_dates)]
        fill_features = _select_training_features(
            feature_fill_train,
            current_features,
            "fill_actual",
            max_features=12,
        )
        return_features = _select_training_features(
            feature_return_train,
            current_features,
            "net_ret",
            max_features=10,
        )
        selected_features = sorted(set(fill_features) | set(return_features))
        if len(fill_features) < 8 or len(return_features) < 8:
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=history_days,
                fill_rows=len(fill),
                return_rows=len(returns),
                feature_count=len(selected_features),
                selected_fill_features=tuple(fill_features),
                selected_return_features=tuple(return_features),
                reason="actual_execution_features_insufficient",
            )
            return _failed_frame(candidates, diagnostics)

        oof_frames: List[pd.DataFrame] = []
        fold_policy_means: List[float] = []
        for train_dates, valid_dates in folds:
            fill_train = fill.loc[fill["d_date"].isin(train_dates)].copy()
            fill_valid = fill.loc[fill["d_date"].isin(valid_dates)].copy()
            return_train = returns.loc[returns["d_date"].isin(train_dates)].copy()
            if len(fill_train) < 100 or len(fill_valid) < 20 or len(return_train) < 60:
                continue
            pairs = {
                "fill": _fit_pair(fill_train, fill_features, "fill_actual"),
                "profit": _fit_pair(return_train, return_features, "profit_hit"),
                "loss": _fit_pair(return_train, return_features, "big_loss_hit"),
            }
            regression = _fit_regression_pair(return_train, return_features, "net_ret")
            if any(pair is None for pair in pairs.values()) or regression is None:
                continue

            p_fill, p_fill_lcb, _ = _predict_pair(  # type: ignore[arg-type]
                pairs["fill"], fill_valid, fill_features
            )
            _, p_profit_lcb, _ = _predict_pair(  # type: ignore[arg-type]
                pairs["profit"], fill_valid, return_features
            )
            _, _, p_big_loss_ucb = _predict_pair(  # type: ignore[arg-type]
                pairs["loss"], fill_valid, return_features
            )
            expected_return, expected_return_lcb, _ = _predict_regression_pair(
                regression, fill_valid, return_features
            )
            safe_ev, edge = _score_components(
                p_fill_lcb, p_profit_lcb, p_big_loss_ucb, expected_return_lcb
            )
            validation = fill_valid.copy()
            validation["p_fill_oof"] = p_fill
            validation["p_fill_lcb_oof"] = p_fill_lcb
            validation["p_profit_lcb_oof"] = p_profit_lcb
            validation["p_big_loss_ucb_oof"] = p_big_loss_ucb
            validation["expected_return_oof"] = expected_return
            validation["expected_return_lcb_oof"] = expected_return_lcb
            validation["safe_ev_oof"] = safe_ev
            validation["execution_score_oof"] = edge
            validation["candidate_gate"] = _candidate_gate(
                p_fill_lcb,
                p_profit_lcb,
                p_big_loss_ucb,
                expected_return_lcb,
                edge,
            )
            validation["fill_baseline_oof"] = float(fill_train["fill_actual"].mean())
            oof_frames.append(validation)
            fold_returns, _, _, _ = _policy_returns(validation)
            fold_policy_means.append(float(fold_returns.mean()) if len(fold_returns) else 0.0)

        if len(oof_frames) < 2:
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=history_days,
                fill_rows=len(fill),
                return_rows=len(returns),
                feature_count=len(selected_features),
                selected_fill_features=tuple(fill_features),
                selected_return_features=tuple(return_features),
                reason="actual_execution_walk_forward_fit_insufficient",
            )
            return _failed_frame(candidates, diagnostics)

        oof = pd.concat(oof_frames, ignore_index=True, sort=False)
        fill_auc = float(roc_auc_score(oof["fill_actual"], oof["p_fill_oof"]))
        fill_brier = float(brier_score_loss(oof["fill_actual"], oof["p_fill_oof"]))
        fill_baseline_brier = float(
            brier_score_loss(oof["fill_actual"], oof["fill_baseline_oof"])
        )
        ready_oof = oof.loc[oof["status"].eq("READY") & oof["net_ret"].notna()].copy()
        rank_ic = _daily_rank_ic(ready_oof, "execution_score_oof", "net_ret")
        policy_returns, policy_trade_days, policy_filled_days, policy_wins = _policy_returns(oof)
        policy_days = int(len(policy_returns))
        policy_summary = _return_summary(policy_returns)
        forced_returns, forced_filled_days, _ = _forced_top_returns(oof)
        forced_summary = _return_summary(forced_returns)
        post_release = oof.loc[oof["d_date"].astype(str).gt(MODEL_RELEASE_D_DATE)]
        post_returns, post_trade_days, post_filled_days, post_wins = _policy_returns(post_release)
        post_summary = _return_summary(post_returns)
        baseline_returns: List[float] = []
        for _, group in oof.groupby("d_date", sort=True):
            original = group.sort_values("rank", kind="stable").iloc[0]
            actual = pd.to_numeric(original.get("net_ret"), errors="coerce")
            filled = pd.to_numeric(original.get("fill_actual"), errors="coerce")
            baseline_returns.append(float(actual) if filled == 1 and np.isfinite(actual) else 0.0)
        baseline_mean = float(np.mean(baseline_returns)) if baseline_returns else float("nan")
        top_mean = policy_summary["mean"]
        top_lift = top_mean - baseline_mean if np.isfinite(top_mean) and np.isfinite(baseline_mean) else float("nan")
        positive_fold_rate = (
            float(np.mean(np.asarray(fold_policy_means) > 0.0))
            if fold_policy_means
            else float("nan")
        )

        gate_checks = {
            "truth_days": history_days >= 30,
            "actual_returns": len(returns) >= 150,
            "fill_auc": fill_auc >= 0.54,
            "fill_brier": fill_brier <= fill_baseline_brier,
            "profit_rank_ic": np.isfinite(rank_ic) and rank_ic >= 0.03,
            "policy_trades": policy_trade_days >= 8,
            "top_net_return": np.isfinite(top_mean) and top_mean >= 0.003,
            "net_return_lcb": np.isfinite(policy_summary["lcb"]) and policy_summary["lcb"] > 0.0,
            "top_lift": np.isfinite(top_lift) and top_lift > 0.0,
            "fold_stability": np.isfinite(positive_fold_rate) and positive_fold_rate >= (2.0 / 3.0),
            "drawdown": np.isfinite(policy_summary["drawdown"]) and policy_summary["drawdown"] >= -0.15,
            "post_release_days": len(post_returns) >= 10,
            "post_release_trades": post_trade_days >= 3,
            "post_release_mean": np.isfinite(post_summary["mean"]) and post_summary["mean"] > 0.0,
            "post_release_lcb": np.isfinite(post_summary["lcb"]) and post_summary["lcb"] > 0.0,
        }
        ready = all(gate_checks.values())
        reason = "ok_active" if ready else "guarded:" + ",".join(name for name, passed in gate_checks.items() if not passed)
        diagnostics = ExecutionProfitDiagnostics(
            trade_date=trade_date,
            cost_bps=cost_bps,
            history_days=history_days,
            fill_rows=len(fill),
            return_rows=len(returns),
            limit_rows=0,
            feature_count=len(selected_features),
            ready=ready,
            reason=reason,
            fill_auc=fill_auc,
            fill_brier=fill_brier,
            fill_baseline_brier=fill_baseline_brier,
            profit_rank_ic=rank_ic,
            top_quintile_mean_net_return=top_mean,
            top_quintile_lift=top_lift,
            policy_days=policy_days,
            policy_trade_days=policy_trade_days,
            policy_filled_days=policy_filled_days,
            policy_winning_filled_days=policy_wins,
            policy_filled_win_rate=(
                float(policy_wins / policy_filled_days) if policy_filled_days else float("nan")
            ),
            policy_mean_net_return=policy_summary["mean"],
            policy_net_return_lcb=policy_summary["lcb"],
            policy_compound_return=policy_summary["compound"],
            policy_max_drawdown=policy_summary["drawdown"],
            policy_ten_percent_compound_return=policy_summary["ten_percent_compound"],
            policy_ten_percent_max_drawdown=policy_summary["ten_percent_drawdown"],
            forced_top_days=len(forced_returns),
            forced_top_filled_days=forced_filled_days,
            forced_top_mean_net_return=forced_summary["mean"],
            forced_top_compound_return=forced_summary["compound"],
            forced_top_max_drawdown=forced_summary["drawdown"],
            post_release_days=len(post_returns),
            post_release_trade_days=post_trade_days,
            post_release_filled_days=post_filled_days,
            post_release_winning_filled_days=post_wins,
            post_release_mean_net_return=post_summary["mean"],
            post_release_net_return_lcb=post_summary["lcb"],
            post_release_compound_return=post_summary["compound"],
            walk_forward_folds=len(oof_frames),
            training_start_date=str(history["d_date"].min()),
            training_end_date=str(history["d_date"].max()),
            selected_fill_features=tuple(fill_features),
            selected_return_features=tuple(return_features),
            data_fingerprint=_fingerprint_frame(
                history,
                ["d_date", "rank", "ts_code", "status", "fill_actual", "net_ret", *selected_features],
            ),
            feature_fingerprint=hashlib.sha256(
                json.dumps(
                    {"fill": fill_features, "return": return_features},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            positive_fold_rate=positive_fold_rate,
        )

        full_pairs = {
            "fill": _fit_pair(fill, fill_features, "fill_actual"),
            "profit": _fit_pair(returns, return_features, "profit_hit"),
            "loss": _fit_pair(returns, return_features, "big_loss_hit"),
        }
        full_regression = _fit_regression_pair(returns, return_features, "net_ret")
        if any(pair is None for pair in full_pairs.values()) or full_regression is None:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                **{**diagnostics.as_dict(), "ready": False, "reason": "execution_refit_failed"}
            ))

        p_fill, p_fill_lcb, _ = _predict_pair(  # type: ignore[arg-type]
            full_pairs["fill"], current_features, fill_features
        )
        _, p_profit_lcb, _ = _predict_pair(  # type: ignore[arg-type]
            full_pairs["profit"], current_features, return_features
        )
        _, _, p_big_loss_ucb = _predict_pair(  # type: ignore[arg-type]
            full_pairs["loss"], current_features, return_features
        )
        expected_return, expected_return_lcb, _ = _predict_regression_pair(
            full_regression, current_features, return_features
        )
        safe_ev, profit_edge = _score_components(
            p_fill_lcb, p_profit_lcb, p_big_loss_ucb, expected_return_lcb
        )
        candidate_gate = _candidate_gate(
            p_fill_lcb,
            p_profit_lcb,
            p_big_loss_ucb,
            expected_return_lcb,
            profit_edge,
        )
        p_limit_lcb = _numeric(
            current, ["t_limitup_prob", "T日涨停概率"], default=np.nan
        ).to_numpy(dtype=float)
        finite_limit = p_limit_lcb[np.isfinite(p_limit_lcb)]
        if len(finite_limit) and float(np.nanmedian(finite_limit)) > 1.0:
            p_limit_lcb = p_limit_lcb / 100.0
        p_limit_lcb = np.clip(p_limit_lcb, 0.0, 1.0)

        current["exec_p_fill"] = p_fill
        current["exec_p_fill_lcb"] = p_fill_lcb
        current["exec_p_profit_lcb"] = p_profit_lcb
        current["exec_p_limitup_lcb"] = p_limit_lcb
        current["exec_p_big_loss_ucb"] = p_big_loss_ucb
        current["exec_expected_net_return"] = safe_ev
        current["exec_conditional_net_return"] = expected_return
        current["exec_conditional_net_return_lcb"] = expected_return_lcb
        current["exec_profit_edge"] = profit_edge
        current["execution_profit_score"] = pd.Series(profit_edge).rank(method="average", pct=True).to_numpy() * 100.0
        current_rank = _numeric(current, ["rank", "rank_premium_final", "rank_adaptive_score"])
        current["exec_trade_eligible"] = (
            ready
            & (current_rank <= 10)
            & candidate_gate
        ).astype(int)
        current["exec_model_ready"] = int(ready)
        current["exec_model_reason"] = reason
        current["exec_model_version"] = MODEL_VERSION
        current["exec_cost_bps"] = cost_bps
        current["exec_fill_auc"] = fill_auc
        current["exec_profit_rank_ic"] = rank_ic
        current["exec_holdout_top_net"] = top_mean
        return current, diagnostics
    except Exception as exc:
        error_message = re.sub(r"\s+", " ", str(exc)).strip()[:240]
        reason = f"execution_model_error:{type(exc).__name__}"
        if error_message:
            reason = f"{reason}:{error_message}"
        diagnostics = ExecutionProfitDiagnostics(
            trade_date=trade_date,
            cost_bps=cost_bps,
            reason=reason,
            error_type=type(exc).__name__,
            error_message=error_message,
        )
        return _failed_frame(candidates, diagnostics)


__all__ = [
    "ExecutionProfitDiagnostics",
    "MODEL_VERSION",
    "execution_backtest_payload",
    "score_execution_candidates",
]
