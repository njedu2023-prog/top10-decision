#!/usr/bin/env python3
"""Guarded T-auction to T+1-close execution-profit model.

The model is intentionally isolated from Decision.  It reconstructs labels
from frozen Premium predictions and the A-share market cache, validates on the
latest chronological holdout, and permits a buy only when the holdout gate and
the candidate-level conservative EV gate both pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "premium_exec_profit_v1_walkforward_guarded"
DATE_RE = re.compile(r"(20\d{6})")

FEATURES: Tuple[str, ...] = (
    "rank_pct",
    "ret_1d",
    "gap_d",
    "intraday_d",
    "range_d",
    "close_pos_d",
    "one_price_d",
    "log_amount_d",
    "log_vol_d",
    "mom_2d",
    "mom_3d",
    "mom_5d",
    "mom_10d",
    "amount_ratio_3d",
    "amount_ratio_5d",
    "amount_ratio_10d",
    "vol_ratio_3d",
    "vol_ratio_5d",
    "vol_ratio_10d",
    "volatility_3d",
    "volatility_5d",
    "volatility_10d",
    "drawdown_10d",
    "market_up_ratio",
    "market_avg_ret",
    "market_volatility",
    "market_log_amount",
    "limit_up_strength",
    "open_times",
    "fd_amount",
    "seal_amount",
    "break_count_proxy",
    "is_hot_board",
    "board_rank",
    "board_limit_up_count",
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

    if "up_limit" in limit.columns:
        t_limit = limit[["d_date", "ts_code", "up_limit"]].rename(columns={"d_date": "t_date"})
        out = out.merge(t_limit, on=["t_date", "ts_code"], how="left")
    else:
        out["up_limit"] = np.nan

    cap_col = _first_col(out, ["t_max_buy_price", "T日可接受买入价", "max_buy_price"])
    cap = _parse_price(out[cap_col]) if cap_col else pd.Series(np.nan, index=out.index)
    prices_ready = out[["t_open", "t_high", "t_low", "t_close", "t1_close"]].notna().all(axis=1)
    official_limit = pd.to_numeric(out["up_limit"], errors="coerce")
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
    daily_features = daily.drop(columns=["open", "high", "low", "close", "vol", "amount"], errors="ignore")
    out = out.merge(daily_features, on=["d_date", "ts_code"], how="left", suffixes=("", "_market"))
    d_limit = limit.drop(columns=["up_limit"], errors="ignore")
    out = out.merge(d_limit, on=["d_date", "ts_code"], how="left", suffixes=("", "_limit"))
    rank = _numeric(out, ["rank", "rank_premium_final", "rank_adaptive_score"])
    count = out.groupby("d_date")["ts_code"].transform("count").clip(lower=1)
    # Frozen verification files contain TOP30 while current full output can be
    # wider.  Keep the rank scale invariant instead of leaking pool width.
    out["rank_pct"] = (rank / 30.0).clip(lower=0.0, upper=2.0)
    out["candidate_count"] = count.astype(float)
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


def _fit_pair(frame: pd.DataFrame, features: Sequence[str], target: str) -> Optional[Tuple[Pipeline, Pipeline]]:
    if frame.empty or frame[target].nunique() < 2:
        return None
    linear = _linear_classifier()
    tree = _tree_classifier()
    linear.fit(frame[list(features)], frame[target].astype(int))
    tree.fit(frame[list(features)], frame[target].astype(int))
    return linear, tree


def _predict_pair(
    pair: Tuple[Pipeline, Pipeline], frame: pd.DataFrame, features: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = pair[0].predict_proba(frame[list(features)])[:, 1]
    right = pair[1].predict_proba(frame[list(features)])[:, 1]
    return 0.5 * (left + right), np.minimum(left, right), np.maximum(left, right)


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


def score_execution_candidates(
    candidates: pd.DataFrame,
    *,
    out_root: Path,
    market_root: Path,
    trade_date: str,
    max_market_days: int = 105,
) -> Tuple[pd.DataFrame, ExecutionProfitDiagnostics]:
    cost_bps = _cost_bps()
    cost_rate = cost_bps / 10000.0
    initial = ExecutionProfitDiagnostics(trade_date=trade_date, cost_bps=cost_bps)
    if candidates is None or candidates.empty:
        return _failed_frame(pd.DataFrame(), initial)

    try:
        calendar = _candidate_dates(Path(market_root), trade_date, max_market_days)
        if trade_date not in calendar or len(calendar) < 30:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                trade_date=trade_date, cost_bps=cost_bps, reason="market_history_insufficient"
            ))
        daily = _load_daily_panel(Path(market_root), calendar)
        limit = _load_limit_panel(Path(market_root), calendar)
        history = _load_frozen_history(Path(out_root), trade_date, calendar)
        history = _truth_from_market(history, daily, limit, calendar, trade_date, cost_rate)
        history = _attach_features(history, daily, limit)
        if history.empty:
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                trade_date=trade_date, cost_bps=cost_bps, reason="frozen_history_missing"
            ))

        fill = history[history["fill_proxy"].notna()].copy()
        returns = history[(history["fill_proxy"] == 1) & history["net_ret"].notna()].copy()
        limit_rows = history[history["t_limitup_hit_exec"].notna()].copy()
        if len(fill) < 450 or len(returns) < 300 or len(limit_rows) < 500:
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=int(history["d_date"].nunique()),
                fill_rows=len(fill),
                return_rows=len(returns),
                limit_rows=len(limit_rows),
                reason="execution_samples_insufficient",
            )
            return _failed_frame(candidates, diagnostics)

        returns["profit_hit"] = (returns["net_ret"] > 0.0).astype(int)
        returns["big_loss_hit"] = (returns["net_ret"] <= -0.03).astype(int)
        limit_rows["limit_hit"] = limit_rows["t_limitup_hit_exec"].astype(int)
        fill_train, fill_valid = _date_split(fill)
        return_train, return_valid = _date_split(returns)
        limit_train, limit_valid = _date_split(limit_rows)

        current = candidates.copy().reset_index(drop=True)
        current["d_date"] = trade_date
        current = _normalize_code(current)
        current_features = _attach_features(current, daily, limit)
        usable = [
            feature
            for feature in FEATURES
            if fill_train[feature].notna().sum() >= 30
            and return_train[feature].notna().sum() >= 30
            and return_valid[feature].notna().sum() >= 5
            and current_features[feature].notna().any()
        ]
        if len(usable) < 8:
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=int(history["d_date"].nunique()),
                fill_rows=len(fill),
                return_rows=len(returns),
                limit_rows=len(limit_rows),
                feature_count=len(usable),
                reason="execution_features_insufficient",
            )
            return _failed_frame(candidates, diagnostics)

        validation_pairs = {
            "fill": _fit_pair(fill_train, usable, "fill_proxy"),
            "profit": _fit_pair(return_train, usable, "profit_hit"),
            "loss": _fit_pair(return_train, usable, "big_loss_hit"),
            "limit": _fit_pair(limit_train, usable, "limit_hit"),
        }
        if any(pair is None for pair in validation_pairs.values()):
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date,
                cost_bps=cost_bps,
                history_days=int(history["d_date"].nunique()),
                fill_rows=len(fill),
                return_rows=len(returns),
                limit_rows=len(limit_rows),
                feature_count=len(usable),
                reason="execution_validation_single_class",
            )
            return _failed_frame(candidates, diagnostics)

        fill_valid_pred = _predict_pair(validation_pairs["fill"], fill_valid, usable)[0]  # type: ignore[arg-type]
        profit_valid = _predict_pair(validation_pairs["profit"], return_valid, usable)[0]  # type: ignore[arg-type]
        loss_valid = _predict_pair(validation_pairs["loss"], return_valid, usable)[0]  # type: ignore[arg-type]
        limit_valid_pred = _predict_pair(validation_pairs["limit"], return_valid, usable)[0]  # type: ignore[arg-type]

        gain = float(return_train.loc[return_train["net_ret"] > 0, "net_ret"].clip(upper=0.15).mean())
        loss = float(return_train.loc[return_train["net_ret"] <= 0, "net_ret"].clip(lower=-0.15).mean())
        if not all(np.isfinite(value) for value in (gain, loss)):
            diagnostics = ExecutionProfitDiagnostics(
                trade_date=trade_date, cost_bps=cost_bps, reason="execution_payoff_missing"
            )
            return _failed_frame(candidates, diagnostics)

        fill_on_return_valid = _predict_pair(validation_pairs["fill"], return_valid, usable)[0]  # type: ignore[arg-type]
        # Ranking objective: conservative probability of a profitable trade,
        # penalized by the upper-bound probability of a loss worse than 3%.
        validation_score = fill_on_return_valid * (profit_valid - 0.50) - 0.25 * loss_valid
        rank_ic = _rank_ic(validation_score, return_valid["net_ret"])
        cutoff = float(np.quantile(validation_score, 0.80))
        top = return_valid.loc[validation_score >= cutoff, "net_ret"]
        top_mean = float(top.mean()) if len(top) else float("nan")
        top_lift = top_mean - float(return_valid["net_ret"].mean()) if np.isfinite(top_mean) else float("nan")
        fill_auc = float(roc_auc_score(fill_valid["fill_proxy"], fill_valid_pred))
        fill_brier = float(brier_score_loss(fill_valid["fill_proxy"], fill_valid_pred))
        baseline = np.repeat(float(fill_train["fill_proxy"].mean()), len(fill_valid))
        fill_baseline_brier = float(brier_score_loss(fill_valid["fill_proxy"], baseline))

        gate_checks = {
            "fill_auc": fill_auc >= 0.55,
            "fill_brier": fill_brier <= fill_baseline_brier,
            "profit_rank_ic": np.isfinite(rank_ic) and rank_ic >= 0.03,
            "top_net_return": np.isfinite(top_mean) and top_mean >= 0.005,
            "top_lift": np.isfinite(top_lift) and top_lift > 0.0,
        }
        ready = all(gate_checks.values())
        reason = "ok_active" if ready else "guarded:" + ",".join(name for name, passed in gate_checks.items() if not passed)
        diagnostics = ExecutionProfitDiagnostics(
            trade_date=trade_date,
            cost_bps=cost_bps,
            history_days=int(history["d_date"].nunique()),
            fill_rows=len(fill),
            return_rows=len(returns),
            limit_rows=len(limit_rows),
            feature_count=len(usable),
            ready=ready,
            reason=reason,
            fill_auc=fill_auc,
            fill_brier=fill_brier,
            fill_baseline_brier=fill_baseline_brier,
            profit_rank_ic=rank_ic,
            top_quintile_mean_net_return=top_mean,
            top_quintile_lift=top_lift,
        )

        full_pairs = {
            "fill": _fit_pair(fill, usable, "fill_proxy"),
            "profit": _fit_pair(returns, usable, "profit_hit"),
            "loss": _fit_pair(returns, usable, "big_loss_hit"),
            "limit": _fit_pair(limit_rows, usable, "limit_hit"),
        }
        if any(pair is None for pair in full_pairs.values()):
            return _failed_frame(candidates, ExecutionProfitDiagnostics(
                **{**diagnostics.as_dict(), "ready": False, "reason": "execution_refit_failed"}
            ))

        p_fill, p_fill_lcb, _ = _predict_pair(full_pairs["fill"], current_features, usable)  # type: ignore[arg-type]
        _, p_profit_lcb, _ = _predict_pair(full_pairs["profit"], current_features, usable)  # type: ignore[arg-type]
        _, _, p_big_loss_ucb = _predict_pair(full_pairs["loss"], current_features, usable)  # type: ignore[arg-type]
        _, p_limit_lcb, _ = _predict_pair(full_pairs["limit"], current_features, usable)  # type: ignore[arg-type]
        profit_edge = p_fill_lcb * (p_profit_lcb - 0.50) - 0.25 * p_big_loss_ucb
        safe_ev = p_fill_lcb * (p_profit_lcb * gain + (1.0 - p_profit_lcb) * loss)

        current["exec_p_fill"] = p_fill
        current["exec_p_fill_lcb"] = p_fill_lcb
        current["exec_p_profit_lcb"] = p_profit_lcb
        current["exec_p_limitup_lcb"] = p_limit_lcb
        current["exec_p_big_loss_ucb"] = p_big_loss_ucb
        current["exec_expected_net_return"] = safe_ev
        current["exec_profit_edge"] = profit_edge
        current["execution_profit_score"] = pd.Series(profit_edge).rank(method="average", pct=True).to_numpy() * 100.0
        current_rank = _numeric(current, ["rank", "rank_premium_final", "rank_adaptive_score"])
        current["exec_trade_eligible"] = (
            ready
            & (current_rank <= 30)
            & (profit_edge >= 0.0)
            & (safe_ev > 0.0)
            & (p_fill_lcb >= 0.55)
            & (p_big_loss_ucb <= 0.25)
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
        diagnostics = ExecutionProfitDiagnostics(
            trade_date=trade_date,
            cost_bps=cost_bps,
            reason=f"execution_model_error:{type(exc).__name__}",
        )
        return _failed_frame(candidates, diagnostics)


__all__ = [
    "ExecutionProfitDiagnostics",
    "MODEL_VERSION",
    "score_execution_candidates",
]
