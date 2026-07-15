from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import AuctionV3Config


DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
EPS = 1e-9

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "source_rank": ("rank", "rank_v2", "排名"),
    "prior_probability": ("prob_final", "Probability", "prob", "p_limit_up_calibrated"),
    "strength_score": ("StrengthScore", "strength_score", "强度分"),
    "theme_boost": ("ThemeBoost", "theme_boost", "题材加成"),
    "final_score": ("final_score_v2", "final_score", "rank_score", "最终分"),
    "intraday_quality": ("intraday_quality_score", "limitup_quality_score", "分时质量"),
    "intraday_risk": ("intraday_risk_score", "intraday_soft_risk_score", "软风险"),
    "intraday_hard_risk": ("intraday_hard_risk_flag", "硬风险"),
    "auction_strength": ("auction_strength_score", "竞价强度"),
    "intraday_confidence": ("intraday_confidence_score",),
    "stage_quality": ("stage_quality_weight",),
    "stage_risk": ("stage_risk_weight", "stage_risk_penalty"),
    "stage_prior": ("stage_prior",),
    "limit_times": ("limit_times",),
    "open_board_count": ("open_board_count", "open_times"),
    "reseal_score": ("reseal_score",),
    "late_withdraw": ("late_withdraw_score",),
}

MODEL_FEATURES = [
    "source_rank",
    "prior_probability",
    "strength_score",
    "theme_boost",
    "final_score",
    "intraday_quality",
    "intraday_risk",
    "intraday_hard_risk",
    "auction_strength",
    "intraday_confidence",
    "stage_quality",
    "stage_risk",
    "stage_prior",
    "limit_times",
    "open_board_count",
    "reseal_score",
    "late_withdraw",
    "d_return",
    "d_range",
    "d_turnover_proxy",
    "d_amount_log",
    "limit_ratio",
    "proposed_gap",
]


@dataclass
class RunResult:
    signal_date: str
    prediction_path: str
    backtest_path: str
    verification_path: str
    current_report_path: str
    verification_report_path: str
    dashboard_path: str
    model_ready: bool
    promoted: bool
    selected_count: int
    warnings: list[str]


@dataclass
class ModelBundle:
    return_model: Pipeline
    profit_model: Optional[Pipeline]
    loss_model: Optional[Pipeline]
    fill_model: Optional[Pipeline]
    profit_constant: float
    loss_constant: float
    fill_constant: float
    residual_q10: float
    residual_q90: float
    gap_min: float
    gap_max: float
    train_rows: int
    train_dates: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normal_date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normal_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        return f"{left.zfill(6)}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        symbol = digits[-6:]
        suffix = "SH" if symbol.startswith(("5", "6", "9")) else "BJ" if symbol.startswith(("4", "8")) else "SZ"
        return f"{symbol}.{suffix}"
    return text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _safe_metric(value: Any) -> Optional[float]:
    result = _finite(value)
    return round(result, 10) if math.isfinite(result) else None


def _numeric_from(row: pd.Series, aliases: Sequence[str], default: float = float("nan")) -> float:
    for name in aliases:
        if name not in row.index:
            continue
        value = _finite(row.get(name))
        if math.isfinite(value):
            return value
    return default


def _is_close(left: Any, right: Any, tolerance: float = 0.0025) -> bool:
    a, b = _finite(left), _finite(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(0.01, abs(b) * tolerance)


def _round_price(value: float) -> float:
    return round(max(0.01, value) + 1e-9, 2)


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _business_day_after(date_text: str) -> str:
    date = pd.Timestamp(datetime.strptime(date_text, "%Y%m%d"))
    return (date + pd.offsets.BDay(1)).strftime("%Y%m%d")


def _probability(model: Optional[Pipeline], frame: pd.DataFrame, constant: float) -> np.ndarray:
    if model is None:
        return np.repeat(float(np.clip(constant, 0.0, 1.0)), len(frame))
    return np.clip(model.predict_proba(frame[MODEL_FEATURES])[:, 1], 0.0, 1.0)


class AuctionV3Engine:
    """Builds immutable predictions, walk-forward evidence, and matured truth ledgers."""

    def __init__(self, config: AuctionV3Config):
        self.config = config
        self.config.ensure_directories()
        self._market_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Source discovery and point-in-time inputs
    # ------------------------------------------------------------------
    def market_dates(self) -> list[str]:
        root = self.config.root / "data" / "market" / "raw"
        dates: set[str] = set()
        if root.exists():
            for path in root.rglob("daily.csv"):
                if re.fullmatch(r"20\d{6}", path.parent.name):
                    dates.add(path.parent.name)
            for path in root.glob("daily_*.csv"):
                match = DATE_RE.search(path.name)
                if match:
                    dates.add(match.group(1))
        return sorted(dates)

    def _market_path(self, trade_date: str, name: str) -> Optional[Path]:
        root = self.config.root / "data" / "market" / "raw"
        candidates = [
            root / trade_date[:4] / trade_date / f"{name}.csv",
            root / trade_date / f"{name}.csv",
            root / f"{name}_{trade_date}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def market_table(self, trade_date: str, name: str) -> pd.DataFrame:
        key = (trade_date, name)
        if key in self._market_cache:
            return self._market_cache[key]
        path = self._market_path(trade_date, name)
        frame = _read_csv(path) if path else pd.DataFrame()
        if not frame.empty and "ts_code" in frame.columns:
            frame = frame.copy()
            frame["ts_code"] = frame["ts_code"].map(_normal_code)
            frame = frame.drop_duplicates("ts_code", keep="last").set_index("ts_code", drop=False)
        self._market_cache[key] = frame
        return frame

    def candidate_snapshots(self) -> dict[str, Path]:
        pred_root = self.config.root / "data" / "pred"
        paths: list[Path] = []
        if pred_root.exists():
            paths.extend(pred_root.rglob("pred_source_*.csv"))
            paths.extend(pred_root.rglob("pred_decisio_*.csv"))
        selected: dict[str, Path] = {}
        for path in sorted(set(paths)):
            frame = _read_csv(path)
            if frame.empty:
                continue
            trade_date = ""
            if "trade_date" in frame.columns:
                values = frame["trade_date"].dropna().map(_normal_date)
                values = values[values.str.len() == 8]
                if not values.empty:
                    trade_date = values.iloc[0]
            if not trade_date:
                match = DATE_RE.search(path.name)
                trade_date = match.group(1) if match else ""
            if not trade_date:
                continue
            old = selected.get(trade_date)
            if old is None or ("archive" in path.parts and "archive" not in old.parts):
                selected[trade_date] = path
        latest = pred_root / "pred_source_latest.csv"
        if latest.exists():
            frame = _read_csv(latest)
            if not frame.empty and "trade_date" in frame.columns:
                td = _normal_date(frame["trade_date"].iloc[0])
                if td:
                    selected[td] = latest
        return dict(sorted(selected.items()))

    def load_candidates(self, signal_date: str, path: Optional[Path] = None) -> pd.DataFrame:
        path = path or self.candidate_snapshots().get(signal_date)
        frame = _read_csv(path) if path else pd.DataFrame()
        if frame.empty:
            return frame
        frame = frame.copy()
        code_col = next((c for c in ("ts_code", "code", "代码") if c in frame.columns), "")
        if not code_col:
            return pd.DataFrame()
        frame["ts_code"] = frame[code_col].map(_normal_code)
        frame = frame[frame["ts_code"] != ""].drop_duplicates("ts_code", keep="first")
        rank_col = next((c for c in ("rank", "rank_v2", "排名") if c in frame.columns), "")
        frame["_source_rank"] = pd.to_numeric(frame[rank_col], errors="coerce") if rank_col else np.arange(1, len(frame) + 1)
        frame = frame.sort_values("_source_rank", na_position="last").head(self.config.max_candidates)
        return frame.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Historical truth and feature construction
    # ------------------------------------------------------------------
    def _row(self, frame: pd.DataFrame, code: str) -> Optional[pd.Series]:
        if frame.empty or code not in frame.index:
            return None
        row = frame.loc[code]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def _limit_ratio(self, daily_row: Optional[pd.Series], limit_row: Optional[pd.Series]) -> float:
        if daily_row is not None and limit_row is not None:
            pre_close = _numeric_from(daily_row, ("pre_close", "pre_close_est"))
            up_limit = _numeric_from(limit_row, ("up_limit",))
            if pre_close > 0 and up_limit > 0:
                ratio = up_limit / pre_close - 1.0
                if 0.03 <= ratio <= 0.35:
                    return ratio
        return 0.10

    def _one_price_limit(self, daily_row: Optional[pd.Series], limit_row: Optional[pd.Series], side: str) -> bool:
        if daily_row is None or limit_row is None:
            return False
        limit_name = "up_limit" if side == "up" else "down_limit"
        limit_price = _numeric_from(limit_row, (limit_name,))
        if not math.isfinite(limit_price):
            return False
        return all(_is_close(daily_row.get(name), limit_price) for name in ("open", "high", "low", "close"))

    def _auction_amount(self, trade_date: str, code: str) -> float:
        row = self._row(self.market_table(trade_date, "stk_auction"), code)
        if row is None:
            return float("nan")
        return _numeric_from(row, ("amount", "auction_amount"))

    def _auction_price(self, trade_date: str, code: str) -> float:
        row = self._row(self.market_table(trade_date, "stk_auction"), code)
        if row is None:
            return float("nan")
        return _numeric_from(row, ("price", "auction_price"))

    def _market_buyable(self, trade_date: str, code: str) -> tuple[int, str]:
        daily = self._row(self.market_table(trade_date, "daily"), code)
        limit = self._row(self.market_table(trade_date, "stk_limit"), code)
        if daily is None:
            return 0, "suspended_or_daily_missing"
        open_price = _numeric_from(daily, ("open",))
        auction_price = self._auction_price(trade_date, code)
        if math.isfinite(auction_price) and open_price > 0 and abs(auction_price - open_price) > 0.011:
            return 0, "auction_daily_open_conflict"
        if self._one_price_limit(daily, limit, "up"):
            return 0, "one_price_limit_up"
        auction_amount = self._auction_amount(trade_date, code)
        if math.isfinite(auction_amount) and auction_amount > 0:
            capacity = auction_amount * self.config.max_auction_participation
            if capacity + EPS < self.config.order_amount_cny:
                return 0, "auction_capacity_insufficient"
        return 1, "market_proxy_buyable"

    def _realized_gross_return(
        self,
        code: str,
        buy_date: str,
        buy_price: float,
        exit_date: str,
        dates: Sequence[str],
    ) -> float:
        """Chain close/pre-close returns so corporate actions do not create fake PnL."""
        if buy_price <= 0 or buy_date not in dates or exit_date not in dates:
            return float("nan")
        start, end = dates.index(buy_date), dates.index(exit_date)
        buy_row = self._row(self.market_table(buy_date, "daily"), code)
        if buy_row is None:
            return float("nan")
        buy_close = _numeric_from(buy_row, ("close",))
        if buy_close <= 0:
            return float("nan")
        wealth = buy_close / buy_price
        for idx in range(start + 1, end + 1):
            row = self._row(self.market_table(dates[idx], "daily"), code)
            if row is None:
                return float("nan")
            pre_close = _numeric_from(row, ("pre_close", "pre_close_est"))
            if pre_close <= 0:
                return float("nan")
            if idx == end:
                exit_open = _numeric_from(row, ("open",))
                if exit_open <= 0:
                    return float("nan")
                wealth *= exit_open / pre_close
            else:
                close_price = _numeric_from(row, ("close",))
                if close_price <= 0:
                    return float("nan")
                wealth *= close_price / pre_close
        return wealth - 1.0

    def _resolve_exit(self, code: str, start_index: int, dates: Sequence[str]) -> tuple[str, float, int, str]:
        for offset, trade_date in enumerate(dates[start_index:], start=0):
            daily = self._row(self.market_table(trade_date, "daily"), code)
            if daily is None:
                continue
            limit = self._row(self.market_table(trade_date, "stk_limit"), code)
            if self._one_price_limit(daily, limit, "down"):
                continue
            exit_price = _numeric_from(daily, ("open",))
            if exit_price > 0:
                return trade_date, exit_price, offset, "t1_open" if offset == 0 else "delayed_first_tradable_open"
        return "", float("nan"), -1, "exit_truth_pending"

    def _feature_dict(self, candidate: pd.Series, d_daily: Optional[pd.Series], limit_ratio: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for canonical, aliases in FEATURE_ALIASES.items():
            out[canonical] = _numeric_from(candidate, aliases)
        if d_daily is None:
            out.update({"d_return": np.nan, "d_range": np.nan, "d_turnover_proxy": np.nan, "d_amount_log": np.nan})
        else:
            open_price = _numeric_from(d_daily, ("open",))
            close_price = _numeric_from(d_daily, ("close",))
            high = _numeric_from(d_daily, ("high",))
            low = _numeric_from(d_daily, ("low",))
            pre_close = _numeric_from(d_daily, ("pre_close", "pre_close_est"))
            amount = _numeric_from(d_daily, ("amount",))
            out["d_return"] = close_price / pre_close - 1.0 if pre_close > 0 and close_price > 0 else _numeric_from(d_daily, ("pct_chg",)) / 100.0
            out["d_range"] = (high - low) / pre_close if pre_close > 0 and high > 0 and low > 0 else np.nan
            out["d_turnover_proxy"] = abs(close_price - open_price) / pre_close if pre_close > 0 and open_price > 0 else np.nan
            out["d_amount_log"] = math.log1p(amount) if amount > 0 else np.nan
        out["limit_ratio"] = limit_ratio
        out["proposed_gap"] = np.nan
        return out

    def build_history(self) -> pd.DataFrame:
        dates = self.market_dates()
        date_index = {date: idx for idx, date in enumerate(dates)}
        snapshots = self.candidate_snapshots()
        records: list[dict[str, Any]] = []
        for signal_date, path in snapshots.items():
            idx = date_index.get(signal_date)
            if idx is None or idx + 2 >= len(dates):
                continue
            buy_date, target_exit_date = dates[idx + 1], dates[idx + 2]
            candidates = self.load_candidates(signal_date, path)
            if candidates.empty:
                continue
            d_daily_table = self.market_table(signal_date, "daily")
            d_limit_table = self.market_table(signal_date, "stk_limit")
            buy_daily_table = self.market_table(buy_date, "daily")
            for _, candidate in candidates.iterrows():
                code = candidate["ts_code"]
                d_daily = self._row(d_daily_table, code)
                buy_daily = self._row(buy_daily_table, code)
                if d_daily is None or buy_daily is None:
                    continue
                d_close = _numeric_from(d_daily, ("close",))
                buy_open = _numeric_from(buy_daily, ("open",))
                if d_close <= 0 or buy_open <= 0:
                    continue
                d_limit = self._row(d_limit_table, code)
                limit_ratio = self._limit_ratio(d_daily, d_limit)
                market_fill, fill_reason = self._market_buyable(buy_date, code)
                exit_date, exit_price, delay_days, exit_reason = self._resolve_exit(code, idx + 2, dates)
                if not exit_date or exit_price <= 0:
                    continue
                gross_return = self._realized_gross_return(code, buy_date, buy_open, exit_date, dates)
                if not math.isfinite(gross_return):
                    continue
                net_return = gross_return - self.config.cost_rate
                features = self._feature_dict(candidate, d_daily, limit_ratio)
                features["proposed_gap"] = buy_open / d_close - 1.0
                records.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "target_exit_date": target_exit_date,
                        "actual_exit_date": exit_date,
                        "exit_delay_days": delay_days,
                        "ts_code": code,
                        "name": str(candidate.get("name", candidate.get("股票", ""))),
                        "stage": str(candidate.get("晋阶", candidate.get("advance_stage", ""))),
                        "source_rank": features["source_rank"],
                        "d_close": d_close,
                        "buy_open": buy_open,
                        "exit_open": exit_price,
                        "actual_buy_gap": features["proposed_gap"],
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "profit_hit": int(net_return > 0.0),
                        "big_loss_hit": int(net_return <= self.config.big_loss_threshold),
                        "market_fill": market_fill,
                        "fill_reason": fill_reason,
                        "exit_reason": exit_reason,
                        **features,
                    }
                )
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        frame = frame.sort_values(["signal_date", "source_rank", "ts_code"]).reset_index(drop=True)
        return frame

    # ------------------------------------------------------------------
    # Model fitting, cap optimization, and walk-forward evidence
    # ------------------------------------------------------------------
    def _regression_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.045,
                        max_iter=180,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=0.25,
                        random_state=20260716,
                    ),
                ),
            ]
        )

    def _classifier_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.045,
                        max_iter=160,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=0.25,
                        random_state=20260716,
                    ),
                ),
            ]
        )

    def fit_models(self, history: pd.DataFrame) -> Optional[ModelBundle]:
        if history.empty:
            return None
        all_clean = history.dropna(subset=["net_return", "proposed_gap", "market_fill"]).copy()
        clean = all_clean[all_clean["market_fill"].eq(1)].copy()
        date_count = clean["signal_date"].nunique()
        if len(clean) < self.config.min_train_rows or date_count < self.config.min_train_dates:
            return None
        X = clean[MODEL_FEATURES]
        return_model = self._regression_pipeline()
        return_model.fit(X, clean["net_return"])
        fitted = return_model.predict(X)
        residual = clean["net_return"].to_numpy() - fitted

        def fit_classifier(target: str) -> tuple[Optional[Pipeline], float]:
            values = clean[target].astype(int)
            constant = float(values.mean())
            counts = values.value_counts()
            if len(counts) < 2 or int(counts.min()) < 10:
                return None, constant
            model = self._classifier_pipeline()
            model.fit(X, values)
            return model, constant

        profit_model, profit_constant = fit_classifier("profit_hit")
        loss_model, loss_constant = fit_classifier("big_loss_hit")
        fill_parts: list[pd.DataFrame] = []
        proposal_grid = np.arange(
            self.config.gap_grid_min,
            self.config.gap_grid_max + self.config.gap_grid_step / 2.0,
            self.config.gap_grid_step,
        )
        for _, row in all_clean.iterrows():
            max_gap = min(self.config.gap_grid_max, _finite(row.get("limit_ratio"), 0.10))
            gaps = proposal_grid[proposal_grid <= max_gap + EPS]
            if not len(gaps):
                continue
            expanded = pd.DataFrame([row.to_dict()] * len(gaps))
            expanded["proposed_gap"] = gaps
            expanded["fill_at_cap"] = (
                (int(row["market_fill"]) == 1)
                & (float(row["actual_buy_gap"]) <= gaps + EPS)
            ).astype(int)
            fill_parts.append(expanded)
        fill_train = pd.concat(fill_parts, ignore_index=True) if fill_parts else pd.DataFrame()
        fill_values = fill_train.get("fill_at_cap", pd.Series(dtype=int)).astype(int)
        fill_constant = float(fill_values.mean()) if len(fill_values) else 0.0
        fill_model: Optional[Pipeline] = None
        fill_counts = fill_values.value_counts()
        if len(fill_counts) >= 2 and int(fill_counts.min()) >= 10:
            fill_model = self._classifier_pipeline()
            fill_model.fit(fill_train[MODEL_FEATURES], fill_values)
        return ModelBundle(
            return_model=return_model,
            profit_model=profit_model,
            loss_model=loss_model,
            fill_model=fill_model,
            profit_constant=profit_constant,
            loss_constant=loss_constant,
            fill_constant=fill_constant,
            residual_q10=float(np.quantile(residual, self.config.lower_confidence_quantile)),
            residual_q90=float(np.quantile(residual, self.config.prediction_interval_upper_quantile)),
            gap_min=float(clean["proposed_gap"].quantile(0.01)),
            gap_max=float(clean["proposed_gap"].quantile(0.99)),
            train_rows=len(clean),
            train_dates=date_count,
        )

    def _score_candidate_at_gaps(self, row: pd.Series, bundle: ModelBundle) -> Optional[dict[str, float]]:
        limit_ratio = _finite(row.get("limit_ratio"), 0.10)
        low = max(self.config.gap_grid_min, bundle.gap_min)
        high = min(self.config.gap_grid_max, bundle.gap_max, limit_ratio)
        if high < low:
            return None
        gaps = np.arange(low, high + self.config.gap_grid_step / 2.0, self.config.gap_grid_step)
        grid = pd.DataFrame([row.to_dict()] * len(gaps))
        grid["proposed_gap"] = gaps
        pred = bundle.return_model.predict(grid[MODEL_FEATURES])
        p_profit = _probability(bundle.profit_model, grid, bundle.profit_constant)
        p_loss = _probability(bundle.loss_model, grid, bundle.loss_constant)
        # A less aggressive limit price cannot have a higher execution chance.
        p_fill = np.maximum.accumulate(_probability(bundle.fill_model, grid, bundle.fill_constant))
        lower = pred + bundle.residual_q10
        upper = pred + bundle.residual_q90
        conservative_ev = p_fill * lower
        supported = (
            (conservative_ev >= self.config.min_edge)
            & (p_loss <= self.config.max_big_loss_probability)
            & (p_fill >= 0.55)
        )
        if not supported.any():
            return None
        supported_indices = np.where(supported)[0]
        chosen = int(supported_indices[np.argmax(conservative_ev[supported_indices])])
        return {
            "recommended_max_gap": float(gaps[chosen]),
            "predicted_net_return": float(pred[chosen]),
            "predicted_return_lcb": float(lower[chosen]),
            "predicted_return_ucb": float(upper[chosen]),
            "predicted_profit_probability": float(p_profit[chosen]),
            "predicted_big_loss_probability": float(p_loss[chosen]),
            "predicted_fill_probability": float(p_fill[chosen]),
            "conservative_ev": float(conservative_ev[chosen]),
        }

    def score_candidates(self, base: pd.DataFrame, bundle: Optional[ModelBundle]) -> pd.DataFrame:
        out = base.copy().reset_index(drop=True)
        score_columns = [
            "recommended_max_gap",
            "predicted_net_return",
            "predicted_return_lcb",
            "predicted_return_ucb",
            "predicted_profit_probability",
            "predicted_big_loss_probability",
            "predicted_fill_probability",
            "conservative_ev",
        ]
        for name in score_columns:
            out[name] = np.nan
        if bundle is None:
            out["model_reason"] = "insufficient_independent_history"
            out["selected"] = 0
            return out
        for index, row in out.iterrows():
            score = self._score_candidate_at_gaps(row, bundle)
            if score is None:
                continue
            for name, value in score.items():
                out.loc[index, name] = value
        out["model_reason"] = np.where(out["conservative_ev"].notna(), "ok", "no_safe_price")
        out = out.sort_values(
            ["conservative_ev", "predicted_return_lcb", "source_rank"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        out["selected"] = 0
        eligible = out.index[out["conservative_ev"].notna()][: self.config.max_positions]
        out.loc[eligible, "selected"] = 1
        return out

    def _current_base(self, signal_date: str, candidates: pd.DataFrame) -> pd.DataFrame:
        d_daily_table = self.market_table(signal_date, "daily")
        d_limit_table = self.market_table(signal_date, "stk_limit")
        rows: list[dict[str, Any]] = []
        for _, candidate in candidates.iterrows():
            code = candidate["ts_code"]
            d_daily = self._row(d_daily_table, code)
            if d_daily is None:
                continue
            d_close = _numeric_from(d_daily, ("close",))
            if d_close <= 0:
                continue
            d_limit = self._row(d_limit_table, code)
            limit_ratio = self._limit_ratio(d_daily, d_limit)
            features = self._feature_dict(candidate, d_daily, limit_ratio)
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "name": str(candidate.get("name", candidate.get("股票", ""))),
                    "stage": str(candidate.get("晋阶", candidate.get("advance_stage", ""))),
                    "source_rank": features["source_rank"],
                    "d_close": d_close,
                    "estimated_up_limit": _round_price(d_close * (1.0 + limit_ratio)),
                    **features,
                }
            )
        return pd.DataFrame(rows)

    def _walkforward_predictions(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty:
            return pd.DataFrame()
        dates = sorted(history["signal_date"].unique())
        output: list[pd.DataFrame] = []
        start = self.config.min_train_dates + self.config.embargo_dates
        for block_start in range(start, len(dates), self.config.backtest_block_dates):
            test_dates = dates[block_start : block_start + self.config.backtest_block_dates]
            train_end = block_start - self.config.embargo_dates
            train_dates = dates[:train_end]
            if len(train_dates) < self.config.min_train_dates:
                continue
            train = history[history["signal_date"].isin(train_dates)].copy()
            bundle = self.fit_models(train)
            if bundle is None:
                continue
            test = history[history["signal_date"].isin(test_dates)].copy()
            scored_parts: list[pd.DataFrame] = []
            for signal_date, group in test.groupby("signal_date", sort=True):
                scored = self.score_candidates(group, bundle)
                scored["oos_train_end"] = train_dates[-1]
                scored["oos_train_dates"] = len(train_dates)
                scored_parts.append(scored)
            if scored_parts:
                output.append(pd.concat(scored_parts, ignore_index=True))
        if not output:
            return pd.DataFrame()
        out = pd.concat(output, ignore_index=True)
        out["cap_accepted"] = (out["actual_buy_gap"] <= out["recommended_max_gap"] + EPS).astype(int)
        out["strategy_filled"] = (
            out["selected"].eq(1) & out["cap_accepted"].eq(1) & out["market_fill"].eq(1)
        ).astype(int)
        out["strategy_net_return"] = np.where(out["strategy_filled"].eq(1), out["net_return"], np.nan)
        out["forecast_error"] = np.where(
            out["strategy_filled"].eq(1), out["net_return"] - out["predicted_net_return"], np.nan
        )
        out["direction_success"] = np.where(
            out["strategy_filled"].eq(1),
            ((out["predicted_net_return"] > 0) == (out["net_return"] > 0)).astype(int),
            np.nan,
        )
        out["within_prediction_interval"] = np.where(
            out["strategy_filled"].eq(1),
            ((out["net_return"] >= out["predicted_return_lcb"]) & (out["net_return"] <= out["predicted_return_ucb"])).astype(int),
            np.nan,
        )
        return out

    def _block_bootstrap_positive_probability(self, daily: pd.Series, block: int = 5, samples: int = 1000) -> float:
        values = pd.to_numeric(daily, errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < block:
            return float("nan")
        rng = np.random.default_rng(20260716)
        starts = np.arange(0, max(1, len(values) - block + 1))
        means = []
        blocks_needed = int(math.ceil(len(values) / block))
        for _ in range(samples):
            sample = np.concatenate([values[s : s + block] for s in rng.choice(starts, blocks_needed, replace=True)])[: len(values)]
            means.append(float(np.mean(sample)))
        return float(np.mean(np.asarray(means) > 0.0))

    def _portfolio_metrics(self, oos: pd.DataFrame, history_dates: int) -> dict[str, Any]:
        if oos.empty:
            return {
                "status": "insufficient_oos_history",
                "history_dates": history_dates,
                "oos_dates": 0,
                "promoted": False,
                "promotion_failures": ["no_walkforward_predictions"],
            }
        selected = oos[oos["selected"].eq(1)].copy()
        filled = selected[selected["strategy_filled"].eq(1)].copy()
        dates = sorted(oos["signal_date"].unique())
        daily = filled.groupby("signal_date")["strategy_net_return"].sum().reindex(dates, fill_value=0.0)
        daily = daily / float(self.config.max_positions)
        benchmark_rows = oos[pd.to_numeric(oos["source_rank"], errors="coerce") <= self.config.max_positions].copy()
        benchmark_rows = benchmark_rows[benchmark_rows["market_fill"].eq(1)]
        benchmark = benchmark_rows.groupby("signal_date")["net_return"].sum().reindex(dates, fill_value=0.0)
        benchmark = benchmark / float(self.config.max_positions)
        uncapped_rows = selected[selected["market_fill"].eq(1)].copy()
        uncapped = uncapped_rows.groupby("signal_date")["net_return"].sum().reindex(dates, fill_value=0.0)
        uncapped = uncapped / float(self.config.max_positions)
        nav = (1.0 + daily).cumprod()
        drawdown = nav / nav.cummax() - 1.0
        stress_15_trade = filled["gross_return"] - 1.5 * self.config.cost_rate
        stress_15_daily = stress_15_trade.groupby(filled["signal_date"]).sum().reindex(dates, fill_value=0.0) / float(self.config.max_positions)
        stress_trade = filled["gross_return"] - 2.0 * self.config.cost_rate
        stress_daily = stress_trade.groupby(filled["signal_date"]).sum().reindex(dates, fill_value=0.0) / float(self.config.max_positions)
        positives = filled.loc[filled["strategy_net_return"] > 0, "strategy_net_return"].sum()
        negatives = -filled.loc[filled["strategy_net_return"] < 0, "strategy_net_return"].sum()
        rolling = daily.rolling(20).sum().dropna()
        month_keys = pd.Index(dates).str.slice(0, 6)
        monthly = daily.groupby(month_keys).sum()
        positive_month_ratio = float((monthly > 0).mean()) if len(monthly) else float("nan")
        positive_months = monthly[monthly > 0]
        max_month_contribution = float(positive_months.max() / positive_months.sum()) if len(positive_months) and positive_months.sum() > 0 else float("nan")
        mean_daily = float(daily.mean()) if len(daily) else float("nan")
        std_daily = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
        sharpe = mean_daily / std_daily * math.sqrt(252.0) if std_daily > 0 else float("nan")
        bootstrap = self._block_bootstrap_positive_probability(daily)
        rank_pair = filled[["predicted_net_return", "strategy_net_return"]].dropna()
        rank_ic = float(rank_pair.corr(method="spearman").iloc[0, 1]) if len(rank_pair) >= 5 else float("nan")
        calibration: list[dict[str, Any]] = []
        if len(rank_pair) >= 20:
            try:
                bins = pd.qcut(rank_pair["predicted_net_return"], q=min(5, rank_pair["predicted_net_return"].nunique()), duplicates="drop")
                grouped = rank_pair.assign(_bin=bins).groupby("_bin", observed=True)
                for label, group in grouped:
                    calibration.append(
                        {
                            "bucket": str(label),
                            "trades": int(len(group)),
                            "predicted_mean": _safe_metric(group["predicted_net_return"].mean()),
                            "actual_mean": _safe_metric(group["strategy_net_return"].mean()),
                        }
                    )
            except Exception:
                calibration = []
        metrics: dict[str, Any] = {
            "status": "ok",
            "generated_at_utc": _utc_now(),
            "model_version": self.config.model_version,
            "history_dates": history_dates,
            "oos_dates": len(dates),
            "signals": int(len(selected)),
            "filled_trades": int(len(filled)),
            "fill_rate": _safe_metric(len(filled) / len(selected) if len(selected) else np.nan),
            "mean_trade_net_return": _safe_metric(filled["strategy_net_return"].mean()),
            "median_trade_net_return": _safe_metric(filled["strategy_net_return"].median()),
            "win_rate": _safe_metric((filled["strategy_net_return"] > 0).mean()),
            "profit_factor": _safe_metric(positives / negatives if negatives > 0 else np.nan),
            "mean_daily_return": _safe_metric(mean_daily),
            "stress_1_5x_cost_mean_daily_return": _safe_metric(stress_15_daily.mean()),
            "stress_2x_cost_mean_daily_return": _safe_metric(stress_daily.mean()),
            "benchmark_mean_daily_return": _safe_metric(benchmark.mean()),
            "uncapped_model_mean_daily_return": _safe_metric(uncapped.mean()),
            "cumulative_return": _safe_metric(nav.iloc[-1] - 1.0 if len(nav) else np.nan),
            "max_drawdown": _safe_metric(drawdown.min() if len(drawdown) else np.nan),
            "sharpe": _safe_metric(sharpe),
            "positive_20d_window_ratio": _safe_metric((rolling > 0).mean() if len(rolling) else np.nan),
            "positive_month_ratio": _safe_metric(positive_month_ratio),
            "max_positive_month_profit_contribution": _safe_metric(max_month_contribution),
            "bootstrap_probability_mean_positive": _safe_metric(bootstrap),
            "forecast_mae": _safe_metric(filled["forecast_error"].abs().mean()),
            "forecast_rmse": _safe_metric(math.sqrt(float((filled["forecast_error"] ** 2).mean())) if len(filled) else np.nan),
            "direction_accuracy": _safe_metric(pd.to_numeric(filled["direction_success"], errors="coerce").mean()),
            "prediction_interval_coverage": _safe_metric(pd.to_numeric(filled["within_prediction_interval"], errors="coerce").mean()),
            "return_rank_ic": _safe_metric(rank_ic),
            "return_calibration": calibration,
        }
        failures: list[str] = []
        checks = {
            "history_dates": history_dates >= self.config.promotion_min_dates,
            "oos_dates": len(dates) >= self.config.promotion_min_oos_dates,
            "filled_trades": len(filled) >= 80,
            "mean_daily_return": math.isfinite(mean_daily) and mean_daily > 0.0,
            "stress_2x_cost": len(stress_daily) > 0 and float(stress_daily.mean()) > 0.0,
            "bootstrap_95": math.isfinite(bootstrap) and bootstrap >= 0.95,
            "rolling_20d_60pct": len(rolling) > 0 and float((rolling > 0).mean()) >= 0.60,
            "positive_months_60pct": len(monthly) >= 6 and positive_month_ratio >= 0.60,
            "month_concentration": math.isfinite(max_month_contribution) and max_month_contribution <= 0.50,
            "beats_source_topn": len(benchmark) > 0 and mean_daily > float(benchmark.mean()),
            "price_cap_not_worse": len(uncapped) > 0 and mean_daily >= float(uncapped.mean()),
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        metrics["promotion_checks"] = checks
        metrics["promotion_failures"] = failures
        metrics["promoted"] = not failures
        metrics["daily_equity"] = [
            {"signal_date": date, "daily_return": _safe_metric(daily.loc[date]), "nav": _safe_metric(nav.loc[date])}
            for date in dates
        ]
        return metrics

    def run_backtest(self, history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        oos = self._walkforward_predictions(history)
        metrics = self._portfolio_metrics(oos, history["signal_date"].nunique() if not history.empty else 0)
        _write_csv(history, self.config.metrics_root / "training_history_latest.csv")
        _write_csv(oos, self.config.metrics_root / "backtest_trades_latest.csv")
        _write_json(metrics, self.config.metrics_root / "backtest_latest.json")
        return oos, metrics

    # ------------------------------------------------------------------
    # Immutable current prediction and matured true-price verification
    # ------------------------------------------------------------------
    def _prediction_dates(self, signal_date: str, candidates: pd.DataFrame) -> tuple[str, str]:
        expected_buy = ""
        if "verify_date" in candidates.columns:
            values = candidates["verify_date"].dropna().map(_normal_date)
            values = values[values.str.len() == 8]
            if not values.empty:
                expected_buy = values.iloc[0]
        expected_buy = expected_buy or _business_day_after(signal_date)
        return expected_buy, _business_day_after(expected_buy)

    def build_prediction(
        self,
        signal_date: str,
        candidates: pd.DataFrame,
        bundle: Optional[ModelBundle],
        backtest_metrics: dict[str, Any],
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        dated_path = self.config.prediction_root / f"pred_{signal_date}.csv"
        if dated_path.exists() and not force:
            frozen = _read_csv(dated_path)
            if not frozen.empty:
                _write_csv(frozen, self.config.prediction_root / "pred_latest.csv")
                return frozen
        base = self._current_base(signal_date, candidates)
        scored = self.score_candidates(base, bundle)
        expected_buy, expected_exit = self._prediction_dates(signal_date, candidates)
        promoted = backtest_metrics.get("promoted") is True
        scored["prediction_id"] = [f"{signal_date}-{code}-{self.config.model_version}" for code in scored["ts_code"]]
        scored["expected_buy_date"] = expected_buy
        scored["expected_exit_date"] = expected_exit
        scored["recommended_max_price"] = [
            _round_price(min(row["estimated_up_limit"], row["d_close"] * (1.0 + row["recommended_max_gap"])))
            if math.isfinite(_finite(row["recommended_max_gap"]))
            else np.nan
            for _, row in scored.iterrows()
        ]
        scored["max_auction_change_pct"] = (100.0 * scored["recommended_max_gap"]).round(2)
        scored["model_version"] = self.config.model_version
        scored["model_ready"] = int(bundle is not None)
        scored["model_promoted"] = int(promoted)
        scored["action"] = np.where(
            scored["selected"].eq(1),
            "BUY" if promoted else "SHADOW_BUY",
            np.where(scored["model_reason"].eq("no_safe_price"), "REJECT", "WATCH"),
        )
        scored["price_action"] = np.where(
            scored["recommended_max_price"].notna(),
            "竞价不高于上限价；超过即放弃",
            "没有安全买入价格，放弃",
        )
        scored["generated_at_utc"] = _utc_now()
        scored["source_snapshot_sha256"] = _hash_frame(candidates)
        scored["feature_contract"] = "D_CLOSE_ONLY_NO_T_AUCTION_LEAKAGE_V1"
        ordered = [
            "prediction_id",
            "signal_date",
            "expected_buy_date",
            "expected_exit_date",
            "ts_code",
            "name",
            "stage",
            "source_rank",
            "d_close",
            "recommended_max_price",
            "max_auction_change_pct",
            "predicted_net_return",
            "predicted_return_lcb",
            "predicted_return_ucb",
            "predicted_fill_probability",
            "predicted_profit_probability",
            "predicted_big_loss_probability",
            "conservative_ev",
            "selected",
            "action",
            "price_action",
            "model_ready",
            "model_promoted",
            "model_reason",
            "model_version",
            "generated_at_utc",
            "source_snapshot_sha256",
            "feature_contract",
        ]
        scored = scored[[name for name in ordered if name in scored.columns]]
        _write_csv(scored, dated_path)
        _write_csv(scored, self.config.prediction_root / "pred_latest.csv")
        return scored

    def _broker_fills(self) -> pd.DataFrame:
        frame = _read_csv(self.config.broker_fills_path)
        if frame.empty:
            return frame
        frame = frame.copy()
        if "ts_code" in frame.columns:
            frame["ts_code"] = frame["ts_code"].map(_normal_code)
        if "signal_date" in frame.columns:
            frame["signal_date"] = frame["signal_date"].map(_normal_date)
        return frame

    def _broker_row(self, broker: pd.DataFrame, signal_date: str, code: str) -> Optional[pd.Series]:
        if broker.empty or not {"signal_date", "ts_code"}.issubset(broker.columns):
            return None
        hit = broker[(broker["signal_date"] == signal_date) & (broker["ts_code"] == code)]
        return hit.iloc[-1] if not hit.empty else None

    def _verify_prediction_file(self, path: Path, dates: Sequence[str], broker: pd.DataFrame) -> pd.DataFrame:
        pred = _read_csv(path)
        if pred.empty:
            return pd.DataFrame()
        signal_date = _normal_date(pred.get("signal_date", pd.Series([""])).iloc[0])
        later = [date for date in dates if date > signal_date]
        records: list[dict[str, Any]] = []
        for _, row in pred.iterrows():
            code = _normal_code(row.get("ts_code"))
            selected = int(_finite(row.get("selected"), 0)) == 1
            base = row.to_dict()
            base.update(
                {
                    "actual_buy_date": "",
                    "actual_buy_price": np.nan,
                    "actual_exit_date": "",
                    "actual_exit_price": np.nan,
                    "actual_gross_return": np.nan,
                    "actual_net_return": np.nan,
                    "truth_source": "pending",
                    "verification_status": "PENDING_BUY",
                    "actual_fill": np.nan,
                    "actual_fill_reason": "truth_pending",
                    "price_guidance_success": np.nan,
                    "direction_success": np.nan,
                    "trade_success": np.nan,
                    "risk_prediction_success": np.nan,
                    "forecast_error": np.nan,
                    "correct_rejection": np.nan,
                    "missed_opportunity": np.nan,
                }
            )
            if not later:
                records.append(base)
                continue
            buy_date = later[0]
            buy_daily = self._row(self.market_table(buy_date, "daily"), code)
            if buy_daily is None:
                base.update({"actual_buy_date": buy_date, "verification_status": "NO_FILL", "actual_fill": 0, "actual_fill_reason": "suspended_or_daily_missing"})
                records.append(base)
                continue
            buy_price = _numeric_from(buy_daily, ("open",))
            max_price = _finite(row.get("recommended_max_price"))
            market_fill, fill_reason = self._market_buyable(buy_date, code)
            cap_accept = math.isfinite(max_price) and buy_price <= max_price + 0.005
            actual_fill = int(selected and cap_accept and market_fill == 1)
            base.update(
                {
                    "actual_buy_date": buy_date,
                    "actual_buy_price": buy_price,
                    "actual_fill": actual_fill,
                    "actual_fill_reason": "filled_market_proxy" if actual_fill else ("price_above_cap" if not cap_accept else fill_reason),
                    "price_guidance_success": int((actual_fill == 1 and cap_accept) or (actual_fill == 0 and not cap_accept)),
                    "truth_source": "market_proxy",
                    "verification_status": "PENDING_EXIT" if actual_fill else "NO_FILL",
                }
            )
            if len(later) < 2:
                records.append(base)
                continue
            date_index = dates.index(later[1])
            exit_date, exit_price, _, exit_reason = self._resolve_exit(code, date_index, dates)
            if not exit_date:
                records.append(base)
                continue
            gross = self._realized_gross_return(code, buy_date, buy_price, exit_date, dates)
            net = gross - self.config.cost_rate if math.isfinite(gross) else np.nan
            broker_row = self._broker_row(broker, signal_date, code)
            if broker_row is not None:
                broker_buy = _numeric_from(broker_row, ("buy_price", "actual_buy_price"))
                broker_sell = _numeric_from(broker_row, ("sell_price", "actual_exit_price"))
                broker_fees = _numeric_from(broker_row, ("fees", "total_fees"), 0.0)
                quantity = _numeric_from(broker_row, ("quantity", "qty"), 0.0)
                if broker_buy > 0 and broker_sell > 0:
                    gross = broker_sell / broker_buy - 1.0
                    fee_rate = broker_fees / (broker_buy * quantity) if quantity > 0 else self.config.cost_rate
                    net = gross - fee_rate
                    buy_price, exit_price = broker_buy, broker_sell
                    base["truth_source"] = "broker_actual"
                    base["actual_fill"] = 1
                    base["actual_fill_reason"] = "broker_fill"
            predicted = _finite(row.get("predicted_net_return"))
            predicted_loss = _finite(row.get("predicted_big_loss_probability"))
            base.update(
                {
                    "actual_buy_price": buy_price,
                    "actual_exit_date": exit_date,
                    "actual_exit_price": exit_price,
                    "actual_gross_return": gross,
                    "actual_net_return": net,
                    "exit_reason": exit_reason,
                    "verification_status": "VERIFIED" if int(base["actual_fill"]) == 1 else "COUNTERFACTUAL_READY",
                    "direction_success": int((predicted > 0) == (net > 0)) if math.isfinite(predicted) else np.nan,
                    "trade_success": int(net > 0) if int(base["actual_fill"]) == 1 else np.nan,
                    "risk_prediction_success": int((predicted_loss >= self.config.max_big_loss_probability) == (net <= self.config.big_loss_threshold)) if math.isfinite(predicted_loss) else np.nan,
                    "forecast_error": net - predicted if math.isfinite(predicted) else np.nan,
                    "correct_rejection": int(net <= 0) if selected is False else np.nan,
                    "missed_opportunity": int(net > 0) if selected is False else np.nan,
                }
            )
            records.append(base)
        return pd.DataFrame(records)

    def settle_predictions(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        dates = self.market_dates()
        broker = self._broker_fills()
        parts: list[pd.DataFrame] = []
        for path in sorted(self.config.prediction_root.glob("pred_20*.csv")):
            verified = self._verify_prediction_file(path, dates, broker)
            if verified.empty:
                continue
            signal_date = _normal_date(verified["signal_date"].iloc[0])
            _write_csv(verified, self.config.verification_root / f"verify_{signal_date}.csv")
            parts.append(verified)
        ledger = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _write_csv(ledger, self.config.verification_root / "verify_latest.csv")
        metrics = self._verification_metrics(ledger)
        _write_json(metrics, self.config.metrics_root / "cumulative_latest.json")
        return ledger, metrics

    def _verification_metrics(self, ledger: pd.DataFrame) -> dict[str, Any]:
        if ledger.empty:
            return {"status": "no_frozen_predictions", "generated_at_utc": _utc_now(), "verified_trades": 0}
        selected = ledger[pd.to_numeric(ledger["selected"], errors="coerce").fillna(0).eq(1)]
        verified = selected[selected["verification_status"].eq("VERIFIED")].copy()
        returns = pd.to_numeric(verified["actual_net_return"], errors="coerce").dropna()
        positive = returns[returns > 0].sum()
        negative = -returns[returns < 0].sum()
        payload: dict[str, Any] = {
            "status": "ok",
            "generated_at_utc": _utc_now(),
            "frozen_predictions": int(len(ledger)),
            "selected_predictions": int(len(selected)),
            "verified_trades": int(len(verified)),
            "pending_or_no_fill": int(len(selected) - len(verified)),
            "broker_actual_trades": int((verified.get("truth_source", pd.Series(dtype=str)) == "broker_actual").sum()),
            "market_proxy_trades": int((verified.get("truth_source", pd.Series(dtype=str)) == "market_proxy").sum()),
            "fill_rate": _safe_metric(pd.to_numeric(selected.get("actual_fill"), errors="coerce").mean()),
            "price_guidance_accuracy": _safe_metric(pd.to_numeric(selected.get("price_guidance_success"), errors="coerce").mean()),
            "win_rate": _safe_metric((returns > 0).mean()),
            "mean_actual_net_return": _safe_metric(returns.mean()),
            "cumulative_trade_return": _safe_metric((1.0 + returns).prod() - 1.0 if len(returns) else np.nan),
            "profit_factor": _safe_metric(positive / negative if negative > 0 else np.nan),
            "forecast_mae": _safe_metric(pd.to_numeric(verified.get("forecast_error"), errors="coerce").abs().mean()),
            "direction_accuracy": _safe_metric(pd.to_numeric(verified.get("direction_success"), errors="coerce").mean()),
            "risk_prediction_accuracy": _safe_metric(pd.to_numeric(verified.get("risk_prediction_success"), errors="coerce").mean()),
            "correct_rejection_rate": _safe_metric(pd.to_numeric(ledger.get("correct_rejection"), errors="coerce").mean()),
            "missed_opportunity_rate": _safe_metric(pd.to_numeric(ledger.get("missed_opportunity"), errors="coerce").mean()),
        }
        windows: dict[str, Any] = {}
        if "signal_date" in verified.columns:
            verified = verified.sort_values("signal_date")
            for size in (20, 60, 120):
                sample = pd.to_numeric(verified.tail(size)["actual_net_return"], errors="coerce").dropna()
                windows[str(size)] = {
                    "trades": int(len(sample)),
                    "mean_net_return": _safe_metric(sample.mean()),
                    "win_rate": _safe_metric((sample > 0).mean()),
                }
        payload["rolling_trade_windows"] = windows
        return payload

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------
    def run(self, signal_date: str = "", *, force_prediction: bool = False) -> RunResult:
        from .reporting import write_reports

        snapshots = self.candidate_snapshots()
        signal_date = _normal_date(signal_date) or (sorted(snapshots)[-1] if snapshots else "")
        if not signal_date or signal_date not in snapshots:
            raise RuntimeError("cannot resolve a dated pred_source candidate snapshot")
        candidates = self.load_candidates(signal_date, snapshots[signal_date])
        if candidates.empty:
            raise RuntimeError(f"candidate snapshot is empty for {signal_date}")

        history = self.build_history()
        oos, backtest_metrics = self.run_backtest(history)
        bundle = self.fit_models(history)
        prediction = self.build_prediction(
            signal_date,
            candidates,
            bundle,
            backtest_metrics,
            force=force_prediction,
        )
        ledger, cumulative_metrics = self.settle_predictions()
        report_paths = write_reports(
            self.config,
            prediction=prediction,
            ledger=ledger,
            backtest_trades=oos,
            backtest_metrics=backtest_metrics,
            cumulative_metrics=cumulative_metrics,
        )
        model_meta = {
            "generated_at_utc": _utc_now(),
            "model_version": self.config.model_version,
            "ready": bundle is not None,
            "promoted": backtest_metrics.get("promoted") is True,
            "training_rows": bundle.train_rows if bundle else 0,
            "training_dates": bundle.train_dates if bundle else 0,
            "residual_q10": _safe_metric(bundle.residual_q10 if bundle else np.nan),
            "residual_q90": _safe_metric(bundle.residual_q90 if bundle else np.nan),
            "gap_support": {
                "min": _safe_metric(bundle.gap_min if bundle else np.nan),
                "max": _safe_metric(bundle.gap_max if bundle else np.nan),
            },
            "backtest_gate": backtest_metrics,
            "contract": {
                "signal": "D close",
                "entry": "T opening auction with frozen maximum limit price",
                "exit": "T+1 opening auction; delayed to first tradable open after one-price limit-down",
                "cost_rate": self.config.cost_rate,
                "future_features_forbidden": True,
            },
        }
        _write_json(model_meta, self.config.model_root / "model_meta_latest.json")
        warnings: list[str] = []
        if bundle is None:
            warnings.append("模型独立交易日或样本不足，仅生成审计结果，不给出正式买入指令")
        if backtest_metrics.get("promoted") is not True:
            warnings.append("样本外回测未达到正式晋级门槛，当前名单为影子验证")
        return RunResult(
            signal_date=signal_date,
            prediction_path=str(self.config.prediction_root / f"pred_{signal_date}.csv"),
            backtest_path=str(self.config.metrics_root / "backtest_latest.json"),
            verification_path=str(self.config.verification_root / "verify_latest.csv"),
            current_report_path=report_paths["current"],
            verification_report_path=report_paths["verification"],
            dashboard_path=report_paths["dashboard"],
            model_ready=bundle is not None,
            promoted=backtest_metrics.get("promoted") is True,
            selected_count=int(pd.to_numeric(prediction.get("selected"), errors="coerce").fillna(0).sum()),
            warnings=warnings,
        )
