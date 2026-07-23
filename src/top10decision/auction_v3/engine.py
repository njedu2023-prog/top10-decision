from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from top10decision.data.tushare_minute import opening_auction_price_from_snapshot
from top10decision.decision.eligibility import filter_standard_limit_universe
from top10decision.decision.exit_policy import simulate_tplus1_exit
from top10decision.decision.observation import (
    observation_price_contract,
    rank_observation_rows,
)
from top10decision.writers.io_contract import (
    choose_exec_date,
    choose_exit_date,
    is_a_share_trading_day,
    next_a_share_trading_day,
)

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
    "limit_open_times",
    "limit_first_time_minutes",
    "limit_last_time_minutes",
    "limit_fd_amount_log",
    "limit_seal_to_amount",
    "limit_seal_to_float_mv",
    "reseal_score",
    "late_withdraw",
    "d_return",
    "d_range",
    "d_turnover_proxy",
    "d_amount_log",
    "d_amount_percentile",
    "d_turnover_rate",
    "d_volume_ratio",
    "d_float_mv_log",
    "is_hot_board",
    "board_rank",
    "board_limit_up_count",
    "limit_ratio",
    "market_median_return",
    "market_up_ratio",
    "market_return_dispersion",
    "relative_d_return",
    "minute_available",
    "minute_realized_vol",
    "minute_first_30m_return",
    "minute_last_30m_return",
    "minute_vwap_deviation",
    "minute_opening_volume_share",
    "minute_closing_volume_share",
    "minute_close_location",
    "proposed_gap",
]
CONTINUATION_FEATURES = [name for name in MODEL_FEATURES if name != "proposed_gap"]

INDUSTRY_ALIASES = ("industry", "industry_tag", "行业", "行业板块", "board")


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
    continuation_model: Optional[Pipeline]
    fill_model: Optional[Pipeline]
    exit_model: Optional[Pipeline]
    profit_constant: float
    loss_constant: float
    continuation_constant: float
    fill_constant: float
    exit_constant: float
    calibration_bias: float
    expected_return_margin: float
    residual_q10: float
    residual_q90: float
    gap_min: float
    gap_max: float
    train_rows: int
    train_dates: int
    calibration_rows: int
    calibration_dates: int
    return_selection: dict[str, Any]
    classifier_selection: dict[str, dict[str, Any]]


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


def _text_from(row: pd.Series, aliases: Sequence[str], default: str = "") -> str:
    for name in aliases:
        if name not in row.index:
            continue
        value = row.get(name)
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value or "").strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return default


def _pre_close(row: Optional[pd.Series]) -> float:
    if row is None:
        return float("nan")
    direct = _numeric_from(row, ("pre_close", "pre_close_est"))
    if direct > 0:
        return direct
    close_price = _numeric_from(row, ("close",))
    pct_chg = _numeric_from(row, ("pct_chg",))
    denominator = 1.0 + pct_chg / 100.0
    if close_price > 0 and math.isfinite(pct_chg) and denominator > 0:
        return close_price / denominator
    return float("nan")


def _is_close(left: Any, right: Any, tolerance: float = 0.0025) -> bool:
    a, b = _finite(left), _finite(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(0.01, abs(b) * tolerance)


def _round_price(value: float) -> float:
    return round(max(0.01, value) + 1e-9, 2)


def _time_to_minutes(value: Any) -> float:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not text:
        return float("nan")
    text = text[-6:].zfill(6)
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:6])
    if hour > 23 or minute > 59 or second > 59:
        return float("nan")
    return hour * 60.0 + minute + second / 60.0


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _probability(
    model: Optional[Pipeline],
    frame: pd.DataFrame,
    constant: float,
    features: Sequence[str] = MODEL_FEATURES,
) -> np.ndarray:
    if model is None:
        return np.repeat(float(np.clip(constant, 0.0, 1.0)), len(frame))
    return np.clip(model.predict_proba(frame[list(features)])[:, 1], 0.0, 1.0)


def _date_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or "signal_date" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    date_key = frame["signal_date"].astype(str)
    counts = date_key.groupby(date_key).transform("count").clip(lower=1)
    weights = 1.0 / counts.astype(float)
    return (weights / weights.mean()).to_numpy(dtype=float)


class AuctionV3Engine:
    """Builds immutable predictions, walk-forward evidence, and matured truth ledgers."""

    def __init__(self, config: AuctionV3Config):
        self.config = config
        self.config.ensure_directories()
        self._market_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._minute_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        self._eligibility_audit: dict[str, Any] = {}

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
        # Raw folders can exist for a holiday because upstream sync jobs run on
        # weekdays. They are not evidence of an exchange session.
        return sorted(date for date in dates if is_a_share_trading_day(date))

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

    def _minute_path(self, trade_date: str, code: str) -> Path:
        safe_code = _normal_code(code).replace(".", "_")
        return self.config.root / "data" / "market" / "minute_1m" / trade_date[:4] / trade_date / f"{safe_code}.csv"

    def minute_table(self, trade_date: str, code: str) -> pd.DataFrame:
        key = (trade_date, _normal_code(code))
        if key in self._minute_cache:
            return self._minute_cache[key]
        frame = _read_csv(self._minute_path(trade_date, code))
        if not frame.empty:
            frame = frame.copy()
            for column in ("open", "close", "high", "low", "vol", "amount"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if "time" in frame.columns:
                frame["time"] = frame["time"].astype(str).str.strip()
                frame = frame.sort_values("time").drop_duplicates("time", keep="last")
        self._minute_cache[key] = frame
        return frame

    def _market_context(self, trade_date: str) -> dict[str, Any]:
        if trade_date in self._context_cache:
            return self._context_cache[trade_date]
        daily = self.market_table(trade_date, "daily")
        if daily.empty:
            context = {
                "market_median_return": np.nan,
                "market_up_ratio": np.nan,
                "market_return_dispersion": np.nan,
                "amount_percentile": {},
            }
            self._context_cache[trade_date] = context
            return context

        pct = pd.to_numeric(daily["pct_chg"], errors="coerce") / 100.0 if "pct_chg" in daily.columns else pd.Series(np.nan, index=daily.index)
        if pct.notna().sum() < max(20, len(daily) // 2):
            close = pd.to_numeric(daily["close"], errors="coerce") if "close" in daily.columns else pd.Series(np.nan, index=daily.index)
            if "pre_close" in daily.columns:
                pre_close = pd.to_numeric(daily["pre_close"], errors="coerce")
            elif "pre_close_est" in daily.columns:
                pre_close = pd.to_numeric(daily["pre_close_est"], errors="coerce")
            else:
                pre_close = pd.Series(np.nan, index=daily.index)
            derived = close / pre_close.replace(0.0, np.nan) - 1.0
            pct = pct.where(pct.notna(), derived)
        valid = pct.replace([np.inf, -np.inf], np.nan).dropna()
        amount = pd.to_numeric(daily.get("amount"), errors="coerce") if "amount" in daily.columns else pd.Series(np.nan, index=daily.index)
        context = {
            "market_median_return": float(valid.median()) if len(valid) else np.nan,
            "market_up_ratio": float((valid > 0).mean()) if len(valid) else np.nan,
            "market_return_dispersion": float(valid.std(ddof=0)) if len(valid) else np.nan,
            "amount_percentile": amount.rank(pct=True).to_dict(),
        }
        self._context_cache[trade_date] = context
        return context

    def _minute_features(self, trade_date: str, code: str) -> dict[str, float]:
        defaults = {
            "minute_available": 0.0,
            "minute_realized_vol": np.nan,
            "minute_first_30m_return": np.nan,
            "minute_last_30m_return": np.nan,
            "minute_vwap_deviation": np.nan,
            "minute_opening_volume_share": np.nan,
            "minute_closing_volume_share": np.nan,
            "minute_close_location": np.nan,
        }
        frame = self.minute_table(trade_date, code)
        needed = {"time", "open", "close", "high", "low", "vol"}
        if frame.empty or not needed.issubset(frame.columns):
            return defaults
        clean = frame.dropna(subset=["open", "close", "high", "low"]).copy()
        if len(clean) < 5:
            return defaults

        time_text = clean["time"].astype(str)
        hhmm = pd.to_numeric(time_text.str.extract(r"(\d{2}):(\d{2})", expand=True).fillna("0").agg("".join, axis=1), errors="coerce")
        close = pd.to_numeric(clean["close"], errors="coerce")
        open_price = pd.to_numeric(clean["open"], errors="coerce")
        high = pd.to_numeric(clean["high"], errors="coerce")
        low = pd.to_numeric(clean["low"], errors="coerce")
        vol = pd.to_numeric(clean["vol"], errors="coerce").fillna(0.0).clip(lower=0.0)
        returns = np.log(close.replace(0.0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
        first = clean[hhmm.le(1000)]
        last = clean[hhmm.ge(1430)]
        total_vol = float(vol.sum())
        vwap = float((close * vol).sum() / total_vol) if total_vol > 0 else np.nan
        price_range = float(high.max() - low.min())

        defaults.update(
            {
                "minute_available": 1.0,
                "minute_realized_vol": float(returns.std(ddof=0)) if len(returns) else np.nan,
                "minute_first_30m_return": float(first["close"].iloc[-1] / first["open"].iloc[0] - 1.0) if len(first) else np.nan,
                "minute_last_30m_return": float(last["close"].iloc[-1] / last["open"].iloc[0] - 1.0) if len(last) else np.nan,
                "minute_vwap_deviation": float(close.iloc[-1] / vwap - 1.0) if vwap > 0 else np.nan,
                "minute_opening_volume_share": float(vol.loc[first.index].sum() / total_vol) if total_vol > 0 and len(first) else np.nan,
                "minute_closing_volume_share": float(vol.loc[last.index].sum() / total_vol) if total_vol > 0 and len(last) else np.nan,
                "minute_close_location": float((close.iloc[-1] - low.min()) / price_range) if price_range > 0 else 0.5,
            }
        )
        return defaults

    def _execution_open_price(self, trade_date: str, code: str, daily_row: Optional[pd.Series] = None) -> float:
        auction = self._auction_price(trade_date, code)
        if auction > 0:
            return auction
        minute = self.minute_table(trade_date, code)
        minute_open = opening_auction_price_from_snapshot(minute)
        if minute_open is not None and minute_open > 0:
            return float(minute_open)
        row = daily_row if daily_row is not None else self._row(self.market_table(trade_date, "daily"), code)
        return _numeric_from(row, ("open",)) if row is not None else float("nan")

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
        source = _read_csv(path) if path else pd.DataFrame()
        if not source.empty:
            source = source.copy()
            source_code_col = next(
                (c for c in ("ts_code", "code", "代码") if c in source.columns),
                "",
            )
            if source_code_col:
                source["ts_code"] = source[source_code_col].map(_normal_code)
                source = source[source["ts_code"] != ""].drop_duplicates("ts_code", keep="first")

        authoritative = self.market_table(signal_date, "limit_list_d")
        if not authoritative.empty:
            frame = authoritative.reset_index(drop=True).copy()
            if "limit_type" in frame.columns:
                limit_type = frame["limit_type"].astype(str).str.upper().str.strip()
                frame = frame[limit_type.isin({"U", "UP", "涨停"}) | limit_type.eq("")]
            frame["ts_code"] = frame["ts_code"].map(_normal_code)
            frame = frame[frame["ts_code"] != ""].drop_duplicates("ts_code", keep="first")
            stock_basic = self.market_table(signal_date, "stock_basic")
            if not stock_basic.empty:
                enrich_columns = [name for name in ("name", "industry", "list_date") if name in stock_basic.columns]
                if enrich_columns:
                    frame = frame.set_index("ts_code", drop=False)
                    for name in enrich_columns:
                        values = stock_basic[name]
                        frame[name] = frame[name].where(frame[name].notna(), values) if name in frame.columns else values
                    frame = frame.reset_index(drop=True)
            if not source.empty and "ts_code" in source.columns:
                frame = frame.set_index("ts_code", drop=False)
                source = source.set_index("ts_code", drop=False)
                for name in source.columns:
                    if name == "ts_code":
                        continue
                    values = source[name]
                    if name in frame.columns:
                        frame[name] = values.combine_first(frame[name])
                    else:
                        frame[name] = values
                frame = frame.reset_index(drop=True)
        else:
            frame = source.copy()

        if frame.empty:
            return frame
        code_col = next((c for c in ("ts_code", "code", "代码") if c in frame.columns), "")
        if not code_col:
            return pd.DataFrame()
        frame["ts_code"] = frame[code_col].map(_normal_code)
        frame = frame[frame["ts_code"] != ""].drop_duplicates("ts_code", keep="first")
        frame, self._eligibility_audit = filter_standard_limit_universe(frame, code_col="ts_code", name_col="name")
        rank_col = next((c for c in ("rank", "rank_v2", "排名") if c in frame.columns), "")
        frame["_source_rank"] = pd.to_numeric(frame[rank_col], errors="coerce") if rank_col else np.nan
        missing_rank = frame["_source_rank"].isna()
        rank_start = int(frame["_source_rank"].max()) if frame["_source_rank"].notna().any() else 0
        frame.loc[missing_rank, "_source_rank"] = np.arange(
            rank_start + 1,
            rank_start + 1 + int(missing_rank.sum()),
        )
        frame = frame.sort_values("_source_rank", na_position="last")
        if self.config.max_candidates > 0:
            frame = frame.head(self.config.max_candidates)
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
            pre_close = _pre_close(daily_row)
            up_limit = _numeric_from(limit_row, ("up_limit",))
            if pre_close > 0 and up_limit > 0:
                ratio = up_limit / pre_close - 1.0
                if 0.03 <= ratio <= 0.35:
                    return ratio
        return 0.10

    def _consecutive_limit_up_count(
        self,
        signal_date: str,
        code: str,
        dates: Optional[Sequence[str]] = None,
    ) -> int:
        """Count consecutive close-at-limit sessions ending on D without using future data."""
        trading_dates = list(dates or self.market_dates())
        try:
            index = trading_dates.index(signal_date)
        except ValueError:
            return 0
        count = 0
        newer_date = ""
        for position in range(index, -1, -1):
            trade_date = trading_dates[position]
            if newer_date and next_a_share_trading_day(trade_date) != newer_date:
                break
            daily = self._row(self.market_table(trade_date, "daily"), code)
            limit = self._row(self.market_table(trade_date, "stk_limit"), code)
            up_limit = _numeric_from(limit, ("up_limit",)) if limit is not None else float("nan")
            if daily is None or not math.isfinite(up_limit) or not _is_close(daily.get("close"), up_limit):
                break
            count += 1
            newer_date = trade_date
        return count

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
        execution_price = self._execution_open_price(trade_date, code, daily)
        if math.isfinite(execution_price) and open_price > 0 and abs(execution_price - open_price) > 0.011:
            return 0, "auction_daily_open_conflict"
        up_limit = _numeric_from(limit, ("up_limit",)) if limit is not None else float("nan")
        if math.isfinite(execution_price) and math.isfinite(up_limit) and abs(execution_price - up_limit) <= 0.011:
            return 0, "opening_auction_limit_up_unconfirmed"
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
        *,
        exit_price: float = float("nan"),
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
            pre_close = _pre_close(row)
            if pre_close <= 0:
                return float("nan")
            if idx == end:
                realized_exit = exit_price if math.isfinite(exit_price) and exit_price > 0 else self._execution_open_price(dates[idx], code, row)
                if realized_exit <= 0:
                    return float("nan")
                wealth *= realized_exit / pre_close
            else:
                close_price = _numeric_from(row, ("close",))
                if close_price <= 0:
                    return float("nan")
                wealth *= close_price / pre_close
        return wealth - 1.0

    def _resolve_exit(
        self,
        code: str,
        start_index: int,
        dates: Sequence[str],
        *,
        entry_price: float = float("nan"),
        buy_date: str = "",
    ) -> tuple[str, float, int, str]:
        buy_daily = self._row(self.market_table(buy_date, "daily"), code) if buy_date else None
        buy_close = _numeric_from(buy_daily, ("close",)) if buy_daily is not None else float("nan")
        for offset, trade_date in enumerate(dates[start_index:], start=0):
            daily = self._row(self.market_table(trade_date, "daily"), code)
            if daily is None:
                continue
            limit = self._row(self.market_table(trade_date, "stk_limit"), code)
            if self._one_price_limit(daily, limit, "down"):
                continue
            if offset == 0 and entry_price > 0:
                timed = simulate_tplus1_exit(
                    entry_price=entry_price,
                    buy_close=buy_close,
                    target_pre_close=_pre_close(daily),
                    open_price=daily.get("open"),
                    high_price=daily.get("high"),
                    low_price=daily.get("low"),
                    close_price=daily.get("close"),
                    down_limit=limit.get("down_limit") if limit is not None else None,
                    minute_frame=self.minute_table(trade_date, code),
                    take_profit_pct=self.config.take_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    latest_exit_time=self.config.latest_exit_time,
                )
                if timed.executable and timed.exit_price is not None and timed.exit_price > 0:
                    return trade_date, float(timed.exit_price), 0, timed.reason
                continue
            exit_price = self._execution_open_price(trade_date, code, daily)
            if exit_price > 0:
                return trade_date, exit_price, offset, "delayed_first_tradable_open"
        return "", float("nan"), -1, "exit_truth_pending"

    def _feature_dict(
        self,
        candidate: pd.Series,
        d_daily: Optional[pd.Series],
        limit_ratio: float,
        market_context: Optional[dict[str, Any]] = None,
        signal_date: str = "",
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for canonical, aliases in FEATURE_ALIASES.items():
            out[canonical] = _numeric_from(candidate, aliases)
        d_return = np.nan
        if d_daily is None:
            out.update(
                {
                    "d_return": np.nan,
                    "d_range": np.nan,
                    "d_turnover_proxy": np.nan,
                    "d_amount_log": np.nan,
                }
            )
        else:
            open_price = _numeric_from(d_daily, ("open",))
            close_price = _numeric_from(d_daily, ("close",))
            high = _numeric_from(d_daily, ("high",))
            low = _numeric_from(d_daily, ("low",))
            pre_close = _pre_close(d_daily)
            amount = _numeric_from(d_daily, ("amount",))
            d_return = close_price / pre_close - 1.0 if pre_close > 0 and close_price > 0 else _numeric_from(d_daily, ("pct_chg",)) / 100.0
            out["d_return"] = d_return
            out["d_range"] = (high - low) / pre_close if pre_close > 0 and high > 0 and low > 0 else np.nan
            out["d_turnover_proxy"] = abs(close_price - open_price) / pre_close if pre_close > 0 and open_price > 0 else np.nan
            out["d_amount_log"] = math.log1p(amount) if amount > 0 else np.nan
        context = market_context or {}
        market_median = _finite(context.get("market_median_return"))
        code = _normal_code(candidate.get("ts_code"))
        limit_detail = self._row(self.market_table(signal_date, "limit_list_d"), code) if signal_date and code else None
        daily_basic = self._row(self.market_table(signal_date, "daily_basic"), code) if signal_date and code else None
        limit_tag = self._row(self.market_table(signal_date, "limit_up_tags"), code) if signal_date and code else None
        daily_amount_yuan = (
            _numeric_from(d_daily, ("amount",)) * 1_000.0 if d_daily is not None else float("nan")
        )
        detail_amount = _numeric_from(limit_detail, ("amount",)) if limit_detail is not None else float("nan")
        amount_yuan = detail_amount if detail_amount > 0 else daily_amount_yuan
        fd_amount = _numeric_from(limit_detail, ("fd_amount",)) if limit_detail is not None else float("nan")
        seal_amount = _numeric_from(limit_detail, ("seal_amount", "fd_amount")) if limit_detail is not None else float("nan")
        detail_float_mv = _numeric_from(limit_detail, ("float_mv",)) if limit_detail is not None else float("nan")
        basic_float_mv = _numeric_from(daily_basic, ("float_mv",)) if daily_basic is not None else float("nan")
        float_mv_yuan = detail_float_mv if detail_float_mv > 0 else basic_float_mv * 10_000.0
        out["limit_open_times"] = _numeric_from(limit_detail, ("open_times",)) if limit_detail is not None else np.nan
        out["limit_first_time_minutes"] = _time_to_minutes(limit_detail.get("first_time")) if limit_detail is not None else np.nan
        out["limit_last_time_minutes"] = _time_to_minutes(limit_detail.get("last_time")) if limit_detail is not None else np.nan
        out["limit_fd_amount_log"] = math.log1p(fd_amount) if fd_amount > 0 else np.nan
        out["limit_seal_to_amount"] = seal_amount / amount_yuan if seal_amount > 0 and amount_yuan > 0 else np.nan
        out["limit_seal_to_float_mv"] = seal_amount / float_mv_yuan if seal_amount > 0 and float_mv_yuan > 0 else np.nan
        turnover_rate = _numeric_from(daily_basic, ("turnover_rate",)) if daily_basic is not None else float("nan")
        out["d_turnover_rate"] = turnover_rate / 100.0 if math.isfinite(turnover_rate) else np.nan
        out["d_volume_ratio"] = _numeric_from(daily_basic, ("volume_ratio",)) if daily_basic is not None else np.nan
        out["d_float_mv_log"] = math.log1p(float_mv_yuan) if float_mv_yuan > 0 else np.nan
        out["is_hot_board"] = _numeric_from(limit_tag, ("is_hot_board",)) if limit_tag is not None else np.nan
        out["board_rank"] = _numeric_from(limit_tag, ("board_rank",)) if limit_tag is not None else np.nan
        out["board_limit_up_count"] = _numeric_from(limit_tag, ("board_limit_up_count",)) if limit_tag is not None else np.nan
        out["d_amount_percentile"] = _finite((context.get("amount_percentile") or {}).get(code))
        out["market_median_return"] = market_median
        out["market_up_ratio"] = _finite(context.get("market_up_ratio"))
        out["market_return_dispersion"] = _finite(context.get("market_return_dispersion"))
        out["relative_d_return"] = d_return - market_median if math.isfinite(d_return) and math.isfinite(market_median) else np.nan
        out.update(self._minute_features(signal_date, code) if signal_date and code else self._minute_features("", ""))
        out["limit_ratio"] = limit_ratio
        out["proposed_gap"] = np.nan
        return out

    def build_history(self) -> pd.DataFrame:
        dates = self.market_dates()
        date_index = {date: idx for idx, date in enumerate(dates)}
        snapshots = self.candidate_snapshots()
        records: list[dict[str, Any]] = []
        for signal_date in dates:
            idx = date_index.get(signal_date)
            if idx is None or idx + 2 >= len(dates):
                continue
            buy_date, target_exit_date = dates[idx + 1], dates[idx + 2]
            candidates = self.load_candidates(signal_date, snapshots.get(signal_date))
            if candidates.empty:
                continue
            d_daily_table = self.market_table(signal_date, "daily")
            d_limit_table = self.market_table(signal_date, "stk_limit")
            buy_daily_table = self.market_table(buy_date, "daily")
            buy_limit_table = self.market_table(buy_date, "stk_limit")
            market_context = self._market_context(signal_date)
            for _, candidate in candidates.iterrows():
                code = candidate["ts_code"]
                d_daily = self._row(d_daily_table, code)
                buy_daily = self._row(buy_daily_table, code)
                if d_daily is None or buy_daily is None:
                    continue
                d_close = _numeric_from(d_daily, ("close",))
                buy_open = self._execution_open_price(buy_date, code, buy_daily)
                if d_close <= 0 or buy_open <= 0:
                    continue
                d_limit = self._row(d_limit_table, code)
                limit_ratio = self._limit_ratio(d_daily, d_limit)
                mechanism_limit_pct = _finite(
                    candidate.get("decision_limit_pct"),
                    limit_ratio * 100.0,
                )
                if mechanism_limit_pct > self.config.max_mechanism_limit_pct + EPS:
                    continue
                d_up_limit = _numeric_from(d_limit, ("up_limit",)) if d_limit is not None else float("nan")
                if not math.isfinite(d_up_limit) or not _is_close(d_daily.get("close"), d_up_limit):
                    continue
                buy_limit = self._row(buy_limit_table, code)
                buy_up_limit = _numeric_from(buy_limit, ("up_limit",)) if buy_limit is not None else float("nan")
                continuation_hit = int(math.isfinite(buy_up_limit) and _is_close(buy_daily.get("close"), buy_up_limit))
                market_fill, fill_reason = self._market_buyable(buy_date, code)
                exit_date, exit_price, delay_days, exit_reason = self._resolve_exit(
                    code,
                    idx + 2,
                    dates,
                    entry_price=buy_open,
                    buy_date=buy_date,
                )
                if not exit_date or exit_price <= 0:
                    continue
                gross_return = self._realized_gross_return(
                    code,
                    buy_date,
                    buy_open,
                    exit_date,
                    dates,
                    exit_price=exit_price,
                )
                if not math.isfinite(gross_return):
                    continue
                net_return = gross_return - self.config.cost_rate
                features = self._feature_dict(
                    candidate,
                    d_daily,
                    limit_ratio,
                    market_context=market_context,
                    signal_date=signal_date,
                )
                consecutive_limit_ups = self._consecutive_limit_up_count(
                    signal_date,
                    code,
                    dates,
                )
                features["limit_times"] = float(consecutive_limit_ups)
                features["proposed_gap"] = buy_open / d_close - 1.0
                industry = _text_from(candidate, INDUSTRY_ALIASES)
                records.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "target_exit_date": target_exit_date,
                        "actual_exit_date": exit_date,
                        "exit_delay_days": delay_days,
                        "ts_code": code,
                        "name": str(candidate.get("name", candidate.get("股票", ""))),
                        "industry": industry,
                        "stage": f"{consecutive_limit_ups}→{consecutive_limit_ups + 1}",
                        "source_rank": features["source_rank"],
                        "d_close": d_close,
                        "buy_open": buy_open,
                        "exit_open": exit_price,
                        "actual_buy_gap": features["proposed_gap"],
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "profit_hit": int(net_return > 0.0),
                        "big_loss_hit": int(net_return <= self.config.big_loss_threshold),
                        "continuation_limit_up_hit": continuation_hit,
                        "exit_on_time": int(delay_days == 0),
                        "market_fill": market_fill,
                        "mechanism_limit_pct": mechanism_limit_pct,
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
    def _regression_pipeline(self, kind: str = "hgb") -> Pipeline:
        if kind == "extra_trees":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=200,
                            min_samples_leaf=20,
                            max_features=0.70,
                            n_jobs=1,
                            random_state=20260716,
                        ),
                    ),
                ]
            )
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

    def _classifier_pipeline(self, kind: str = "hgb") -> Pipeline:
        if kind == "extra_trees":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=200,
                            min_samples_leaf=20,
                            max_features=0.70,
                            n_jobs=1,
                            random_state=20260716,
                        ),
                    ),
                ]
            )
        if kind == "lr":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.20,
                            max_iter=2_000,
                            random_state=20260716,
                        ),
                    ),
                ]
            )
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
        dates = sorted(clean["signal_date"].astype(str).unique())
        date_count = len(dates)
        if len(clean) < self.config.min_train_rows or date_count < self.config.min_train_dates:
            return None

        n_calibration_dates = max(
            self.config.calibration_min_dates,
            int(math.ceil(date_count * self.config.calibration_fraction)),
        )
        n_calibration_dates = min(n_calibration_dates, max(1, date_count // 3))
        split_at = date_count - n_calibration_dates - self.config.calibration_embargo_dates
        if split_at < max(8, self.config.min_train_dates // 2):
            return None
        fit_dates = set(dates[:split_at])
        calibration_dates = set(dates[-n_calibration_dates:])
        fit = clean[clean["signal_date"].astype(str).isin(fit_dates)].copy()
        calibration = clean[clean["signal_date"].astype(str).isin(calibration_dates)].copy()
        if len(fit) < max(100, self.config.min_train_rows // 2) or calibration.empty:
            return None

        regression_candidates: list[tuple[float, float, str, Pipeline, float]] = []
        regression_audit: dict[str, Any] = {}
        for kind in ("hgb", "extra_trees"):
            provisional = self._regression_pipeline(kind)
            provisional.fit(
                fit[MODEL_FEATURES],
                fit["net_return"],
                model__sample_weight=_date_balanced_weights(fit),
            )
            prediction = provisional.predict(calibration[MODEL_FEATURES])
            error = calibration["net_return"].to_numpy(dtype=float) - prediction
            calibration_weights = _date_balanced_weights(calibration)
            rmse = float(math.sqrt(np.average(error**2, weights=calibration_weights)))
            rank_frame = calibration[["signal_date", "net_return"]].copy()
            rank_frame["prediction"] = prediction
            daily_rank_ic = []
            for _, group in rank_frame.groupby("signal_date"):
                if len(group) < 3 or group["prediction"].nunique() < 2 or group["net_return"].nunique() < 2:
                    continue
                daily_rank_ic.append(
                    float(group[["prediction", "net_return"]].corr(method="spearman").iloc[0, 1])
                )
            mean_rank_ic = float(np.nanmean(daily_rank_ic)) if daily_rank_ic else 0.0
            objective = rmse - 0.01 * float(np.clip(mean_rank_ic, -0.5, 0.5))
            regression_candidates.append((objective, rmse, kind, provisional, mean_rank_ic))
            regression_audit[kind] = {
                "calibration_rmse": _safe_metric(rmse),
                "daily_spearman": _safe_metric(mean_rank_ic),
                "selection_objective": _safe_metric(objective),
            }
        _, best_return_rmse, best_return_kind, provisional, best_return_rank_ic = min(
            regression_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        calibration_prediction = provisional.predict(calibration[MODEL_FEATURES])
        raw_residual = calibration["net_return"].to_numpy(dtype=float) - calibration_prediction
        calibration_bias = float(np.clip(np.median(raw_residual), -0.05, 0.05))
        residual = raw_residual - calibration_bias
        residual_by_date = (
            pd.DataFrame(
                {
                    "signal_date": calibration["signal_date"].astype(str).to_numpy(),
                    "residual": residual,
                }
            )
            .groupby("signal_date")["residual"]
            .mean()
        )
        if len(residual_by_date) > 1:
            standard_error = float(
                residual_by_date.std(ddof=1) / math.sqrt(len(residual_by_date))
            )
        elif len(residual) > 1:
            standard_error = float(np.std(residual, ddof=1) / math.sqrt(len(residual)))
        else:
            standard_error = 0.0
        expected_return_margin = float(
            np.clip(
                self.config.expected_return_confidence_z * standard_error,
                self.config.min_expected_return_margin,
                0.05,
            )
        )

        X = clean[MODEL_FEATURES]
        return_model = self._regression_pipeline(best_return_kind)
        return_model.fit(
            X,
            clean["net_return"],
            model__sample_weight=_date_balanced_weights(clean),
        )

        def fit_classifier(
            target: str,
            features: Sequence[str] = MODEL_FEATURES,
        ) -> tuple[Optional[Pipeline], float, dict[str, Any]]:
            values = clean[target].astype(int)
            constant = float(values.mean())
            counts = values.value_counts()
            if len(counts) < 2 or int(counts.min()) < 10:
                return None, constant, {
                    "selected": "constant",
                    "calibration_brier": None,
                    "base_rate": _safe_metric(constant),
                }
            fit_values = fit[target].astype(int)
            calibration_values = calibration[target].astype(int)
            candidates: list[tuple[float, str]] = []
            for kind in ("hgb", "lr", "extra_trees"):
                provisional_classifier = self._classifier_pipeline(kind)
                provisional_classifier.fit(
                    fit[list(features)],
                    fit_values,
                    model__sample_weight=_date_balanced_weights(fit),
                )
                probability = provisional_classifier.predict_proba(
                    calibration[list(features)]
                )[:, 1]
                weights = _date_balanced_weights(calibration)
                brier = float(np.average((probability - calibration_values.to_numpy()) ** 2, weights=weights))
                candidates.append((brier, kind))
            best_brier, best_kind = min(candidates, key=lambda item: (item[0], item[1]))
            model = self._classifier_pipeline(best_kind)
            model.fit(
                clean[list(features)],
                values,
                model__sample_weight=_date_balanced_weights(clean),
            )
            return model, constant, {
                "selected": best_kind,
                "calibration_brier": _safe_metric(best_brier),
                "base_rate": _safe_metric(constant),
                "features": list(features),
            }

        profit_model, profit_constant, profit_selection = fit_classifier("profit_hit")
        loss_model, loss_constant, loss_selection = fit_classifier("big_loss_hit")
        continuation_model, continuation_constant, continuation_selection = fit_classifier(
            "continuation_limit_up_hit",
            CONTINUATION_FEATURES,
        )
        exit_model, exit_constant, exit_selection = fit_classifier("exit_on_time")
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
            fill_model.fit(
                fill_train[MODEL_FEATURES],
                fill_values,
                model__sample_weight=_date_balanced_weights(fill_train),
            )
        return ModelBundle(
            return_model=return_model,
            profit_model=profit_model,
            loss_model=loss_model,
            continuation_model=continuation_model,
            fill_model=fill_model,
            exit_model=exit_model,
            profit_constant=profit_constant,
            loss_constant=loss_constant,
            continuation_constant=continuation_constant,
            fill_constant=fill_constant,
            exit_constant=exit_constant,
            calibration_bias=calibration_bias,
            expected_return_margin=expected_return_margin,
            residual_q10=float(np.quantile(residual, self.config.lower_confidence_quantile)),
            residual_q90=float(np.quantile(residual, self.config.prediction_interval_upper_quantile)),
            gap_min=float(clean["proposed_gap"].quantile(0.01)),
            gap_max=float(clean["proposed_gap"].quantile(0.99)),
            train_rows=len(clean),
            train_dates=date_count,
            calibration_rows=len(calibration),
            calibration_dates=len(calibration_dates),
            return_selection={
                "selected": best_return_kind,
                "calibration_rmse": _safe_metric(best_return_rmse),
                "daily_spearman": _safe_metric(best_return_rank_ic),
                "candidates": regression_audit,
            },
            classifier_selection={
                "profit": profit_selection,
                "big_loss": loss_selection,
                "continuation_limit_up": continuation_selection,
                "exit_on_time": exit_selection,
            },
        )

    def _score_candidate_at_gaps(self, row: pd.Series, bundle: ModelBundle) -> Optional[dict[str, Any]]:
        limit_ratio = _finite(row.get("limit_ratio"), 0.10)
        low = max(self.config.gap_grid_min, bundle.gap_min)
        high = min(self.config.gap_grid_max, bundle.gap_max, limit_ratio)
        if high < low:
            return None
        gaps = np.arange(low, high + self.config.gap_grid_step / 2.0, self.config.gap_grid_step)
        grid = pd.DataFrame([row.to_dict()] * len(gaps))
        grid["proposed_gap"] = gaps
        pred = bundle.return_model.predict(grid[MODEL_FEATURES]) + bundle.calibration_bias
        p_profit = _probability(bundle.profit_model, grid, bundle.profit_constant)
        p_loss = _probability(bundle.loss_model, grid, bundle.loss_constant)
        p_continuation = _probability(
            bundle.continuation_model,
            grid,
            bundle.continuation_constant,
            CONTINUATION_FEATURES,
        )
        p_exit = _probability(bundle.exit_model, grid, bundle.exit_constant)
        # A less aggressive limit price cannot have a higher execution chance.
        p_fill = np.maximum.accumulate(_probability(bundle.fill_model, grid, bundle.fill_constant))
        lower = pred - bundle.expected_return_margin
        upper = pred + bundle.expected_return_margin
        outcome_q10 = pred + bundle.residual_q10
        outcome_q90 = pred + bundle.residual_q90
        risk_adjusted_return = pred - (
            self.config.tail_risk_aversion * p_loss * abs(self.config.big_loss_threshold)
        ) - ((1.0 - p_exit) * self.config.blocked_exit_loss)
        conservative_ev = p_fill * risk_adjusted_return
        limit_times = _finite(row.get("limit_times"), 0.0)
        stage_focus = 1.0 if int(round(limit_times)) in (2, 3) else 0.0
        selection_score = conservative_ev + (
            self.config.continuation_score_weight * stage_focus * p_continuation
        )
        big_loss_ok = p_loss <= self.config.max_big_loss_probability
        lower_bound_ok = lower >= self.config.min_return_lcb
        profit_ok = p_profit >= self.config.min_profit_probability
        fill_ok = p_fill >= self.config.min_fill_probability
        exit_ok = p_exit >= self.config.min_exit_probability
        edge_ok = conservative_ev >= self.config.min_edge
        supported = (
            big_loss_ok
            & lower_bound_ok
            & profit_ok
            & fill_ok
            & exit_ok
            & edge_ok
        )
        if supported.any():
            supported_indices = np.where(supported)[0]
            chosen = int(supported_indices[np.argmax(selection_score[supported_indices])])
            model_reason = "ok"
        else:
            finite = np.where(np.isfinite(conservative_ev))[0]
            if not len(finite):
                return None
            chosen = int(finite[np.argmax(conservative_ev[finite])])
            progressive = big_loss_ok
            if not progressive.any():
                model_reason = "big_loss_probability_exceeds_cap"
            elif not (progressive & lower_bound_ok).any():
                model_reason = "return_lcb_not_positive"
            elif not (progressive & lower_bound_ok & exit_ok).any():
                model_reason = "exit_probability_below_floor"
            elif not (progressive & lower_bound_ok & exit_ok & fill_ok).any():
                model_reason = "fill_probability_below_floor"
            elif not (progressive & lower_bound_ok & exit_ok & fill_ok & profit_ok).any():
                model_reason = "profit_probability_below_floor"
            else:
                model_reason = "conservative_edge_below_floor"
        return {
            "recommended_max_gap": float(gaps[chosen]) if supported[chosen] else np.nan,
            "diagnostic_gap": float(gaps[chosen]),
            "predicted_net_return": float(pred[chosen]),
            "predicted_return_lcb": float(lower[chosen]),
            "predicted_return_ucb": float(upper[chosen]),
            "predicted_outcome_q10": float(outcome_q10[chosen]),
            "predicted_outcome_q90": float(outcome_q90[chosen]),
            "predicted_profit_probability": float(p_profit[chosen]),
            "predicted_big_loss_probability": float(p_loss[chosen]),
            "predicted_continuation_limit_up_probability": float(p_continuation[chosen]),
            "predicted_fill_probability": float(p_fill[chosen]),
            "predicted_exit_probability": float(p_exit[chosen]),
            "conservative_ev": float(conservative_ev[chosen]),
            "selection_score": float(selection_score[chosen]),
            "stage_focus": int(stage_focus),
            "risk_gate_pass": int(supported[chosen]),
            "model_reason": model_reason,
        }

    def score_candidates(self, base: pd.DataFrame, bundle: Optional[ModelBundle]) -> pd.DataFrame:
        out = base.copy().reset_index(drop=True)
        score_columns = [
            "recommended_max_gap",
            "diagnostic_gap",
            "predicted_net_return",
            "predicted_return_lcb",
            "predicted_return_ucb",
            "predicted_outcome_q10",
            "predicted_outcome_q90",
            "predicted_profit_probability",
            "predicted_big_loss_probability",
            "predicted_continuation_limit_up_probability",
            "predicted_fill_probability",
            "predicted_exit_probability",
            "conservative_ev",
            "selection_score",
            "stage_focus",
            "risk_gate_pass",
        ]
        for name in score_columns:
            out[name] = np.nan
        if bundle is None:
            out["model_reason"] = "insufficient_independent_history"
            out["risk_gate_pass"] = 0
            out["selected"] = 0
            return out
        out["model_reason"] = "no_safe_price"
        for index, row in out.iterrows():
            score = self._score_candidate_at_gaps(row, bundle)
            if score is None:
                continue
            for name, value in score.items():
                out.loc[index, name] = value
        out = out.sort_values(
            ["selection_score", "predicted_return_lcb", "source_rank"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        out["selected"] = 0
        eligible = out.index[pd.to_numeric(out["risk_gate_pass"], errors="coerce").fillna(0).eq(1)][
            : self.config.max_positions
        ]
        out.loc[eligible, "selected"] = 1
        return out

    def _current_base(self, signal_date: str, candidates: pd.DataFrame) -> pd.DataFrame:
        dates = self.market_dates()
        d_daily_table = self.market_table(signal_date, "daily")
        d_limit_table = self.market_table(signal_date, "stk_limit")
        market_context = self._market_context(signal_date)
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
            mechanism_limit_pct = _finite(
                candidate.get("decision_limit_pct"),
                limit_ratio * 100.0,
            )
            if mechanism_limit_pct > self.config.max_mechanism_limit_pct + EPS:
                continue
            d_up_limit = _numeric_from(d_limit, ("up_limit",)) if d_limit is not None else float("nan")
            if not math.isfinite(d_up_limit) or not _is_close(d_daily.get("close"), d_up_limit):
                continue
            features = self._feature_dict(
                candidate,
                d_daily,
                limit_ratio,
                market_context=market_context,
                signal_date=signal_date,
            )
            consecutive_limit_ups = self._consecutive_limit_up_count(signal_date, code, dates)
            features["limit_times"] = float(consecutive_limit_ups)
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "name": str(candidate.get("name", candidate.get("股票", ""))),
                    "industry": _text_from(candidate, INDUSTRY_ALIASES),
                    "stage": f"{consecutive_limit_ups}→{consecutive_limit_ups + 1}",
                    "source_rank": features["source_rank"],
                    "d_close": d_close,
                    "estimated_up_limit": _round_price(d_close * (1.0 + limit_ratio)),
                    "mechanism_limit_pct": mechanism_limit_pct,
                    "decision_universe_rule": str(candidate.get("decision_universe_rule", "a_share_price_limit_le_10_v2")),
                    "decision_universe_reason": str(candidate.get("decision_universe_reason", "eligible")),
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
            (
                (out["net_return"] >= out["predicted_outcome_q10"])
                & (out["net_return"] <= out["predicted_outcome_q90"])
            ).astype(int),
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
        stage_focus_filled = filled[pd.to_numeric(filled.get("stage_focus"), errors="coerce").fillna(0).eq(1)]
        dates = sorted(oos["signal_date"].unique())
        signal_date_set = set(selected["signal_date"].astype(str))
        signal_dates = len(signal_date_set)
        signal_date_ratio = signal_dates / len(dates) if dates else float("nan")
        max_no_signal_streak = 0
        current_no_signal_streak = 0
        for date in dates:
            if str(date) in signal_date_set:
                current_no_signal_streak = 0
            else:
                current_no_signal_streak += 1
                max_no_signal_streak = max(max_no_signal_streak, current_no_signal_streak)
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
        big_loss_rate = float((filled["strategy_net_return"] <= self.config.big_loss_threshold).mean()) if len(filled) else float("nan")
        tail_count = max(1, int(math.ceil(len(filled) * 0.10))) if len(filled) else 0
        tail_mean = float(filled["strategy_net_return"].nsmallest(tail_count).mean()) if tail_count else float("nan")
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
        exit_on_time_rate = float(pd.to_numeric(filled.get("exit_on_time"), errors="coerce").mean()) if len(filled) else float("nan")
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
            "signal_dates": int(signal_dates),
            "signal_date_ratio": _safe_metric(signal_date_ratio),
            "max_no_signal_streak": int(max_no_signal_streak),
            "average_signals_per_signal_date": _safe_metric(
                len(selected) / signal_dates if signal_dates else np.nan
            ),
            "filled_trades": int(len(filled)),
            "fill_rate": _safe_metric(len(filled) / len(selected) if len(selected) else np.nan),
            "exit_on_time_rate": _safe_metric(exit_on_time_rate),
            "mean_trade_net_return": _safe_metric(filled["strategy_net_return"].mean()),
            "median_trade_net_return": _safe_metric(filled["strategy_net_return"].median()),
            "win_rate": _safe_metric((filled["strategy_net_return"] > 0).mean()),
            "stage_focus_signals": int(pd.to_numeric(selected.get("stage_focus"), errors="coerce").fillna(0).eq(1).sum()),
            "stage_focus_filled_trades": int(len(stage_focus_filled)),
            "stage_focus_continuation_hit_rate": _safe_metric(
                pd.to_numeric(stage_focus_filled.get("continuation_limit_up_hit"), errors="coerce").mean()
            ),
            "realized_big_loss_rate": _safe_metric(big_loss_rate),
            "big_loss_threshold": self.config.big_loss_threshold,
            "tail_10pct_mean_return": _safe_metric(tail_mean),
            "worst_trade_net_return": _safe_metric(filled["strategy_net_return"].min()),
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
            "filled_trades": len(filled) >= self.config.promotion_min_filled_trades,
            "stage_focus_filled_trades": (
                len(stage_focus_filled) >= self.config.promotion_min_stage_focus_filled_trades
            ),
            "signal_date_ratio": (
                math.isfinite(signal_date_ratio)
                and signal_date_ratio >= self.config.min_oos_signal_date_ratio
            ),
            "no_signal_streak": max_no_signal_streak <= self.config.max_oos_no_signal_streak,
            "mean_daily_return": math.isfinite(mean_daily) and mean_daily > 0.0,
            "stress_2x_cost": len(stress_daily) > 0 and float(stress_daily.mean()) > 0.0,
            "bootstrap_95": math.isfinite(bootstrap) and bootstrap >= 0.95,
            "rolling_20d_60pct": len(rolling) > 0 and float((rolling > 0).mean()) >= 0.60,
            "positive_months_60pct": len(monthly) >= 6 and positive_month_ratio >= 0.60,
            "month_concentration": math.isfinite(max_month_contribution) and max_month_contribution <= 0.50,
            "beats_source_topn": len(benchmark) > 0 and mean_daily > float(benchmark.mean()),
            "price_cap_not_worse": len(uncapped) > 0 and mean_daily >= float(uncapped.mean()),
            "exit_on_time_90pct": math.isfinite(exit_on_time_rate) and exit_on_time_rate >= self.config.min_exit_probability,
            "big_loss_rate_cap": math.isfinite(big_loss_rate) and big_loss_rate <= self.config.max_big_loss_probability,
            "tail_loss_floor": math.isfinite(tail_mean) and tail_mean >= self.config.min_tail_mean_return,
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
        expected_buy = choose_exec_date(signal_date, expected_buy)
        return expected_buy, choose_exit_date(expected_buy)

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
                frozen_version = str(frozen.get("model_version", pd.Series(["legacy"])).iloc[0])
                if frozen_version == self.config.model_version:
                    _write_csv(frozen, self.config.prediction_root / "pred_latest.csv")
                    return frozen
                safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", frozen_version).strip("._")[:80] or "legacy"
                archive_path = self.config.prediction_root / f"pred_{signal_date}_{safe_version}.csv"
                if not archive_path.exists():
                    _write_csv(frozen, archive_path)
        base = self._current_base(signal_date, candidates)
        scored = self.score_candidates(base, bundle)
        expected_buy, expected_exit = self._prediction_dates(signal_date, candidates)
        promoted = backtest_metrics.get("promoted") is True
        scored["prediction_id"] = [f"{signal_date}-{code}-{self.config.model_version}" for code in scored["ts_code"]]
        scored["expected_buy_date"] = expected_buy
        scored["expected_exit_date"] = expected_exit
        scored["recommended_max_price"] = [
            _round_price(min(row["estimated_up_limit"] - 0.01, row["d_close"] * (1.0 + row["recommended_max_gap"])))
            if math.isfinite(_finite(row["recommended_max_gap"]))
            else np.nan
            for _, row in scored.iterrows()
        ]
        scored["max_auction_change_pct"] = (100.0 * scored["recommended_max_gap"]).round(2)
        scored["stage_transition"] = scored["stage"].where(scored["stage"].astype(str).str.strip().ne(""), "-")
        observation_contracts = [
            observation_price_contract(row.to_dict())
            for _, row in scored.iterrows()
        ]
        scored["observation_max_price"] = [
            item["observation_max_price"] for item in observation_contracts
        ]
        scored["observation_auction_change_pct"] = [
            item["observation_auction_change_pct"] for item in observation_contracts
        ]
        scored["observation_price_basis"] = [
            item["observation_price_basis"] for item in observation_contracts
        ]
        scored["observation_price_is_formal"] = [
            int(item["observation_price_is_formal"]) for item in observation_contracts
        ]
        observation_rows, observation_pool_size = rank_observation_rows(
            scored.to_dict(orient="records"),
            limit=self.config.max_observation_candidates,
        )
        observation_rank = {
            str(row.get("ts_code")): int(row["observation_rank"])
            for row in observation_rows
        }
        scored["observation_rank"] = scored["ts_code"].map(observation_rank)
        scored["observation_selected"] = scored["observation_rank"].notna().astype(int)
        scored["observation_pool_size"] = int(observation_pool_size)
        scored["take_profit_pct"] = float(self.config.take_profit_pct)
        scored["stop_loss_pct"] = float(self.config.stop_loss_pct)
        scored["take_profit_price"] = scored["recommended_max_price"].map(
            lambda value: _round_price(float(value) * (1.0 + self.config.take_profit_pct))
            if math.isfinite(_finite(value)) else np.nan
        )
        scored["stop_loss_price"] = scored["recommended_max_price"].map(
            lambda value: _round_price(float(value) * (1.0 + self.config.stop_loss_pct))
            if math.isfinite(_finite(value)) else np.nan
        )
        scored["latest_exit_time"] = self.config.latest_exit_time
        scored["exit_policy_version"] = self.config.exit_policy_version
        scored["entry_rule"] = "系统仅供人工参考：T日9:25前仅用限价单参与集合竞价；不得使用无上限市价单，高于上限或未成交均放弃"
        scored["exit_rule"] = "T+1按实际成交价计算止盈/止损，首次触发即人工退出；均未触发则14:50退出；一字跌停顺延"
        scored["guidance_only"] = 1
        scored["broker_connected"] = 0
        scored["order_type"] = "LIMIT_ONLY_MANUAL"
        scored["market_order_allowed"] = 0
        scored["max_big_loss_probability"] = float(self.config.max_big_loss_probability)
        scored["big_loss_threshold"] = float(self.config.big_loss_threshold)
        scored["min_return_lcb"] = float(self.config.min_return_lcb)
        scored["model_version"] = self.config.model_version
        scored["model_ready"] = int(bundle is not None)
        scored["model_promoted"] = int(promoted)
        scored["action"] = np.where(
            scored["selected"].eq(1),
            "BUY" if promoted else "SHADOW_ONLY",
            np.where(scored["model_reason"].eq("insufficient_independent_history"), "WATCH", "REJECT"),
        )
        scored["price_action"] = np.where(
            scored["recommended_max_price"].notna(),
            "仅限人工限价单；竞价不高于上限价，超过即放弃" if promoted else "影子验证限价；当前不得买入",
            "没有安全买入价格，放弃",
        )
        scored["generated_at_utc"] = _utc_now()
        scored["source_snapshot_sha256"] = _hash_frame(candidates)
        scored["feature_contract"] = "D_CLOSE_PLUS_D_MINUTE_NO_T_LEAKAGE_V4_DERIVED_LIMIT_STREAK"
        ordered = [
            "prediction_id",
            "signal_date",
            "expected_buy_date",
            "expected_exit_date",
            "ts_code",
            "name",
            "industry",
            "stage",
            "stage_transition",
            "stage_focus",
            "source_rank",
            "d_close",
            "estimated_up_limit",
            "recommended_max_price",
            "max_auction_change_pct",
            "observation_max_price",
            "observation_auction_change_pct",
            "observation_price_basis",
            "observation_price_is_formal",
            "observation_rank",
            "observation_selected",
            "observation_pool_size",
            "take_profit_pct",
            "stop_loss_pct",
            "take_profit_price",
            "stop_loss_price",
            "latest_exit_time",
            "exit_policy_version",
            "predicted_net_return",
            "predicted_return_lcb",
            "predicted_return_ucb",
            "predicted_outcome_q10",
            "predicted_outcome_q90",
            "predicted_fill_probability",
            "predicted_exit_probability",
            "predicted_profit_probability",
            "predicted_big_loss_probability",
            "predicted_continuation_limit_up_probability",
            "max_big_loss_probability",
            "big_loss_threshold",
            "min_return_lcb",
            "conservative_ev",
            "selection_score",
            "risk_gate_pass",
            "selected",
            "action",
            "price_action",
            "entry_rule",
            "exit_rule",
            "guidance_only",
            "broker_connected",
            "order_type",
            "market_order_allowed",
            "mechanism_limit_pct",
            "decision_universe_rule",
            "decision_universe_reason",
            "minute_available",
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

    def _manual_feedback(self) -> pd.DataFrame:
        frame = _read_csv(self.config.manual_feedback_path)
        if frame.empty:
            return frame
        frame = frame.copy()
        if "ts_code" in frame.columns:
            frame["ts_code"] = frame["ts_code"].map(_normal_code)
        if "signal_date" in frame.columns:
            frame["signal_date"] = frame["signal_date"].map(_normal_date)
        return frame

    def _manual_feedback_row(self, feedback: pd.DataFrame, signal_date: str, code: str) -> Optional[pd.Series]:
        if feedback.empty or not {"signal_date", "ts_code"}.issubset(feedback.columns):
            return None
        hit = feedback[(feedback["signal_date"] == signal_date) & (feedback["ts_code"] == code)]
        return hit.iloc[-1] if not hit.empty else None

    def _verify_prediction_file(self, path: Path, dates: Sequence[str], feedback: pd.DataFrame) -> pd.DataFrame:
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
            expected_buy = _normal_date(row.get("expected_buy_date"))
            buy_date = expected_buy or later[0]
            if buy_date not in dates:
                records.append(base)
                continue
            buy_daily = self._row(self.market_table(buy_date, "daily"), code)
            if buy_daily is None:
                base.update({"actual_buy_date": buy_date, "verification_status": "NO_FILL", "actual_fill": 0, "actual_fill_reason": "suspended_or_daily_missing"})
                records.append(base)
                continue
            buy_price = self._execution_open_price(buy_date, code, buy_daily)
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
                    "truth_source": "tushare_minute_proxy" if not self.minute_table(buy_date, code).empty else "market_proxy",
                    "verification_status": "PENDING_EXIT" if actual_fill else "NO_FILL",
                }
            )
            expected_exit = _normal_date(row.get("expected_exit_date"))
            if not expected_exit or expected_exit not in dates:
                records.append(base)
                continue
            date_index = dates.index(expected_exit)
            exit_date, exit_price, _, exit_reason = self._resolve_exit(
                code,
                date_index,
                dates,
                entry_price=buy_price,
                buy_date=buy_date,
            )
            if not exit_date:
                records.append(base)
                continue
            gross = self._realized_gross_return(
                code,
                buy_date,
                buy_price,
                exit_date,
                dates,
                exit_price=exit_price,
            )
            net = gross - self.config.cost_rate if math.isfinite(gross) else np.nan
            feedback_row = self._manual_feedback_row(feedback, signal_date, code)
            if feedback_row is not None:
                feedback_buy = _numeric_from(feedback_row, ("buy_price", "actual_buy_price"))
                feedback_sell = _numeric_from(feedback_row, ("sell_price", "actual_exit_price"))
                feedback_fees = _numeric_from(feedback_row, ("fees", "total_fees"), 0.0)
                quantity = _numeric_from(feedback_row, ("quantity", "qty"), 0.0)
                if feedback_buy > 0 and feedback_sell > 0:
                    gross = feedback_sell / feedback_buy - 1.0
                    fee_rate = feedback_fees / (feedback_buy * quantity) if quantity > 0 else self.config.cost_rate
                    net = gross - fee_rate
                    buy_price, exit_price = feedback_buy, feedback_sell
                    base["truth_source"] = "manual_actual"
                    base["actual_fill"] = 1
                    base["actual_fill_reason"] = "manual_fill_feedback"
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
                    "risk_prediction_success": int(net > self.config.big_loss_threshold) if math.isfinite(predicted_loss) else np.nan,
                    "forecast_error": net - predicted if math.isfinite(predicted) else np.nan,
                    "correct_rejection": int(net <= 0) if selected is False else np.nan,
                    "missed_opportunity": int(net > 0) if selected is False else np.nan,
                }
            )
            records.append(base)
        return pd.DataFrame(records)

    def _verify_observation_prediction_file(
        self,
        path: Path,
        dates: Sequence[str],
    ) -> pd.DataFrame:
        pred = _read_csv(path)
        if pred.empty:
            return pd.DataFrame()
        ranked, pool_size = rank_observation_rows(
            pred.to_dict(orient="records"),
            limit=self.config.max_observation_candidates,
        )
        records: list[dict[str, Any]] = []
        for row in ranked:
            signal_date = _normal_date(row.get("signal_date"))
            buy_date = _normal_date(row.get("expected_buy_date"))
            exit_date = _normal_date(row.get("expected_exit_date"))
            if not buy_date or buy_date < self.config.observation_validation_start_date:
                continue
            code = _normal_code(row.get("ts_code"))
            timing_status, timing_valid, prediction_deadline = (
                self._prediction_timing_status(
                    row.get("generated_at_utc"),
                    buy_date,
                )
            )
            base = dict(row)
            base.update(
                {
                    "signal_date": signal_date,
                    "expected_buy_date": buy_date,
                    "expected_exit_date": exit_date,
                    "observation_pool_size": int(pool_size),
                    "observation_rank": int(row.get("observation_rank") or 0),
                    "validation_mode": "public_market_proxy",
                    "prediction_timing_status": timing_status,
                    "prediction_timing_valid": timing_valid,
                    "prediction_deadline_utc": prediction_deadline,
                    "validation_status": "PENDING_T",
                    "actual_buy_date": "",
                    "actual_open_price": np.nan,
                    "actual_t_close": np.nan,
                    "market_daily_return": np.nan,
                    "observation_fill": np.nan,
                    "observation_fill_reason": "truth_pending",
                    "observation_t_return": np.nan,
                    "continuation_limit_up_hit": np.nan,
                    "actual_exit_date": "",
                    "actual_exit_price": np.nan,
                    "actual_gross_return": np.nan,
                    "actual_net_return": np.nan,
                    "exit_reason": "",
                    "truth_source": "pending",
                    "truth_generated_at_utc": _utc_now(),
                }
            )
            if buy_date not in dates:
                records.append(base)
                continue
            buy_daily = self._row(self.market_table(buy_date, "daily"), code)
            if buy_daily is None:
                base.update(
                    {
                        "actual_buy_date": buy_date,
                        "observation_fill": 0,
                        "observation_fill_reason": "suspended_or_daily_missing",
                        "validation_status": "FINAL_NO_FILL",
                        "truth_source": "daily_missing",
                    }
                )
                records.append(base)
                continue

            open_price = self._execution_open_price(buy_date, code, buy_daily)
            close_price = _numeric_from(buy_daily, ("close",))
            daily_pct = _numeric_from(buy_daily, ("pct_chg",))
            d_close = _finite(row.get("d_close"))
            market_return = (
                daily_pct / 100.0
                if math.isfinite(daily_pct)
                else close_price / d_close - 1.0
                if close_price > 0 and d_close > 0
                else np.nan
            )
            max_price = _finite(row.get("observation_max_price"))
            market_fill, market_reason = self._market_buyable(buy_date, code)
            cap_accept = math.isfinite(max_price) and open_price > 0 and open_price <= max_price + 0.005
            observation_fill = int(cap_accept and market_fill == 1)
            fill_reason = (
                "filled_public_market_proxy"
                if observation_fill
                else "price_above_observation_cap"
                if not cap_accept
                else market_reason
            )
            limit_row = self._row(self.market_table(buy_date, "stk_limit"), code)
            up_limit = _numeric_from(limit_row, ("up_limit",)) if limit_row is not None else float("nan")
            continuation_hit = (
                int(_is_close(close_price, up_limit))
                if close_price > 0 and math.isfinite(up_limit)
                else np.nan
            )
            t_return = (
                close_price / open_price - 1.0
                if observation_fill and close_price > 0 and open_price > 0
                else np.nan
            )
            base.update(
                {
                    "actual_buy_date": buy_date,
                    "actual_open_price": open_price,
                    "actual_t_close": close_price,
                    "market_daily_return": market_return,
                    "observation_fill": observation_fill,
                    "observation_fill_reason": fill_reason,
                    "observation_t_return": t_return,
                    "continuation_limit_up_hit": continuation_hit,
                    "validation_status": "T_VERIFIED_FILLED" if observation_fill else "T_VERIFIED_NO_FILL",
                    "truth_source": (
                        "tushare_minute_proxy"
                        if not self.minute_table(buy_date, code).empty
                        else "official_daily_open_proxy"
                    ),
                }
            )
            if not observation_fill:
                base["validation_status"] = "FINAL_NO_FILL"
                records.append(base)
                continue
            if not exit_date or exit_date not in dates:
                base["validation_status"] = "PENDING_T1"
                records.append(base)
                continue

            date_index = dates.index(exit_date)
            actual_exit_date, actual_exit_price, _, exit_reason = self._resolve_exit(
                code,
                date_index,
                dates,
                entry_price=open_price,
                buy_date=buy_date,
            )
            if not actual_exit_date:
                base["validation_status"] = "PENDING_EXIT_TRUTH"
                records.append(base)
                continue
            gross = self._realized_gross_return(
                code,
                buy_date,
                open_price,
                actual_exit_date,
                dates,
                exit_price=actual_exit_price,
            )
            net = gross - self.config.cost_rate if math.isfinite(gross) else np.nan
            base.update(
                {
                    "actual_exit_date": actual_exit_date,
                    "actual_exit_price": actual_exit_price,
                    "actual_gross_return": gross,
                    "actual_net_return": net,
                    "exit_reason": exit_reason,
                    "validation_status": "FINAL_VERIFIED",
                }
            )
            records.append(base)
        return pd.DataFrame(records)

    @staticmethod
    def _prediction_timing_status(
        generated_at_utc: Any,
        buy_date: str,
    ) -> tuple[str, int, str]:
        normalized_buy_date = _normal_date(buy_date)
        if not normalized_buy_date:
            return "UNKNOWN_BUY_DATE", 0, ""
        deadline = datetime.strptime(normalized_buy_date, "%Y%m%d").replace(
            hour=9,
            minute=25,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        )
        raw_generated = str(generated_at_utc or "").strip()
        if not raw_generated:
            return "UNKNOWN_GENERATION_TIME", 0, deadline.astimezone(timezone.utc).isoformat()
        try:
            generated = datetime.fromisoformat(raw_generated.replace("Z", "+00:00"))
        except ValueError:
            return "UNKNOWN_GENERATION_TIME", 0, deadline.astimezone(timezone.utc).isoformat()
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        valid = generated <= deadline
        return (
            "PREMARKET_VALID" if valid else "RETROSPECTIVE_LATE_GENERATION",
            int(valid),
            deadline.astimezone(timezone.utc).isoformat(),
        )

    @staticmethod
    def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
        if total <= 0:
            return None, None
        z = 1.959963984540054
        rate = successes / total
        denominator = 1.0 + z * z / total
        centre = (rate + z * z / (2.0 * total)) / denominator
        radius = (
            z
            * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
            / denominator
        )
        return _safe_metric(max(0.0, centre - radius)), _safe_metric(min(1.0, centre + radius))

    def _observation_metrics(self, ledger: pd.DataFrame) -> dict[str, Any]:
        if ledger.empty:
            return {
                "schema_version": "decision_observation_validation_v2_timing_audited",
                "status": "no_observation_predictions",
                "generated_at_utc": _utc_now(),
                "validation_start_exec_date": self.config.observation_validation_start_date,
                "observation_rows": 0,
            }
        frame = ledger.copy()
        frame["expected_buy_date"] = frame["expected_buy_date"].map(_normal_date)
        frame["observation_rank"] = pd.to_numeric(frame.get("observation_rank"), errors="coerce")
        timing_valid = pd.to_numeric(
            frame.get("prediction_timing_valid"), errors="coerce"
        ).fillna(0).eq(1)
        eligible_frame = frame[timing_valid].copy()
        all_t_validated = frame[frame["validation_status"].ne("PENDING_T")].copy()
        t_validated = eligible_frame[
            eligible_frame["validation_status"].ne("PENDING_T")
        ].copy()
        retrospective_truth = all_t_validated[
            all_t_validated["prediction_timing_status"].eq(
                "RETROSPECTIVE_LATE_GENERATION"
            )
        ]
        unknown_timing_truth = all_t_validated[
            all_t_validated["prediction_timing_status"].astype(str).str.startswith(
                "UNKNOWN_"
            )
        ]
        market_returns = pd.to_numeric(t_validated.get("market_daily_return"), errors="coerce").dropna()
        fill_flags = pd.to_numeric(t_validated.get("observation_fill"), errors="coerce").dropna()
        t_fill_returns = pd.to_numeric(
            t_validated.loc[
                pd.to_numeric(t_validated.get("observation_fill"), errors="coerce").eq(1),
                "observation_t_return",
            ],
            errors="coerce",
        ).dropna()
        continuation = pd.to_numeric(
            t_validated.get("continuation_limit_up_hit"), errors="coerce"
        ).dropna()
        final = eligible_frame[
            eligible_frame["validation_status"].eq("FINAL_VERIFIED")
        ].copy()
        final_returns = pd.to_numeric(final.get("actual_net_return"), errors="coerce").dropna()
        all_final = frame[frame["validation_status"].eq("FINAL_VERIFIED")].copy()
        all_final_returns = pd.to_numeric(
            all_final.get("actual_net_return"), errors="coerce"
        ).dropna()
        positive = float(final_returns[final_returns > 0].sum())
        negative = float(-final_returns[final_returns < 0].sum())

        daily_records: list[dict[str, Any]] = []
        for exec_date, group in eligible_frame.groupby("expected_buy_date", sort=True):
            statuses = set(group["validation_status"].astype(str))
            if not statuses or any(not status.startswith("FINAL_") for status in statuses):
                continue
            slot_returns = pd.to_numeric(group.get("actual_net_return"), errors="coerce").fillna(0.0)
            daily_records.append(
                {
                    "exec_date": exec_date,
                    "observation_slots": int(len(group)),
                    "filled_slots": int(pd.to_numeric(group.get("observation_fill"), errors="coerce").fillna(0).sum()),
                    "equal_slot_net_return": float(slot_returns.sum() / max(len(group), 1)),
                }
            )
        daily = pd.DataFrame(daily_records)
        if not daily.empty:
            daily["nav"] = (1.0 + daily["equal_slot_net_return"]).cumprod()
            peak = daily["nav"].cummax().clip(lower=1.0)
            drawdown = daily["nav"] / peak - 1.0
            cumulative_return = float(daily["nav"].iloc[-1] - 1.0)
            max_drawdown = float(drawdown.min())
        else:
            cumulative_return = float("nan")
            max_drawdown = float("nan")

        win_low, win_high = self._wilson_interval(
            int((final_returns > 0).sum()),
            int(len(final_returns)),
        )
        continuation_low, continuation_high = self._wilson_interval(
            int(continuation.sum()),
            int(len(continuation)),
        )
        payload: dict[str, Any] = {
            "schema_version": "decision_observation_validation_v2_timing_audited",
            "status": "ok",
            "generated_at_utc": _utc_now(),
            "validation_start_exec_date": self.config.observation_validation_start_date,
            "validation_mode": "public_market_proxy",
            "performance_scope": "premarket_valid_predictions_only",
            "prediction_deadline": "T 09:25 Asia/Shanghai",
            "top_n": int(self.config.max_observation_candidates),
            "observation_dates": int(frame["expected_buy_date"].nunique()),
            "observation_rows": int(len(frame)),
            "t_validated_rows": int(len(all_t_validated)),
            "t_pending_rows": int(len(frame) - len(all_t_validated)),
            "premarket_valid_rows": int(timing_valid.sum()),
            "premarket_validated_rows": int(len(t_validated)),
            "retrospective_truth_rows": int(len(retrospective_truth)),
            "unknown_timing_truth_rows": int(len(unknown_timing_truth)),
            "market_positive_rate": _safe_metric((market_returns > 0).mean()),
            "mean_market_daily_return": _safe_metric(market_returns.mean()),
            "median_market_daily_return": _safe_metric(market_returns.median()),
            "fillable_rows": int(fill_flags.sum()) if len(fill_flags) else 0,
            "observation_fill_rate": _safe_metric(fill_flags.mean()),
            "mean_t_observation_return": _safe_metric(t_fill_returns.mean()),
            "median_t_observation_return": _safe_metric(t_fill_returns.median()),
            "continuation_hits": int(continuation.sum()) if len(continuation) else 0,
            "continuation_samples": int(len(continuation)),
            "continuation_hit_rate": _safe_metric(continuation.mean()),
            "continuation_hit_rate_95ci_low": continuation_low,
            "continuation_hit_rate_95ci_high": continuation_high,
            "final_verified_trades": int(len(final_returns)),
            "final_win_rate": _safe_metric((final_returns > 0).mean()),
            "final_win_rate_95ci_low": win_low,
            "final_win_rate_95ci_high": win_high,
            "mean_final_net_return": _safe_metric(final_returns.mean()),
            "median_final_net_return": _safe_metric(final_returns.median()),
            "worst_final_net_return": _safe_metric(final_returns.min()),
            "tail_10pct_mean_return": _safe_metric(
                final_returns.nsmallest(max(1, math.ceil(len(final_returns) * 0.10))).mean()
                if len(final_returns)
                else np.nan
            ),
            "profit_factor": _safe_metric(positive / negative if negative > 0 else np.nan),
            "matured_portfolio_dates": int(len(daily)),
            "equal_slot_cumulative_return": _safe_metric(cumulative_return),
            "equal_slot_max_drawdown": _safe_metric(max_drawdown),
            "equal_slot_mean_daily_return": _safe_metric(
                daily["equal_slot_net_return"].mean() if not daily.empty else np.nan
            ),
            "latest_t_validated_exec_date": (
                max(t_validated["expected_buy_date"]) if not t_validated.empty else ""
            ),
            "latest_all_truth_exec_date": (
                max(all_t_validated["expected_buy_date"])
                if not all_t_validated.empty
                else ""
            ),
            "latest_final_exec_date": (
                max(daily["exec_date"]) if not daily.empty else ""
            ),
            "daily_portfolio": [
                {
                    "exec_date": str(row["exec_date"]),
                    "observation_slots": int(row["observation_slots"]),
                    "filled_slots": int(row["filled_slots"]),
                    "equal_slot_net_return": _safe_metric(row["equal_slot_net_return"]),
                    "nav": _safe_metric(row["nav"]),
                }
                for _, row in daily.iterrows()
            ],
            "all_truth_summary": {
                "t_validated_rows": int(len(all_t_validated)),
                "fillable_rows": int(
                    pd.to_numeric(
                        all_t_validated.get("observation_fill"),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "final_verified_trades": int(len(all_final_returns)),
                "final_win_rate": _safe_metric(
                    (all_final_returns > 0).mean()
                ),
                "mean_final_net_return": _safe_metric(all_final_returns.mean()),
            },
        }

        for transition, key in (("2→3", "stage_2_to_3"), ("3→4", "stage_3_to_4")):
            sample = t_validated[t_validated["stage_transition"].eq(transition)]
            hits = pd.to_numeric(sample.get("continuation_limit_up_hit"), errors="coerce").dropna()
            payload[key] = {
                "samples": int(len(hits)),
                "hits": int(hits.sum()) if len(hits) else 0,
                "hit_rate": _safe_metric(hits.mean()),
            }
        for cutoff in (1, 3, 10):
            sample = t_validated[t_validated["observation_rank"].le(cutoff)]
            hits = pd.to_numeric(sample.get("continuation_limit_up_hit"), errors="coerce").dropna()
            payload[f"top{cutoff}_continuation"] = {
                "samples": int(len(hits)),
                "hits": int(hits.sum()) if len(hits) else 0,
                "hit_rate": _safe_metric(hits.mean()),
            }

        rolling: dict[str, Any] = {}
        for label, size in (("20", 20), ("60", 60), ("all", 0)):
            sample_daily = daily.tail(size) if size else daily
            dates_in_window = set(sample_daily["exec_date"].astype(str)) if not sample_daily.empty else set()
            sample_trades = final[final["expected_buy_date"].isin(dates_in_window)]
            sample_returns = pd.to_numeric(sample_trades.get("actual_net_return"), errors="coerce").dropna()
            rolling[label] = {
                "portfolio_dates": int(len(sample_daily)),
                "filled_trades": int(len(sample_returns)),
                "mean_net_return": _safe_metric(sample_returns.mean()),
                "win_rate": _safe_metric((sample_returns > 0).mean()),
                "equal_slot_cumulative_return": _safe_metric(
                    (1.0 + sample_daily["equal_slot_net_return"]).prod() - 1.0
                    if not sample_daily.empty
                    else np.nan
                ),
            }
        payload["trading_date_windows"] = rolling
        return payload

    def settle_observations(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        dates = self.market_dates()
        parts: list[pd.DataFrame] = []
        for path in sorted(self.config.prediction_root.glob("pred_20*.csv")):
            if not re.fullmatch(r"pred_20\d{6}\.csv", path.name):
                continue
            verified = self._verify_observation_prediction_file(path, dates)
            if not verified.empty:
                parts.append(verified)
        ledger = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not ledger.empty:
            ledger = ledger.sort_values(
                ["expected_buy_date", "observation_rank", "ts_code"],
                kind="stable",
            ).reset_index(drop=True)
            for exec_date, group in ledger.groupby("expected_buy_date", sort=True):
                _write_csv(
                    group.reset_index(drop=True),
                    self.config.verification_root / f"observation_{exec_date}.csv",
                )
        _write_csv(
            ledger,
            self.config.verification_root / "observation_latest.csv",
        )
        metrics = self._observation_metrics(ledger)
        _write_json(
            metrics,
            self.config.metrics_root / "observation_cumulative_latest.json",
        )
        return ledger, metrics

    def settle_predictions(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        dates = self.market_dates()
        feedback = self._manual_feedback()
        parts: list[pd.DataFrame] = []
        for path in sorted(self.config.prediction_root.glob("pred_20*.csv")):
            if not re.fullmatch(r"pred_20\d{6}\.csv", path.name):
                continue
            verified = self._verify_prediction_file(path, dates, feedback)
            if verified.empty:
                continue
            signal_date = _normal_date(verified["signal_date"].iloc[0])
            _write_csv(verified, self.config.verification_root / f"verify_{signal_date}.csv")
            parts.append(verified)
        ledger = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _write_csv(ledger, self.config.verification_root / "verify_latest.csv")
        metrics = self._verification_metrics(ledger)
        _write_json(metrics, self.config.metrics_root / "cumulative_latest.json")
        self.settle_observations()
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
            "manual_actual_trades": int((verified.get("truth_source", pd.Series(dtype=str)) == "manual_actual").sum()),
            "tushare_minute_proxy_trades": int((verified.get("truth_source", pd.Series(dtype=str)) == "tushare_minute_proxy").sum()),
            "market_proxy_trades": int((verified.get("truth_source", pd.Series(dtype=str)) == "market_proxy").sum()),
            "fill_rate": _safe_metric(pd.to_numeric(selected.get("actual_fill"), errors="coerce").mean()),
            "price_guidance_accuracy": _safe_metric(pd.to_numeric(selected.get("price_guidance_success"), errors="coerce").mean()),
            "win_rate": _safe_metric((returns > 0).mean()),
            "realized_big_loss_rate": _safe_metric((returns <= self.config.big_loss_threshold).mean()),
            "big_loss_avoidance_rate": _safe_metric((returns > self.config.big_loss_threshold).mean()),
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
        current_eligibility_audit = dict(self._eligibility_audit)
        if candidates.empty:
            raise RuntimeError(f"candidate snapshot has no <=10% price-limit stocks for {signal_date}")

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
            "calibration_rows": bundle.calibration_rows if bundle else 0,
            "calibration_dates": bundle.calibration_dates if bundle else 0,
            "calibration_bias": _safe_metric(bundle.calibration_bias if bundle else np.nan),
            "expected_return_margin": _safe_metric(
                bundle.expected_return_margin if bundle else np.nan
            ),
            "return_selection": bundle.return_selection if bundle else {},
            "classifier_selection": bundle.classifier_selection if bundle else {},
            "residual_q10": _safe_metric(bundle.residual_q10 if bundle else np.nan),
            "residual_q90": _safe_metric(bundle.residual_q90 if bundle else np.nan),
            "exit_on_time_base_rate": _safe_metric(bundle.exit_constant if bundle else np.nan),
            "continuation_limit_up_base_rate": _safe_metric(
                bundle.continuation_constant if bundle else np.nan
            ),
            "gap_support": {
                "min": _safe_metric(bundle.gap_min if bundle else np.nan),
                "max": _safe_metric(bundle.gap_max if bundle else np.nan),
            },
            "backtest_gate": backtest_metrics,
            "universe_eligibility": current_eligibility_audit,
            "contract": {
                "signal": "D close",
                "candidate_pool": "D-day confirmed limit-up candidates only",
                "stage_focus": "prefer 2-to-3 and 3-to-4 by out-of-sample continuation probability after every risk veto passes",
                "guidance_only": True,
                "broker_connected": False,
                "entry": "manual limit order only before T 09:25 opening-auction cutoff with a frozen maximum price; market order forbidden",
                "exit": "manual T+1 first-touch take-profit/stop-loss, then 14:50 time exit; one-price limit-down delays exit",
                "exit_policy_version": self.config.exit_policy_version,
                "take_profit_pct": self.config.take_profit_pct,
                "stop_loss_pct": self.config.stop_loss_pct,
                "latest_exit_time": self.config.latest_exit_time,
                "cost_rate": self.config.cost_rate,
                "maximum_price_limit_mechanism_pct": self.config.max_mechanism_limit_pct,
                "maximum_big_loss_probability": self.config.max_big_loss_probability,
                "big_loss_threshold": self.config.big_loss_threshold,
                "minimum_return_lcb": self.config.min_return_lcb,
                "minute_data": "Tushare 1-minute data supports D-day features and public-market simulation; actual fills require manual feedback",
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
