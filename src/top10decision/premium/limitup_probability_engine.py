# -*- coding: utf-8 -*-
"""
Premium 子系统 — LimitUp Probability Engine（V3.6：专业概率引擎）

用途：
    把“涨停接力规则评分”升级为可训练、可校验、可落盘、可持续自学习的概率模型。

预测目标：
    t_limitup_prob_model      # T日收盘涨停概率（模型概率）
    t_touch_limitup_prob_model# T日盘中触板概率（模型概率）
    t1_up_prob_model          # T+1上涨概率（模型概率）
    t1_high_profit_prob_model # T+1给过可兑现收益概率（模型概率）
    t1_close_ret_pred         # T+1收盘收益预测
    t1_high_ret_pred          # T+1最高收益预测

说明：
    - 使用时间切分验证，避免未来函数。
    - 优先使用 LightGBM；不可用时降级到 sklearn HistGradientBoosting。
    - 每次有新标签后重新训练，即为当前阶段的“自学习闭环”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


CLASS_TARGETS = [
    "t_up_hit",
    "t_high_profit_hit",
    "t_limitup_hit",
    "t_touch_limitup",
    "t1_up_hit",
    "t1_high_profit_hit",
    "t1_accept_hit",
    "t1_fail_hit",
    "t1_big_drawdown_hit",
]
REG_TARGETS = ["t_close_ret", "t_intraday_ret", "t1_close_ret", "t1_high_ret"]
GATE_VERSION = "premium_target_gate_v4_walkforward_provenance"
BUNDLE_ARTIFACT_VERSION = 3
FEATURE_CONTRACT_VERSION = "premium_d_close_features_v1"
VALIDATION_MODE = "purged_expanding_walk_forward_v1"
RANK_ENABLED_GATE_STATES = {"ACTIVE", "PROVISIONAL"}
DEFAULT_EXCLUDE = set(CLASS_TARGETS + REG_TARGETS + [
    "ts_code", "code", "symbol", "名称", "name", "trade_date", "d_trade_date", "t_trade_date", "t1_trade_date",
    "label_valid", "label_matured", "calendar_source", "calendar_status", "calendar_reason", "label_as_of",
    "feature_as_of_date", "feature_known_at", "feature_snapshot_source", "feature_snapshot_sha256",
])

FEATURE_ALLOW_PREFIXES = [
    "d_", "factor_", "mkt_", "dec_", "eret_", "pfill_", "risk_", "theme_", "amount_",
    "ret_", "vol_", "close_pos_", "rank_", "t_limitup_prob_rule", "t_limitup_strength_rule",
    "t1_continue_up_rate_rule", "limitup_continuation_score_rule",
    "intraday_", "auction_",
]
FEATURE_ALLOW_EXACT = {
    "close_T", "p_premium", "e_premium", "score_ev",
    "confidence", "data_quality", "dec_weight", "dec_rank", "dec_p_fill",
    "eret_pred_raw", "eret_plus_value", "eret_plus_delta", "eret_plus_conf_score",
    "t1_up_rate", "r_p50", "r_p25", "r_p75",
    "late_withdraw_score", "reseal_score", "open_board_count", "auction_strength_score",
    "intraday_attack_edge", "intraday_execution_edge", "intraday_risk_penalty",
}
FEATURE_DENY_KEYWORDS = [
    "actual", "hit", "label", "verify", "future",
    "t_open", "t_high", "t_low", "t_close",
    "t1_open", "t1_high", "t1_low", "t1_close",
    "open_T_actual", "high_T_actual", "close_T_actual",
]


@dataclass
class LimitupModelBundle:
    feature_cols: List[str]
    class_models: Dict[str, object]
    reg_models: Dict[str, object]
    class_priors: Dict[str, float]
    reg_means: Dict[str, float]
    metrics: pd.DataFrame
    train_end_date: str
    valid_start_date: str
    model_can_rank: bool = False
    model_rank_mode: str = "disabled_validation_not_pass"
    validation_days: int = 0
    validation_samples: int = 0
    gate_reason: str = ""
    gate_version: str = GATE_VERSION
    target_gate_status: Dict[str, str] = field(default_factory=dict)
    target_gate_reasons: Dict[str, str] = field(default_factory=dict)
    target_probability_status: Dict[str, str] = field(default_factory=dict)
    target_probability_reasons: Dict[str, str] = field(default_factory=dict)
    validation_mode: str = VALIDATION_MODE
    walk_forward_folds: int = 0
    embargo_days: int = 2
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    data_fingerprint: str = ""
    feature_fingerprint: str = ""
    point_in_time_audit: Dict[str, object] = field(default_factory=dict)
    feature_availability: Dict[str, Dict[str, object]] = field(default_factory=dict)
    target_train_end_dates: Dict[str, str] = field(default_factory=dict)
    fold_boundaries: List[Dict[str, object]] = field(default_factory=list)
    artifact_version: int = BUNDLE_ARTIFACT_VERSION

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _make_X(df, self.feature_cols)
        out = df.copy()
        name_map = {
            "t_up_hit": "t_up_prob_model",
            "t_high_profit_hit": "t_high_profit_prob_model",
            "t_limitup_hit": "t_limitup_prob_model",
            "t_touch_limitup": "t_touch_limitup_prob_model",
            "t1_up_hit": "t1_up_prob_model",
            "t1_high_profit_hit": "t1_high_profit_prob_model",
            "t1_accept_hit": "t1_accept_prob_model",
            "t1_fail_hit": "t1_fail_prob_model",
            "t1_big_drawdown_hit": "t1_big_drawdown_prob_model",
        }
        for target, col in name_map.items():
            model = self.class_models.get(target)
            if model is None:
                out[col] = self.class_priors.get(target, 0.0)
            else:
                if hasattr(model, "predict_proba"):
                    out[col] = model.predict_proba(X)[:, 1]
                else:
                    raw = model.predict(X)
                    out[col] = np.clip(raw, 0.0, 1.0)
        reg_map = {
            "t_close_ret": "t_close_ret_pred",
            "t_intraday_ret": "t_intraday_ret_pred",
            "t1_close_ret": "t1_close_ret_pred",
            "t1_high_ret": "t1_high_ret_pred",
        }
        for target, col in reg_map.items():
            model = self.reg_models.get(target)
            if model is None:
                out[col] = self.reg_means.get(target, 0.0)
            else:
                out[col] = np.asarray(model.predict(X), dtype=float)
        out["limitup_model_score"] = (
            0.18 * out.get("t_up_prob_model", 0)
            + 0.18 * out.get("t_limitup_prob_model", 0)
            + 0.16 * out.get("t_touch_limitup_prob_model", 0)
            + 0.20 * out.get("t1_accept_prob_model", 0)
            + 0.14 * out.get("t1_up_prob_model", 0)
            + 0.14 * out.get("t1_high_profit_prob_model", 0)
        )
        legacy_state = "ACTIVE" if bool(getattr(self, "model_can_rank", False)) else "SHADOW"
        target_status = getattr(self, "target_gate_status", {}) or {}
        t_limit_status = str(target_status.get("t_limit", legacy_state)).upper()
        t_touch_status = str(target_status.get("t_touch", legacy_state)).upper()
        t1_status = str(target_status.get("t1", legacy_state)).upper()
        t_return_status = str(target_status.get("t_return", legacy_state)).upper()
        t1_return_status = str(target_status.get("t1_return", legacy_state)).upper()
        probability_status = getattr(self, "target_probability_status", {}) or {}
        t_limit_probability_status = str(probability_status.get("t_limit", "SHADOW")).upper()
        t_touch_probability_status = str(probability_status.get("t_touch", "SHADOW")).upper()
        t1_probability_status = str(probability_status.get("t1", "SHADOW")).upper()
        t_limit_can_rank = t_limit_status in RANK_ENABLED_GATE_STATES
        t_touch_can_rank = t_touch_status in RANK_ENABLED_GATE_STATES
        t1_can_rank = t1_status in RANK_ENABLED_GATE_STATES
        out["t_limit_gate_status"] = t_limit_status
        out["t_touch_gate_status"] = t_touch_status
        out["t1_gate_status"] = t1_status
        out["t_return_gate_status"] = t_return_status
        out["t1_return_gate_status"] = t1_return_status
        out["t_limit_prob_gate_status"] = t_limit_probability_status
        out["t_touch_prob_gate_status"] = t_touch_probability_status
        out["t1_prob_gate_status"] = t1_probability_status
        out["t_limit_model_can_rank"] = t_limit_can_rank
        out["t_touch_model_can_rank"] = t_touch_can_rank
        out["t1_model_can_rank"] = t1_can_rank
        out["t_return_model_can_rank"] = t_return_status in RANK_ENABLED_GATE_STATES
        out["t1_return_model_can_rank"] = t1_return_status in RANK_ENABLED_GATE_STATES
        out["t_limit_model_probability_ready"] = t_limit_probability_status in RANK_ENABLED_GATE_STATES
        out["t_touch_model_probability_ready"] = t_touch_probability_status in RANK_ENABLED_GATE_STATES
        out["t1_model_probability_ready"] = t1_probability_status in RANK_ENABLED_GATE_STATES
        out["model_gate_version"] = str(getattr(self, "gate_version", GATE_VERSION))
        out["model_can_rank"] = t_limit_can_rank or t_touch_can_rank
        out["model_rank_mode"] = getattr(self, "model_rank_mode", "disabled_validation_not_pass")
        out["model_quality_flag"] = (
            "ok"
            if t_limit_can_rank and t1_can_rank
            else "partial"
            if (t_limit_can_rank or t_touch_can_rank)
            else "degraded"
        )
        return out


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"暂不支持的输入格式: {suffix}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"暂不支持的输出格式: {suffix}")


def _norm_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    dt = pd.to_datetime(raw, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y%m%d", errors="coerce")
    return dt.dt.strftime("%Y%m%d")


def infer_feature_cols(df: pd.DataFrame, extra_exclude: Optional[Iterable[str]] = None) -> List[str]:
    exclude = set(DEFAULT_EXCLUDE)
    if extra_exclude:
        exclude.update(extra_exclude)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        cname = str(c)
        if any(k.lower() in cname.lower() for k in FEATURE_DENY_KEYWORDS):
            continue
        if not (cname in FEATURE_ALLOW_EXACT or any(cname.startswith(p) for p in FEATURE_ALLOW_PREFIXES)):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(20, int(len(df) * 0.05)):
            cols.append(c)
    return cols


def _make_X(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float)


def time_split(
    df: pd.DataFrame,
    date_col: str = "d_trade_date",
    valid_ratio: float = 0.30,
    embargo_days: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    data = df.copy()
    if date_col not in data.columns:
        raise ValueError(f"缺少时间切分字段 {date_col}")
    data[date_col] = _norm_date(data[date_col])
    dates = sorted(data[date_col].dropna().unique())
    if len(dates) < 4:
        raise ValueError("可用交易日太少，无法做时间切分验证")
    cut = max(1, int(len(dates) * (1 - valid_ratio)))
    cut = min(cut, len(dates) - 1)
    # D-day features can be close to the validation boundary. Purging the last
    # two training dates avoids accidental overlap with T/T+1 label horizons.
    train_cut = max(1, cut - max(0, int(embargo_days)))
    train_dates = set(dates[:train_cut])
    valid_dates = set(dates[cut:])
    train = data[data[date_col].isin(train_dates)].copy()
    valid = data[data[date_col].isin(valid_dates)].copy()
    return train, valid, max(train_dates), min(valid_dates)


def _blocked_feature_reason(name: object) -> str:
    cname = str(name).strip()
    lower = cname.lower()
    if cname in CLASS_TARGETS or cname in REG_TARGETS:
        return "target_column"
    for keyword in FEATURE_DENY_KEYWORDS:
        if str(keyword).lower() in lower:
            return f"deny_keyword:{keyword}"
    return ""


def audit_point_in_time_contract(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    date_col: str = "d_trade_date",
) -> Dict[str, object]:
    """Audit that model inputs belong to the D-close information set."""
    data = df.copy()
    blocked = {
        str(col): reason
        for col in feature_cols
        if (reason := _blocked_feature_reason(col))
    }
    d = _norm_date(data[date_col]) if date_col in data.columns else pd.Series(pd.NA, index=data.index)
    invalid_d = d.isna()

    invalid_t_order = pd.Series(False, index=data.index, dtype=bool)
    t_col = next((c for c in ("t_trade_date", "buy_date") if c in data.columns), "")
    t = pd.Series(pd.NA, index=data.index)
    if t_col:
        t = _norm_date(data[t_col])
        has_t = t.notna()
        invalid_t_order = has_t & (d.isna() | d.ge(t))

    invalid_t1_order = pd.Series(False, index=data.index, dtype=bool)
    t1_col = next((c for c in ("t1_trade_date", "target_date") if c in data.columns), "")
    if t1_col:
        t1 = _norm_date(data[t1_col])
        has_t1 = t1.notna()
        invalid_t1_order = has_t1 & (t.isna() | t.ge(t1))

    as_of_future = pd.Series(False, index=data.index, dtype=bool)
    as_of_missing = pd.Series(True, index=data.index, dtype=bool)
    if "feature_as_of_date" in data.columns:
        as_of = _norm_date(data["feature_as_of_date"])
        as_of_missing = as_of.isna()
        as_of_future = as_of.notna() & d.notna() & as_of.gt(d)

    known_at_missing = pd.Series(True, index=data.index, dtype=bool)
    known_at_invalid = pd.Series(False, index=data.index, dtype=bool)
    if "feature_known_at" in data.columns:
        known_at = data["feature_known_at"].astype(str).str.strip().str.upper()
        known_at_missing = known_at.isin({"", "NAN", "NONE", "<NA>"})
        known_at_invalid = ~known_at_missing & known_at.ne("D_CLOSE")

    snapshot_source_missing = pd.Series(True, index=data.index, dtype=bool)
    if "feature_snapshot_source" in data.columns:
        snapshot_source = data["feature_snapshot_source"].astype(str).str.strip()
        snapshot_source_missing = snapshot_source.isin({"", "nan", "None", "<NA>"})

    snapshot_sha_invalid = pd.Series(True, index=data.index, dtype=bool)
    if "feature_snapshot_sha256" in data.columns:
        snapshot_sha = data["feature_snapshot_sha256"].astype(str).str.strip().str.lower()
        snapshot_sha_invalid = ~snapshot_sha.str.fullmatch(r"[0-9a-f]{64}", na=False)

    provenance_columns_present = all(
        col in data.columns
        for col in (
            "feature_as_of_date",
            "feature_known_at",
            "feature_snapshot_source",
            "feature_snapshot_sha256",
        )
    )
    provenance_invalid = bool(
        provenance_columns_present
        and (
            known_at_missing.any()
            or known_at_invalid.any()
            or snapshot_source_missing.any()
            or snapshot_sha_invalid.any()
        )
    )

    hard_fail = bool(
        blocked
        or invalid_d.any()
        or invalid_t_order.any()
        or invalid_t1_order.any()
        or as_of_future.any()
        or provenance_invalid
    )
    complete_provenance = bool(
        provenance_columns_present
        and not as_of_missing.any()
        and not known_at_missing.any()
        and not known_at_invalid.any()
        and not snapshot_source_missing.any()
        and not snapshot_sha_invalid.any()
    )
    status = "FAIL" if hard_fail else (
        "PASS_D_CLOSE_PROVENANCE" if complete_provenance else "PASS_LEGACY_D_CLOSE_ASSUMED"
    )
    return {
        "status": status,
        "contract_version": FEATURE_CONTRACT_VERSION,
        "rows": int(len(data)),
        "feature_count": int(len(feature_cols)),
        "blocked_features": blocked,
        "invalid_d_rows": int(invalid_d.sum()),
        "invalid_t_order_rows": int(invalid_t_order.sum()),
        "invalid_t1_order_rows": int(invalid_t1_order.sum()),
        "feature_as_of_missing_rows": int(as_of_missing.sum()),
        "feature_as_of_future_rows": int(as_of_future.sum()),
        "feature_known_at_missing_rows": int(known_at_missing.sum()),
        "feature_known_at_invalid_rows": int(known_at_invalid.sum()),
        "feature_snapshot_source_missing_rows": int(snapshot_source_missing.sum()),
        "feature_snapshot_sha256_invalid_rows": int(snapshot_sha_invalid.sum()),
    }


def _stable_text(value: object) -> str:
    if value is None or value is pd.NA:
        return "<NA>"
    try:
        if pd.isna(value):
            return "<NA>"
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "<NA>"
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return str(int(value))
    return str(value).strip()


def training_data_fingerprint(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    date_col: str = "d_trade_date",
) -> Tuple[str, str]:
    """Fingerprint the exact D features and matured targets used by training."""
    identity_cols = [
        c for c in (
            date_col, "t_trade_date", "t1_trade_date", "ts_code",
            "feature_as_of_date", "feature_known_at", "feature_snapshot_source",
            "feature_snapshot_sha256",
        ) if c in df.columns
    ]
    target_cols = [c for c in CLASS_TARGETS + REG_TARGETS if c in df.columns]
    ordered_cols = list(dict.fromkeys([*identity_cols, *feature_cols, *target_cols]))
    canonical = df.reindex(columns=ordered_cols).copy()
    for col in canonical.columns:
        canonical[col] = canonical[col].map(_stable_text)
    sort_cols = [c for c in (date_col, "ts_code") if c in canonical.columns]
    if sort_cols:
        canonical = canonical.sort_values(sort_cols, kind="stable")
    feature_digest = hashlib.sha256(
        (FEATURE_CONTRACT_VERSION + "\n" + "\n".join(map(str, feature_cols))).encode("utf-8")
    ).hexdigest()
    digest = hashlib.sha256()
    digest.update(FEATURE_CONTRACT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update("\n".join(ordered_cols).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest(), feature_digest


def purged_walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "d_trade_date",
    n_splits: int = 3,
    embargo_days: int = 2,
    min_train_days: int = 20,
    min_valid_days: int = 3,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]]:
    """Create disjoint expanding-window validation folds with a date embargo."""
    data = df.copy()
    if date_col not in data.columns:
        raise ValueError(f"缺少时间切分字段 {date_col}")
    data[date_col] = _norm_date(data[date_col])
    dates = sorted(data[date_col].dropna().unique())
    if len(dates) < 12:
        raise ValueError(f"walk_forward_days_not_enough:{len(dates)}<12")

    embargo = max(0, int(embargo_days))
    adaptive_min_train = max(6, min(int(min_train_days), len(dates) // 2))
    validation_start = max(adaptive_min_train + embargo, int(len(dates) * 0.50))
    validation_dates = dates[validation_start:]
    possible_splits = max(1, len(validation_dates) // max(1, int(min_valid_days)))
    split_count = min(max(1, int(n_splits)), possible_splits)
    chunks = [list(chunk) for chunk in np.array_split(validation_dates, split_count) if len(chunk)]

    folds: List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]] = []
    date_position = {date: i for i, date in enumerate(dates)}
    for fold_no, valid_dates in enumerate(chunks, start=1):
        first_valid_pos = date_position[valid_dates[0]]
        train_end_pos = first_valid_pos - embargo - 1
        if train_end_pos + 1 < adaptive_min_train:
            continue
        train_dates = dates[: train_end_pos + 1]
        train = data[data[date_col].isin(train_dates)].copy()
        valid = data[data[date_col].isin(valid_dates)].copy()
        if train.empty or valid.empty:
            continue
        boundary = {
            "fold": int(fold_no),
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "valid_start": str(valid_dates[0]),
            "valid_end": str(valid_dates[-1]),
            "train_days": int(len(train_dates)),
            "valid_days": int(len(valid_dates)),
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "embargo_days": int(embargo),
        }
        folds.append((train, valid, boundary))
    if len(folds) < 2:
        raise ValueError(f"walk_forward_folds_not_enough:{len(folds)}<2")
    return folds


def _weighted_fold_mean(rows: Sequence[Dict[str, object]], column: str, weight: str) -> float:
    values: List[float] = []
    weights: List[float] = []
    for row in rows:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        row_weight = pd.to_numeric(pd.Series([row.get(weight)]), errors="coerce").iloc[0]
        if np.isfinite(value) and np.isfinite(row_weight) and row_weight > 0:
            values.append(float(value))
            weights.append(float(row_weight))
    if not values:
        return float("nan")
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def _aggregate_fold_metrics(
    rows: Sequence[Dict[str, object]],
    target: str,
    metric_type: str,
) -> Dict[str, object]:
    usable = [row for row in rows if int(row.get("samples", 0) or 0) > 0]
    samples = int(sum(int(row.get("samples", 0) or 0) for row in usable))
    daily_days = int(sum(int(row.get("daily_metric_days", 0) or 0) for row in usable))
    daily_ic_days = int(sum(int(row.get("daily_ic_days", 0) or 0) for row in usable))
    out: Dict[str, object] = {
        "target": target,
        "type": metric_type,
        "samples": samples,
        "daily_metric_days": daily_days,
        "daily_ic_days": daily_ic_days,
        "walk_forward_folds": int(len(usable)),
        "validation_mode": VALIDATION_MODE,
    }
    if metric_type == "class":
        for col in (
            "positive_rate", "auc", "brier", "baseline_brier", "accuracy",
            "precision_top10", "recall", "calibration_ece",
        ):
            out[col] = _weighted_fold_mean(usable, col, "samples")
        for col in ("daily_precision_top10", "daily_base_rate", "daily_top10_lift"):
            out[col] = _weighted_fold_mean(usable, col, "daily_metric_days")
        for col in ("daily_spearman_ic", "daily_positive_ic_rate"):
            out[col] = _weighted_fold_mean(usable, col, "daily_ic_days")
        brier = float(out.get("brier", np.nan))
        baseline = float(out.get("baseline_brier", np.nan))
        out["brier_skill"] = 1.0 - brier / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
    else:
        for col in ("mae", "spearman_ic", "positive_ic_rate"):
            out[col] = _weighted_fold_mean(usable, col, "samples")
        rmse_sq_rows = [dict(row, rmse_sq=float(row.get("rmse", np.nan)) ** 2) for row in usable]
        rmse_sq = _weighted_fold_mean(rmse_sq_rows, "rmse_sq", "samples")
        out["rmse"] = float(np.sqrt(rmse_sq)) if np.isfinite(rmse_sq) else np.nan
        for col in ("daily_spearman_ic", "daily_positive_ic_rate"):
            out[col] = _weighted_fold_mean(usable, col, "daily_ic_days")
    return out


def _fit_classifier(X: pd.DataFrame, y: pd.Series, min_samples: int = 80):
    yy = pd.to_numeric(y, errors="coerce")
    valid = yy.notna()
    yy = yy[valid].astype(int)
    Xv = X.loc[valid]
    if len(yy) < min_samples or yy.nunique() < 2:
        return None, float(yy.mean()) if len(yy) else 0.0
    if lgb is not None:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=240, learning_rate=0.035, num_leaves=31,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42, verbosity=-1,
        )
    else:
        model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, l2_regularization=1.0, random_state=42)
    model.fit(Xv, yy)
    return model, float(yy.mean())


def _fit_regressor(X: pd.DataFrame, y: pd.Series, min_samples: int = 80):
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = yy.notna()
    yy = yy[valid].astype(float)
    Xv = X.loc[valid]
    if len(yy) < min_samples:
        return None, float(yy.mean()) if len(yy) else 0.0
    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=260, learning_rate=0.035, num_leaves=31,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42, verbosity=-1,
        )
    else:
        model = HistGradientBoostingRegressor(max_iter=240, learning_rate=0.04, l2_regularization=1.0, random_state=42)
    model.fit(Xv, yy)
    return model, float(yy.mean())


def _daily_rank_metrics(prob: pd.Series, actual: pd.Series, dates: pd.Series) -> dict:
    frame = pd.DataFrame({
        "prob": pd.to_numeric(prob, errors="coerce"),
        "actual": pd.to_numeric(actual, errors="coerce"),
        "date": dates.astype(str),
    }).dropna(subset=["prob", "actual"])
    top10_rates: List[float] = []
    base_rates: List[float] = []
    daily_ics: List[float] = []
    for _, day in frame.groupby("date", sort=True):
        if day.empty:
            continue
        top = day.nlargest(min(10, len(day)), "prob")
        top10_rates.append(float(top["actual"].mean()))
        base_rates.append(float(day["actual"].mean()))
        ic = _safe_spearman(day["prob"], day["actual"])
        if np.isfinite(ic):
            daily_ics.append(float(ic))
    return {
        "daily_precision_top10": float(np.mean(top10_rates)) if top10_rates else np.nan,
        "daily_base_rate": float(np.mean(base_rates)) if base_rates else np.nan,
        "daily_top10_lift": float(np.mean(np.asarray(top10_rates) - np.asarray(base_rates))) if top10_rates else np.nan,
        "daily_spearman_ic": float(np.mean(daily_ics)) if daily_ics else np.nan,
        "daily_positive_ic_rate": float(np.mean(np.asarray(daily_ics) > 0)) if daily_ics else np.nan,
        "daily_metric_days": int(len(top10_rates)),
        "daily_ic_days": int(len(daily_ics)),
    }


def _eval_classifier(
    model,
    prior: float,
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    dates: Optional[pd.Series] = None,
) -> dict:
    yy = pd.to_numeric(y, errors="coerce")
    valid = yy.notna()
    yy = yy[valid].astype(int)
    Xv = X.loc[valid]
    if len(yy) == 0:
        return {"target": target, "type": "class", "samples": 0, "positive_rate": np.nan, "auc": np.nan, "brier": np.nan, "baseline_brier": np.nan, "brier_skill": np.nan, "accuracy": np.nan, "precision_top10": np.nan, "recall": np.nan, "calibration_ece": np.nan}
    if model is None:
        prob = np.full(len(yy), prior, dtype=float)
    elif hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xv)[:, 1]
    else:
        prob = np.clip(model.predict(Xv), 0.0, 1.0)
    pred = (prob >= 0.5).astype(int)
    top_n = min(10, len(yy))
    order = np.argsort(-prob)[:top_n] if top_n > 0 else np.array([], dtype=int)
    positives = int(yy.sum())
    ece = _calibration_ece(pd.Series(prob), pd.Series(yy.values))
    brier = float(brier_score_loss(yy, prob))
    baseline_prob = np.full(len(yy), float(np.clip(prior, 0.0, 1.0)), dtype=float)
    baseline_brier = float(brier_score_loss(yy, baseline_prob))
    daily = {}
    if dates is not None:
        date_values = dates.loc[valid]
        daily = _daily_rank_metrics(pd.Series(prob, index=yy.index), yy, date_values)
    return {
        "target": target,
        "type": "class",
        "samples": int(len(yy)),
        "positive_rate": float(yy.mean()) if len(yy) else np.nan,
        "auc": float(roc_auc_score(yy, prob)) if yy.nunique() == 2 else np.nan,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier) if baseline_brier > 0 else np.nan,
        "accuracy": float(accuracy_score(yy, pred)),
        "precision_top10": float(yy.iloc[order].mean()) if len(order) else np.nan,
        "recall": float(yy.iloc[order].sum() / positives) if positives > 0 and len(order) else np.nan,
        "calibration_ece": ece,
        **daily,
    }


def _eval_regressor(
    model,
    mean_value: float,
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    dates: Optional[pd.Series] = None,
) -> dict:
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = yy.notna()
    yy = yy[valid].astype(float)
    Xv = X.loc[valid]
    if len(yy) == 0:
        return {"target": target, "type": "reg", "samples": 0, "mae": np.nan, "rmse": np.nan, "spearman_ic": np.nan, "positive_ic_rate": np.nan}
    pred = np.full(len(yy), mean_value, dtype=float) if model is None else np.asarray(model.predict(Xv), dtype=float)
    ic = _safe_spearman(pd.Series(pred), yy.reset_index(drop=True))
    daily_ics: List[float] = []
    if dates is not None:
        frame = pd.DataFrame({
            "pred": pred,
            "actual": yy.to_numpy(),
            "date": dates.loc[valid].astype(str).to_numpy(),
        })
        for _, day in frame.groupby("date", sort=True):
            day_ic = _safe_spearman(day["pred"], day["actual"])
            if np.isfinite(day_ic):
                daily_ics.append(float(day_ic))
    return {
        "target": target,
        "type": "reg",
        "samples": int(len(yy)),
        "mae": float(mean_absolute_error(yy, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yy, pred))),
        "spearman_ic": ic,
        "positive_ic_rate": float(ic > 0) if np.isfinite(ic) else np.nan,
        "daily_spearman_ic": float(np.mean(daily_ics)) if daily_ics else np.nan,
        "daily_positive_ic_rate": float(np.mean(np.asarray(daily_ics) > 0)) if daily_ics else np.nan,
        "daily_metric_days": int(len(daily_ics)),
        "daily_ic_days": int(len(daily_ics)),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].rank(method="average").corr(pair["y"].rank(method="average"), method="pearson"))


def _calibration_ece(prob: pd.Series, actual: pd.Series, bins: int = 5) -> float:
    p = pd.to_numeric(prob, errors="coerce").clip(0.0, 1.0)
    y = pd.to_numeric(actual, errors="coerce")
    valid = p.notna() & y.notna()
    if not valid.any():
        return float("nan")
    total = int(valid.sum())
    err = 0.0
    for lo in np.linspace(0.0, 1.0, bins, endpoint=False):
        hi = min(1.0, lo + 1.0 / bins)
        m = valid & (p >= lo) & (p < (hi + 1e-9))
        n = int(m.sum())
        if n:
            err += (n / total) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)


def _metric_value(metrics: pd.DataFrame, target: str, col: str) -> float:
    if metrics is None or metrics.empty or col not in metrics.columns:
        return float("nan")
    sub = metrics[metrics["target"].astype(str) == str(target)]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])


def _metric_count(metrics: pd.DataFrame, target: str, col: str) -> int:
    value = _metric_value(metrics, target, col)
    return int(value) if np.isfinite(value) and value > 0 else 0


def _gate_state(quality_ok: bool, metric_days: int, samples: int) -> str:
    if not quality_ok:
        return "SHADOW"
    if metric_days >= 30 and samples >= 600:
        return "ACTIVE"
    if metric_days >= 12 and samples >= 250:
        return "PROVISIONAL"
    return "SHADOW"


def _target_model_gates(
    metrics: pd.DataFrame,
    validation_days: int,
    validation_samples: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    t_limit_auc = _metric_value(metrics, "t_limitup_hit", "auc")
    t_limit_brier_skill = _metric_value(metrics, "t_limitup_hit", "brier_skill")
    t_limit_top10_lift = _metric_value(metrics, "t_limitup_hit", "daily_top10_lift")
    t_limit_daily_ic = _metric_value(metrics, "t_limitup_hit", "daily_spearman_ic")
    t_limit_positive_ic = _metric_value(metrics, "t_limitup_hit", "daily_positive_ic_rate")
    t_limit_days = _metric_count(metrics, "t_limitup_hit", "daily_metric_days")
    t_limit_samples = _metric_count(metrics, "t_limitup_hit", "samples")

    t_touch_auc = _metric_value(metrics, "t_touch_limitup", "auc")
    t_touch_brier_skill = _metric_value(metrics, "t_touch_limitup", "brier_skill")
    t_touch_top10_lift = _metric_value(metrics, "t_touch_limitup", "daily_top10_lift")
    t_touch_daily_ic = _metric_value(metrics, "t_touch_limitup", "daily_spearman_ic")
    t_touch_days = _metric_count(metrics, "t_touch_limitup", "daily_metric_days")
    t_touch_samples = _metric_count(metrics, "t_touch_limitup", "samples")

    t1_accept_auc = _metric_value(metrics, "t1_accept_hit", "auc")
    t1_accept_brier_skill = _metric_value(metrics, "t1_accept_hit", "brier_skill")
    t1_ret_ic = _metric_value(metrics, "t1_close_ret", "daily_spearman_ic")
    t1_positive_ic = _metric_value(metrics, "t1_close_ret", "daily_positive_ic_rate")
    t1_days = min(
        _metric_count(metrics, "t1_accept_hit", "daily_metric_days"),
        _metric_count(metrics, "t1_close_ret", "daily_metric_days"),
    )
    t1_samples = min(
        _metric_count(metrics, "t1_accept_hit", "samples"),
        _metric_count(metrics, "t1_close_ret", "samples"),
    )

    t_ret_ic = _metric_value(metrics, "t_close_ret", "daily_spearman_ic")
    t_ret_positive_ic = _metric_value(metrics, "t_close_ret", "daily_positive_ic_rate")
    t_intraday_ic = _metric_value(metrics, "t_intraday_ret", "daily_spearman_ic")
    t_intraday_positive_ic = _metric_value(metrics, "t_intraday_ret", "daily_positive_ic_rate")
    t_ret_days = min(
        _metric_count(metrics, "t_close_ret", "daily_metric_days"),
        _metric_count(metrics, "t_intraday_ret", "daily_metric_days"),
    )
    t_ret_samples = min(
        _metric_count(metrics, "t_close_ret", "samples"),
        _metric_count(metrics, "t_intraday_ret", "samples"),
    )
    t1_high_ret_ic = _metric_value(metrics, "t1_high_ret", "daily_spearman_ic")
    t1_high_positive_ic = _metric_value(metrics, "t1_high_ret", "daily_positive_ic_rate")
    t1_ret_days = min(
        _metric_count(metrics, "t1_close_ret", "daily_metric_days"),
        _metric_count(metrics, "t1_high_ret", "daily_metric_days"),
    )
    t1_ret_samples = min(
        _metric_count(metrics, "t1_close_ret", "samples"),
        _metric_count(metrics, "t1_high_ret", "samples"),
    )

    t_limit_ok = (
        np.isfinite(t_limit_auc) and t_limit_auc >= 0.52
        and np.isfinite(t_limit_top10_lift) and t_limit_top10_lift >= 0.015
        and np.isfinite(t_limit_daily_ic) and t_limit_daily_ic >= 0.03
        and np.isfinite(t_limit_positive_ic) and t_limit_positive_ic >= 0.55
    )
    t_touch_ok = (
        np.isfinite(t_touch_auc) and t_touch_auc >= 0.52
        and np.isfinite(t_touch_top10_lift) and t_touch_top10_lift >= 0.015
        and np.isfinite(t_touch_daily_ic) and t_touch_daily_ic >= 0.03
    )
    t1_ok = (
        np.isfinite(t1_accept_auc) and t1_accept_auc >= 0.52
        and np.isfinite(t1_ret_ic) and t1_ret_ic >= 0.03
        and np.isfinite(t1_positive_ic) and t1_positive_ic >= 0.55
    )
    t_return_ok = (
        np.isfinite(t_ret_ic) and t_ret_ic >= 0.03
        and np.isfinite(t_ret_positive_ic) and t_ret_positive_ic >= 0.55
        and np.isfinite(t_intraday_ic) and t_intraday_ic >= 0.03
        and np.isfinite(t_intraday_positive_ic) and t_intraday_positive_ic >= 0.55
    )
    t1_return_ok = (
        np.isfinite(t1_ret_ic) and t1_ret_ic >= 0.03
        and np.isfinite(t1_positive_ic) and t1_positive_ic >= 0.55
        and np.isfinite(t1_high_ret_ic) and t1_high_ret_ic >= 0.02
        and np.isfinite(t1_high_positive_ic) and t1_high_positive_ic >= 0.55
    )

    statuses = {
        "t_limit": _gate_state(t_limit_ok, t_limit_days, t_limit_samples),
        "t_touch": _gate_state(t_touch_ok, t_touch_days, t_touch_samples),
        "t1": _gate_state(t1_ok, t1_days, t1_samples),
        "t_return": _gate_state(t_return_ok, t_ret_days, t_ret_samples),
        "t1_return": _gate_state(t1_return_ok, t1_ret_days, t1_ret_samples),
    }
    reasons = {
        "t_limit": (
            f"status={statuses['t_limit']};days={t_limit_days};samples={t_limit_samples};"
            f"validation_days={validation_days};validation_samples={validation_samples};"
            f"t_limit_auc={t_limit_auc};t_limit_brier_skill={t_limit_brier_skill};"
            f"t_limit_top10_lift={t_limit_top10_lift};t_limit_daily_ic={t_limit_daily_ic};"
            f"t_limit_positive_ic_rate={t_limit_positive_ic}"
        ),
        "t_touch": (
            f"status={statuses['t_touch']};days={t_touch_days};samples={t_touch_samples};"
            f"auc={t_touch_auc};brier_skill={t_touch_brier_skill};"
            f"top10_lift={t_touch_top10_lift};daily_ic={t_touch_daily_ic}"
        ),
        "t1": (
            f"status={statuses['t1']};days={t1_days};samples={t1_samples};"
            f"t1_accept_auc={t1_accept_auc};t1_accept_brier_skill={t1_accept_brier_skill};"
            f"t1_ret_daily_ic={t1_ret_ic};t1_positive_ic_rate={t1_positive_ic}"
        ),
        "t_return": (
            f"status={statuses['t_return']};days={t_ret_days};samples={t_ret_samples};"
            f"close_daily_ic={t_ret_ic};close_positive_ic_rate={t_ret_positive_ic};"
            f"intraday_daily_ic={t_intraday_ic};intraday_positive_ic_rate={t_intraday_positive_ic}"
        ),
        "t1_return": (
            f"status={statuses['t1_return']};days={t1_ret_days};samples={t1_ret_samples};"
            f"close_daily_ic={t1_ret_ic};close_positive_ic_rate={t1_positive_ic};"
            f"high_daily_ic={t1_high_ret_ic};high_positive_ic_rate={t1_high_positive_ic}"
        ),
    }
    return statuses, reasons


def _target_probability_gates(
    metrics: pd.DataFrame,
    rank_statuses: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Gate probability levels separately from cross-sectional ranking skill."""
    t_limit_brier_skill = _metric_value(metrics, "t_limitup_hit", "brier_skill")
    t_limit_days = _metric_count(metrics, "t_limitup_hit", "daily_metric_days")
    t_limit_samples = _metric_count(metrics, "t_limitup_hit", "samples")

    t_touch_brier_skill = _metric_value(metrics, "t_touch_limitup", "brier_skill")
    t_touch_days = _metric_count(metrics, "t_touch_limitup", "daily_metric_days")
    t_touch_samples = _metric_count(metrics, "t_touch_limitup", "samples")

    t1_auc = _metric_value(metrics, "t1_up_hit", "auc")
    t1_brier_skill = _metric_value(metrics, "t1_up_hit", "brier_skill")
    t1_top10_lift = _metric_value(metrics, "t1_up_hit", "daily_top10_lift")
    t1_daily_ic = _metric_value(metrics, "t1_up_hit", "daily_spearman_ic")
    t1_positive_ic = _metric_value(metrics, "t1_up_hit", "daily_positive_ic_rate")
    t1_days = _metric_count(metrics, "t1_up_hit", "daily_metric_days")
    t1_samples = _metric_count(metrics, "t1_up_hit", "samples")

    def rank_enabled(target: str) -> bool:
        return str(rank_statuses.get(target, "SHADOW")).upper() in RANK_ENABLED_GATE_STATES

    statuses = {
        "t_limit": _gate_state(
            rank_enabled("t_limit") and np.isfinite(t_limit_brier_skill) and t_limit_brier_skill > 0,
            t_limit_days,
            t_limit_samples,
        ),
        "t_touch": _gate_state(
            rank_enabled("t_touch") and np.isfinite(t_touch_brier_skill) and t_touch_brier_skill > 0,
            t_touch_days,
            t_touch_samples,
        ),
        "t1": _gate_state(
            np.isfinite(t1_auc) and t1_auc >= 0.52
            and np.isfinite(t1_brier_skill) and t1_brier_skill > 0
            and np.isfinite(t1_top10_lift) and t1_top10_lift >= 0.01
            and np.isfinite(t1_daily_ic) and t1_daily_ic >= 0.02
            and np.isfinite(t1_positive_ic) and t1_positive_ic >= 0.55,
            t1_days,
            t1_samples,
        ),
    }
    reasons = {
        "t_limit": (
            f"status={statuses['t_limit']};rank_status={rank_statuses.get('t_limit', 'SHADOW')};"
            f"days={t_limit_days};samples={t_limit_samples};brier_skill={t_limit_brier_skill}"
        ),
        "t_touch": (
            f"status={statuses['t_touch']};rank_status={rank_statuses.get('t_touch', 'SHADOW')};"
            f"days={t_touch_days};samples={t_touch_samples};brier_skill={t_touch_brier_skill}"
        ),
        "t1": (
            f"status={statuses['t1']};days={t1_days};samples={t1_samples};auc={t1_auc};"
            f"brier_skill={t1_brier_skill};top10_lift={t1_top10_lift};"
            f"daily_ic={t1_daily_ic};positive_ic_rate={t1_positive_ic}"
        ),
    }
    return statuses, reasons


def _model_gate(metrics: pd.DataFrame, validation_days: int, validation_samples: int) -> Tuple[bool, str]:
    statuses, reasons = _target_model_gates(metrics, validation_days, validation_samples)
    if statuses["t_limit"] in RANK_ENABLED_GATE_STATES:
        return True, f"rank_enabled_{statuses['t_limit'].lower()}:{reasons['t_limit']}"
    if statuses["t_touch"] in RANK_ENABLED_GATE_STATES:
        return True, f"rank_enabled_touch_{statuses['t_touch'].lower()}:{reasons['t_touch']}"
    return False, (
        "disabled_validation_not_pass:"
        f"{reasons['t_limit']};"
        f"t_touch_status={statuses['t_touch']};"
        f"t1_status={statuses['t1']}"
    )


def fit_limitup_probability_engine(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    valid_ratio: float = 0.30,
    min_samples: int = 80,
) -> LimitupModelBundle:
    data = df.copy()
    for target in CLASS_TARGETS + REG_TARGETS:
        if target not in data.columns:
            data[target] = np.nan
    if "d_trade_date" not in data.columns:
        raise ValueError("缺少时间切分字段 d_trade_date")
    data["d_trade_date"] = _norm_date(data["d_trade_date"])
    folds = purged_walk_forward_splits(
        data,
        date_col="d_trade_date",
        n_splits=3,
        embargo_days=2,
        min_train_days=20,
        min_valid_days=3,
    )
    first_train = folds[0][0]
    inferred_features = feature_cols is None
    if inferred_features:
        feature_cols = infer_feature_cols(first_train)
    requested_features = list(dict.fromkeys(map(str, feature_cols)))
    feature_cols = []
    availability_frame = first_train if inferred_features else data
    for col in requested_features:
        if col not in availability_frame.columns:
            continue
        numeric = pd.to_numeric(availability_frame[col], errors="coerce")
        if numeric.notna().sum() >= max(5, int(len(availability_frame) * 0.03)):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("没有可用数值特征，无法训练概率引擎")

    point_in_time_audit = audit_point_in_time_contract(data, feature_cols)
    if str(point_in_time_audit.get("status", "")).upper() == "FAIL":
        raise ValueError(
            "point_in_time_contract_fail:"
            + json.dumps(point_in_time_audit, ensure_ascii=False, sort_keys=True)
        )
    data_fingerprint, feature_fingerprint = training_data_fingerprint(data, feature_cols)
    feature_availability: Dict[str, Dict[str, object]] = {}
    for col in feature_cols:
        available = pd.to_numeric(data[col], errors="coerce").notna()
        available_dates = data.loc[available, "d_trade_date"].dropna().astype(str)
        feature_availability[col] = {
            "rows": int(available.sum()),
            "days": int(available_dates.nunique()),
            "first_date": str(available_dates.min()) if len(available_dates) else "",
            "last_date": str(available_dates.max()) if len(available_dates) else "",
        }

    fold_metrics: Dict[str, List[Dict[str, object]]] = {
        target: [] for target in CLASS_TARGETS + REG_TARGETS
    }
    fold_boundaries: List[Dict[str, object]] = []
    for train, valid, boundary in folds:
        X_train = _make_X(train, feature_cols)
        X_valid = _make_X(valid, feature_cols)
        fold_boundaries.append(dict(boundary))
        for target in CLASS_TARGETS:
            model, prior = _fit_classifier(X_train, train[target], min_samples=min_samples)
            metric = _eval_classifier(
                model, prior, X_valid, valid[target], target, valid["d_trade_date"]
            )
            metric.update({
                "fold": int(boundary["fold"]),
                "fold_train_end": str(boundary["train_end"]),
                "fold_valid_start": str(boundary["valid_start"]),
                "fold_valid_end": str(boundary["valid_end"]),
            })
            fold_metrics[target].append(metric)
        for target in REG_TARGETS:
            model, mean_value = _fit_regressor(X_train, train[target], min_samples=min_samples)
            metric = _eval_regressor(
                model, mean_value, X_valid, valid[target], target, valid["d_trade_date"]
            )
            metric.update({
                "fold": int(boundary["fold"]),
                "fold_train_end": str(boundary["train_end"]),
                "fold_valid_start": str(boundary["valid_start"]),
                "fold_valid_end": str(boundary["valid_end"]),
            })
            fold_metrics[target].append(metric)

    metric_rows: List[Dict[str, object]] = []
    for target in CLASS_TARGETS:
        metric_rows.append(_aggregate_fold_metrics(fold_metrics[target], target, "class"))
    for target in REG_TARGETS:
        metric_rows.append(_aggregate_fold_metrics(fold_metrics[target], target, "reg"))
    metrics = pd.DataFrame(metric_rows)

    # The walk-forward predictions are used only for model selection.  Once a
    # target passes its independent gate, refit the production candidate on all
    # currently matured history so the newest lawful evidence is not discarded.
    X_full = _make_X(data, feature_cols)
    class_models: Dict[str, object] = {}
    reg_models: Dict[str, object] = {}
    class_priors: Dict[str, float] = {}
    reg_means: Dict[str, float] = {}
    for target in CLASS_TARGETS:
        model, prior = _fit_classifier(X_full, data[target], min_samples=min_samples)
        class_models[target] = model
        class_priors[target] = prior
    for target in REG_TARGETS:
        model, mean_value = _fit_regressor(X_full, data[target], min_samples=min_samples)
        reg_models[target] = model
        reg_means[target] = mean_value

    target_train_end_dates: Dict[str, str] = {}
    for target in CLASS_TARGETS + REG_TARGETS:
        ready = pd.to_numeric(data[target], errors="coerce").notna()
        target_train_end_dates[target] = (
            str(data.loc[ready, "d_trade_date"].max()) if ready.any() else ""
        )
    validation_days = int(sum(int(item["valid_days"]) for item in fold_boundaries))
    validation_samples = int(sum(int(item["valid_rows"]) for item in fold_boundaries))
    train_end = str(data["d_trade_date"].dropna().max())
    valid_start = str(min(item["valid_start"] for item in fold_boundaries))
    target_gate_status, target_gate_reasons = _target_model_gates(metrics, validation_days, validation_samples)
    target_probability_status, target_probability_reasons = _target_probability_gates(
        metrics, target_gate_status
    )
    model_can_rank, model_rank_mode = _model_gate(metrics, validation_days, validation_samples)
    return LimitupModelBundle(
        feature_cols=feature_cols,
        class_models=class_models,
        reg_models=reg_models,
        class_priors=class_priors,
        reg_means=reg_means,
        metrics=metrics,
        train_end_date=train_end,
        valid_start_date=valid_start,
        model_can_rank=model_can_rank,
        model_rank_mode=model_rank_mode,
        validation_days=validation_days,
        validation_samples=validation_samples,
        gate_reason=model_rank_mode,
        gate_version=GATE_VERSION,
        target_gate_status=target_gate_status,
        target_gate_reasons=target_gate_reasons,
        target_probability_status=target_probability_status,
        target_probability_reasons=target_probability_reasons,
        validation_mode=VALIDATION_MODE,
        walk_forward_folds=int(len(fold_boundaries)),
        embargo_days=2,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        data_fingerprint=data_fingerprint,
        feature_fingerprint=feature_fingerprint,
        point_in_time_audit=point_in_time_audit,
        feature_availability=feature_availability,
        target_train_end_dates=target_train_end_dates,
        fold_boundaries=fold_boundaries,
    )


def save_bundle(bundle: LimitupModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = bundle.metrics if isinstance(bundle.metrics, pd.DataFrame) else pd.DataFrame(bundle.metrics)
    # A pandas DataFrame in a joblib artifact is not stable across pandas minor
    # versions (notably StringDtype's constructor). Keep only JSON primitives in
    # the persisted payload and rebuild the frame after loading.
    metric_records = json.loads(metrics.to_json(orient="records"))
    stable_bundle = replace(
        bundle,
        metrics=[],
        artifact_version=BUNDLE_ARTIFACT_VERSION,
    )
    payload = {
        "kind": "premium_limitup_model_bundle",
        "artifact_version": BUNDLE_ARTIFACT_VERSION,
        "bundle": stable_bundle,
        "metric_records": metric_records,
    }
    joblib.dump(payload, path)


def load_bundle(path: Path) -> LimitupModelBundle:
    obj = joblib.load(path)
    if isinstance(obj, dict) and obj.get("kind") == "premium_limitup_model_bundle":
        bundle = obj.get("bundle")
        if not isinstance(bundle, LimitupModelBundle):
            raise TypeError(f"模型文件 bundle 类型不对: {type(bundle)}")
        bundle.metrics = pd.DataFrame(obj.get("metric_records") or [])
        bundle.artifact_version = int(obj.get("artifact_version", BUNDLE_ARTIFACT_VERSION))
        return bundle
    if isinstance(obj, LimitupModelBundle):
        # Legacy same-environment artifact. Contract upgrades force a new stable
        # artifact before production prediction, but retain local compatibility.
        return obj
    raise TypeError(f"模型文件类型不对: {type(obj)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练/应用 Premium 涨停接力专业概率引擎")
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="训练模型并输出校验指标")
    fit.add_argument("--train", required=True, help="含特征+标签的数据文件")
    fit.add_argument("--model-output", required=True, help="模型输出 .joblib")
    fit.add_argument("--metrics-output", required=True, help="校验指标输出 csv/xlsx/parquet")
    fit.add_argument("--feature-cols", default=None, help="逗号分隔特征列；不填则自动识别数值特征")
    fit.add_argument("--valid-ratio", type=float, default=0.25)
    fit.add_argument("--min-samples", type=int, default=80)

    pred = sub.add_parser("predict", help="加载模型，对新样本输出概率")
    pred.add_argument("--model", required=True)
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "fit":
        df = _read_table(Path(args.train))
        feature_cols = None if not args.feature_cols else [x.strip() for x in args.feature_cols.split(",") if x.strip()]
        bundle = fit_limitup_probability_engine(df, feature_cols=feature_cols, valid_ratio=args.valid_ratio, min_samples=args.min_samples)
        save_bundle(bundle, Path(args.model_output))
        _write_table(bundle.metrics, Path(args.metrics_output))
        print(bundle.metrics.to_string(index=False))
        return
    if args.cmd == "predict":
        bundle = load_bundle(Path(args.model))
        df = _read_table(Path(args.input))
        out = bundle.predict(df)
        _write_table(out, Path(args.output))
        return


if __name__ == "__main__":
    main()
