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
GATE_VERSION = "premium_target_gate_v3_stable_artifact"
BUNDLE_ARTIFACT_VERSION = 2
RANK_ENABLED_GATE_STATES = {"ACTIVE", "PROVISIONAL"}
DEFAULT_EXCLUDE = set(CLASS_TARGETS + REG_TARGETS + [
    "ts_code", "code", "symbol", "名称", "name", "trade_date", "d_trade_date", "t_trade_date", "t1_trade_date",
    "label_valid", "label_matured", "calendar_source", "calendar_status", "calendar_reason", "label_as_of",
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
        out["model_can_rank"] = t_limit_can_rank
        out["model_rank_mode"] = getattr(self, "model_rank_mode", "disabled_validation_not_pass")
        out["model_quality_flag"] = (
            "ok" if t_limit_can_rank and t1_can_rank else "partial" if t_limit_can_rank else "degraded"
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
    can_rank = statuses["t_limit"] in RANK_ENABLED_GATE_STATES
    if can_rank:
        return True, f"rank_enabled_{statuses['t_limit'].lower()}:{reasons['t_limit']}"
    return False, (
        "disabled_validation_not_pass:"
        f"{reasons['t_limit']};"
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
    train, valid, train_end, valid_start = time_split(data, valid_ratio=valid_ratio)
    if feature_cols is None:
        feature_cols = infer_feature_cols(train)
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("没有可用数值特征，无法训练概率引擎")

    X_train = _make_X(train, feature_cols)
    X_valid = _make_X(valid, feature_cols)
    class_models: Dict[str, object] = {}
    reg_models: Dict[str, object] = {}
    class_priors: Dict[str, float] = {}
    reg_means: Dict[str, float] = {}
    metric_rows = []

    for target in CLASS_TARGETS:
        model, prior = _fit_classifier(X_train, train[target], min_samples=min_samples)
        class_models[target] = model
        class_priors[target] = prior
        metric_rows.append(_eval_classifier(model, prior, X_valid, valid[target], target, valid["d_trade_date"]))

    for target in REG_TARGETS:
        model, mean_value = _fit_regressor(X_train, train[target], min_samples=min_samples)
        reg_models[target] = model
        reg_means[target] = mean_value
        metric_rows.append(_eval_regressor(model, mean_value, X_valid, valid[target], target, valid["d_trade_date"]))

    metrics = pd.DataFrame(metric_rows)
    validation_days = int(valid["d_trade_date"].nunique()) if "d_trade_date" in valid.columns else 0
    validation_samples = int(len(valid))
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
