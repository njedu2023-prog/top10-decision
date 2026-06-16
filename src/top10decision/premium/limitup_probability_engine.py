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
from dataclasses import dataclass
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
DEFAULT_EXCLUDE = set(CLASS_TARGETS + REG_TARGETS + [
    "ts_code", "code", "symbol", "名称", "name", "trade_date", "d_trade_date", "t_trade_date", "t1_trade_date",
    "label_valid", "label_matured", "calendar_source", "calendar_status", "calendar_reason", "label_as_of",
])

FEATURE_ALLOW_PREFIXES = [
    "d_", "factor_", "mkt_", "dec_", "eret_", "pfill_", "risk_", "theme_", "amount_",
    "ret_", "vol_", "close_pos_", "rank_", "t_limitup_prob_rule", "t_limitup_strength_rule",
    "t1_continue_up_rate_rule", "limitup_continuation_score_rule",
]
FEATURE_ALLOW_EXACT = {
    "rank", "is_top10", "is_top20", "close_T", "p_premium", "e_premium", "score_ev",
    "confidence", "data_quality", "dec_weight", "dec_rank", "dec_p_fill",
    "eret_pred_raw", "eret_plus_value", "eret_plus_delta", "eret_plus_conf_score",
    "t1_up_rate", "r_p50", "r_p25", "r_p75", "in_p10", "in_p50",
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
        out["model_can_rank"] = bool(getattr(self, "model_can_rank", False))
        out["model_rank_mode"] = getattr(self, "model_rank_mode", "disabled_validation_not_pass")
        out["model_quality_flag"] = "ok" if bool(getattr(self, "model_can_rank", False)) else "degraded"
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


def time_split(df: pd.DataFrame, date_col: str = "d_trade_date", valid_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    data = df.copy()
    if date_col not in data.columns:
        raise ValueError(f"缺少时间切分字段 {date_col}")
    data[date_col] = _norm_date(data[date_col])
    dates = sorted(data[date_col].dropna().unique())
    if len(dates) < 4:
        raise ValueError("可用交易日太少，无法做时间切分验证")
    cut = max(1, int(len(dates) * (1 - valid_ratio)))
    cut = min(cut, len(dates) - 1)
    train_dates = set(dates[:cut])
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


def _eval_classifier(model, prior: float, X: pd.DataFrame, y: pd.Series, target: str) -> dict:
    yy = pd.to_numeric(y, errors="coerce")
    valid = yy.notna()
    yy = yy[valid].astype(int)
    Xv = X.loc[valid]
    if len(yy) == 0:
        return {"target": target, "type": "class", "samples": 0, "positive_rate": np.nan, "auc": np.nan, "brier": np.nan, "accuracy": np.nan, "precision_top10": np.nan, "recall": np.nan, "calibration_ece": np.nan}
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
    return {
        "target": target,
        "type": "class",
        "samples": int(len(yy)),
        "positive_rate": float(yy.mean()) if len(yy) else np.nan,
        "auc": float(roc_auc_score(yy, prob)) if yy.nunique() == 2 else np.nan,
        "brier": float(brier_score_loss(yy, prob)),
        "accuracy": float(accuracy_score(yy, pred)),
        "precision_top10": float(yy.iloc[order].mean()) if len(order) else np.nan,
        "recall": float(yy.iloc[order].sum() / positives) if positives > 0 and len(order) else np.nan,
        "calibration_ece": ece,
    }


def _eval_regressor(model, mean_value: float, X: pd.DataFrame, y: pd.Series, target: str) -> dict:
    yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = yy.notna()
    yy = yy[valid].astype(float)
    Xv = X.loc[valid]
    if len(yy) == 0:
        return {"target": target, "type": "reg", "samples": 0, "mae": np.nan, "rmse": np.nan, "spearman_ic": np.nan, "positive_ic_rate": np.nan}
    pred = np.full(len(yy), mean_value, dtype=float) if model is None else np.asarray(model.predict(Xv), dtype=float)
    ic = _safe_spearman(pd.Series(pred), yy.reset_index(drop=True))
    return {
        "target": target,
        "type": "reg",
        "samples": int(len(yy)),
        "mae": float(mean_absolute_error(yy, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yy, pred))),
        "spearman_ic": ic,
        "positive_ic_rate": float(ic > 0) if np.isfinite(ic) else np.nan,
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


def _model_gate(metrics: pd.DataFrame, validation_days: int, validation_samples: int) -> Tuple[bool, str]:
    t1_accept_auc = _metric_value(metrics, "t1_accept_hit", "auc")
    t1_accept_brier = _metric_value(metrics, "t1_accept_hit", "brier")
    t1_ret_ic = _metric_value(metrics, "t1_close_ret", "spearman_ic")
    ok = (
        int(validation_days) >= 20
        and int(validation_samples) >= 500
        and np.isfinite(t1_accept_auc) and t1_accept_auc >= 0.53
        and np.isfinite(t1_accept_brier) and t1_accept_brier <= 0.25
        and np.isfinite(t1_ret_ic) and t1_ret_ic > 0
    )
    if ok:
        return True, "rank_enabled_validation_pass"
    return False, (
        "disabled_validation_not_pass:"
        f"days={validation_days};samples={validation_samples};"
        f"t1_accept_auc={t1_accept_auc};t1_accept_brier={t1_accept_brier};"
        f"t1_ret_spearman_ic={t1_ret_ic}"
    )


def fit_limitup_probability_engine(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    valid_ratio: float = 0.25,
    min_samples: int = 80,
) -> LimitupModelBundle:
    data = df.copy()
    if "label_matured" in data.columns:
        data = data[pd.to_numeric(data["label_matured"], errors="coerce").fillna(0).astype(int) == 1].copy()
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
        metric_rows.append(_eval_classifier(model, prior, X_valid, valid[target], target))

    for target in REG_TARGETS:
        model, mean_value = _fit_regressor(X_train, train[target], min_samples=min_samples)
        reg_models[target] = model
        reg_means[target] = mean_value
        metric_rows.append(_eval_regressor(model, mean_value, X_valid, valid[target], target))

    metrics = pd.DataFrame(metric_rows)
    validation_days = int(valid["d_trade_date"].nunique()) if "d_trade_date" in valid.columns else 0
    validation_samples = int(len(valid))
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
    )


def save_bundle(bundle: LimitupModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> LimitupModelBundle:
    obj = joblib.load(path)
    if not isinstance(obj, LimitupModelBundle):
        raise TypeError(f"模型文件类型不对: {type(obj)}")
    return obj


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
