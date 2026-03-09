#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Train（训练闭环｜V3：E_ret_plus / EHX 残差增强版）

当前主线：
- 不再把 Premium 仅视为“报告层”
- 当前训练目标新增：围绕原始 E_ret 训练二级增强层 EHX
- 第一版技术路线锁定为：
    delta_ret = real_premium_ret - e_ret_pred
    eret_plus = e_ret_pred + delta_hat

保留：
- 旧 Premium LR / LGBM 训练链继续保留，避免一次性推翻旧体系
- Market Truth Layer 仍为唯一 close 真值入口
- 第3日未到（缺 close_3）仍严格 pending，不训练

新增输出：
- outputs/premium/models/ehx_delta.joblib
- outputs/premium/models/ehx_meta.json
- outputs/premium/learning/premium_eval_history.csv（追加 EHX 指标）
- outputs/premium/_last_run.txt（覆盖，新增 EHX 追溯）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import PremiumConfig
from .features import build_features_from_decision_df
from .io import (
    append_eval_history,
    get_commit_sha,
    get_run_id,
    load_decision_inputs,
    utc_now_iso,
    write_last_run,
)
from .labels import build_premium_labels
from .market_truth import ensure_daily_cached, load_daily
from .model_lr import build_y_from_real_ret, fit_lr_classifier, save_lr
from .model_lgbm import fit_lgbm_regressor, save_lgbm


@dataclass(frozen=True)
class TrainResult:
    trained: bool
    reason: str
    n_samples: int
    n_days: int
    model_version: str


# ========= 基础工具 =========

def _spearman_rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    """
    简易 Spearman（不依赖 scipy）：
    - 先 rank，再计算 pearson
    """
    if len(a) < 3:
        return float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return float("nan")

    def rank(x: np.ndarray) -> np.ndarray:
        x2 = np.where(np.isnan(x), -1e18, x)
        order = np.argsort(x2)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x2), dtype=float)
        return r

    ra = rank(a)
    rb = rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _to_yyyymmdd(s: str) -> str:
    s = str(s).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _safe_numeric_series(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> pd.Series:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = cols.get(name.lower())
        if hit is not None:
            return pd.to_numeric(df[hit], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _clip_series(s: pd.Series, lo: Optional[float] = None, hi: Optional[float] = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if lo is not None:
        x = x.clip(lower=lo)
    if hi is not None:
        x = x.clip(upper=hi)
    return x


# ========= 交易日 / close 真值 =========

def _infer_next_trade_date_by_probe(cfg: PremiumConfig, trade_date: str, max_probe_days: int = 10) -> Optional[str]:
    """
    用“探测缓存/拉取是否成功”的方式推断 next_trade_date：
    - 从 trade_date 的次日开始，最多探测 max_probe_days 个自然日
    - 第一个能成功 ensure_daily_cached 的日期，视为 next_trade_date
    """
    import datetime as dt

    trade_date = _to_yyyymmdd(trade_date)
    try:
        d0 = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    except Exception:
        return None

    for i in range(1, int(max_probe_days) + 1):
        d = d0 + dt.timedelta(days=i)
        cand = d.strftime("%Y%m%d")
        r = ensure_daily_cached(cfg, cand)
        if r.ok:
            return cand
    return None


def _build_close_df_for_label(cfg: PremiumConfig, trade_date: str) -> Tuple[pd.DataFrame, Optional[str], str]:
    """
    为 labels.build_premium_labels 构造 close_df（仅包含：trade_date/ts_code/close 的多日表）
    返回：
    - close_df
    - next_trade_date（若可推断到）
    - reason（ok/pending原因）
    """
    trade_date = _to_yyyymmdd(trade_date)

    r2 = ensure_daily_cached(cfg, trade_date)
    if not r2.ok:
        return pd.DataFrame(), None, f"第2日 daily 缓存/拉取失败：{r2.reason}"

    next_td = _infer_next_trade_date_by_probe(cfg, trade_date, max_probe_days=10)
    if not next_td:
        return pd.DataFrame(), None, "找不到 next_trade_date：第3日真实数据尚未到来（正常 pending）"

    df2 = load_daily(cfg, trade_date)[["ts_code", "trade_date", "close"]].copy()
    df3 = load_daily(cfg, next_td)[["ts_code", "trade_date", "close"]].copy()

    close_df = pd.concat([df2, df3], ignore_index=True)
    close_df["trade_date"] = close_df["trade_date"].astype(str)
    close_df["ts_code"] = close_df["ts_code"].astype(str).str.strip()
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")

    return close_df, next_td, "ok"


# ========= 样本构建 =========

def _extract_raw_eret_from_decision_df(df_dec: pd.DataFrame) -> pd.Series:
    """
    从 decision / source 表中尽量提取原始 E_ret。
    这里不要求字段完全统一，尽可能兼容。
    """
    s = _safe_numeric_series(
        df_dec,
        [
            "eret_pred",
            "e_ret_pred",
            "eret_pred_final",
            "e_ret",
            "pred_ret",
            "pred_return",
            "premium_ret",
            "pred_premium_ret",
            "pred_ret_mean",
        ],
        default=np.nan,
    )
    return pd.to_numeric(s, errors="coerce")


def _extract_extra_ehx_inputs(df_dec: pd.DataFrame) -> pd.DataFrame:
    """
    从原始输入中提取 EHX 可能用到的附加输入。
    原则：
    - 不要求全部都有
    - 只做宽松提取，缺失留空，后面统一 fillna
    """
    out = pd.DataFrame(index=df_dec.index)

    out["eret_pred_raw"] = _extract_raw_eret_from_decision_df(df_dec)

    out["p_fill_pred"] = _safe_numeric_series(
        df_dec,
        ["p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill"],
        default=np.nan,
    )

    out["cost_total"] = _safe_numeric_series(
        df_dec,
        ["cost_total", "cost", "cost_value", "cost_all", "trade_cost"],
        default=np.nan,
    )

    out["risk_penalty_total"] = _safe_numeric_series(
        df_dec,
        ["risk_penalty_total", "risk_penalty", "riskpenalty", "risk_score"],
        default=np.nan,
    )

    out["ev"] = _safe_numeric_series(
        df_dec,
        ["ev", "score_ev", "pred_ev", "final_score", "score"],
        default=np.nan,
    )

    out["turnover_rate"] = _safe_numeric_series(df_dec, ["turnover_rate"], default=np.nan)
    out["amount"] = _safe_numeric_series(df_dec, ["amount"], default=np.nan)
    out["vol"] = _safe_numeric_series(df_dec, ["vol", "volume"], default=np.nan)
    out["close"] = _safe_numeric_series(df_dec, ["close", "close_t"], default=np.nan)
    out["pct_chg"] = _safe_numeric_series(df_dec, ["pct_chg", "pct_change"], default=np.nan)
    out["amplitude"] = _safe_numeric_series(df_dec, ["amplitude", "range_1d"], default=np.nan)

    return out


def collect_training_samples(cfg: PremiumConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    从历史 decision 文件中收集“可打标”的样本。
    返回：
    - samples_df：每个股票一行，包含 X 特征列 + meta + risk + real_premium_ret + trade_date/next_trade_date
    - stats：过程统计信息
    """
    decision_files = load_decision_inputs(cfg)

    stats = {
        "n_decision_files": len(decision_files),
        "pending_days": 0,
        "ok_days": 0,
        "skipped_files": 0,
        "notes": [],
        "market_cache_hit": 0,
        "market_fetched": 0,
        "market_failed": 0,
    }

    rows = []

    for item in decision_files:
        df_dec = item.df

        try:
            feat = build_features_from_decision_df(df_dec)
        except Exception as e:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip decision file {item.path.name}: feature error: {e}")
            continue

        trade_date = _to_yyyymmdd(feat.trade_date)

        close_df, next_td, reason = _build_close_df_for_label(cfg, trade_date)
        if close_df.empty:
            stats["pending_days"] += 1
            if "失败" in reason:
                stats["market_failed"] += 1
                stats["notes"].append(f"trade_date={trade_date} market_fail: {reason}")
            continue

        labels_df, meta = build_premium_labels(close_df, trade_date=trade_date)
        if meta.pending:
            stats["pending_days"] += 1
            continue

        df_join = feat.meta.merge(
            labels_df[["ts_code", "next_trade_date", "real_premium_ret"]],
            on="ts_code",
            how="left",
        )
        df_join = pd.concat([df_join.reset_index(drop=True), feat.risk.reset_index(drop=True)], axis=1)

        X = feat.X.copy()
        X["ts_code"] = feat.meta["ts_code"].values
        df_all = df_join.merge(X, on="ts_code", how="left")

        # 新增：补充 EHX 所需原始增强输入
        extra_raw = _extract_extra_ehx_inputs(df_dec).copy()
        extra_raw["ts_code"] = feat.meta["ts_code"].astype(str).values
        df_all = df_all.merge(extra_raw, on="ts_code", how="left")

        stats["ok_days"] += 1
        rows.append(df_all)

    if not rows:
        return pd.DataFrame(), stats

    samples = pd.concat(rows, ignore_index=True)
    samples["trade_date"] = samples["trade_date"].astype(str)
    samples["ts_code"] = samples["ts_code"].astype(str)
    samples["real_premium_ret"] = pd.to_numeric(samples["real_premium_ret"], errors="coerce")

    # EHX 训练目标：delta_ret = y_true - e_ret_pred
    samples["eret_pred_raw"] = pd.to_numeric(samples.get("eret_pred_raw"), errors="coerce")
    samples["delta_ret"] = samples["real_premium_ret"] - samples["eret_pred_raw"]

    return samples, stats


def _filter_recent_days(samples: pd.DataFrame, cfg: PremiumConfig) -> pd.DataFrame:
    """
    只保留最近 train_window_days 天的样本（按 trade_date 排序）。
    """
    if samples.empty:
        return samples
    dates = sorted([d for d in samples["trade_date"].dropna().unique() if str(d).isdigit() and len(str(d)) == 8])
    if not dates:
        return samples
    keep_dates = dates[-int(cfg.train_window_days):]
    return samples[samples["trade_date"].isin(keep_dates)].reset_index(drop=True)


# ========= EHX 模型 =========

def _build_ehx_feature_cols(samples: pd.DataFrame) -> List[str]:
    """
    第一版 EHX 特征列：
    - 原始 premium/features 输出的数值列尽量保留
    - 再叠加少量围绕 E_ret 的增强输入
    """
    cols = []

    # 旧 features.py 的标准输出
    cols.extend([c for c in samples.columns if str(c).startswith("auto__")])

    # 兼容旧 premium 线常见数值特征
    for c in [
        "rank_score",
        "strength_score",
        "theme_boost",
        "probability",
        "final_score",
        "regime_weight",
        "turnover_rate",
        "amount",
        "vol",
    ]:
        if c in samples.columns:
            cols.append(c)

    # EHX 新增围绕 E_ret 的增强输入
    for c in [
        "eret_pred_raw",
        "p_fill_pred",
        "cost_total",
        "risk_penalty_total",
        "ev",
        "close",
        "pct_chg",
        "amplitude",
    ]:
        if c in samples.columns:
            cols.append(c)

    # 去重并保持顺序
    uniq = []
    seen = set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _prepare_numeric_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # 第一版先简单 fillna，保持稳运行
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def fit_ehx_delta_regressor(
    X_train: pd.DataFrame,
    y_delta: pd.Series,
    feature_cols: List[str],
):
    """
    第一版 EHX 模型：
    - 用 sklearn HistGradientBoostingRegressor
    - 原因：依赖轻、稳、能处理非线性，适合第一版残差学习
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = pd.to_numeric(y_delta, errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y)

    class _Bundle:
        def __init__(self, model_obj, cols):
            self.model = model_obj
            self.feature_cols = list(cols)

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return self.model.predict(x)

    return _Bundle(model, feature_cols)


def save_ehx(bundle, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_cols": list(bundle.feature_cols),
            "kind": "ehx_delta_regressor",
        },
        model_path,
    )


def _save_ehx_meta(meta_path: Path, payload: Dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.nanmean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


# ========= 主训练入口 =========

def train_models(cfg: Optional[PremiumConfig] = None) -> TrainResult:
    """
    训练入口（供 scripts / workflow 调用）
    """
    cfg = cfg or PremiumConfig.load()

    samples, stats = collect_training_samples(cfg)
    if samples.empty:
        write_last_run(cfg, trade_date="unknown", extra={"trained": False, "reason": "no_samples"})
        return TrainResult(
            trained=False,
            reason=f"没有可用样本（pending_days={stats['pending_days']}，ok_days={stats['ok_days']})",
            n_samples=0,
            n_days=0,
            model_version=cfg.model_version,
        )

    samples = _filter_recent_days(samples, cfg)
    samples = samples[samples["real_premium_ret"].notna()].reset_index(drop=True)

    n_samples = int(len(samples))
    n_days = int(samples["trade_date"].nunique()) if "trade_date" in samples.columns else 0

    if n_days < int(cfg.min_train_days):
        last_td = sorted(samples["trade_date"].unique())[-1] if n_days > 0 else "unknown"
        write_last_run(cfg, trade_date=str(last_td), extra={"trained": False, "reason": "min_train_days_not_met"})
        return TrainResult(
            trained=False,
            reason=f"可训练天数不足：n_days={n_days} < min_train_days={cfg.min_train_days}",
            n_samples=n_samples,
            n_days=n_days,
            model_version=cfg.model_version,
        )

    # ========== 旧 Premium 训练链（保留） ==========
    feature_cols = [c for c in samples.columns if c.startswith("auto__")] + [
        c for c in [
            "rank_score",
            "strength_score",
            "theme_boost",
            "probability",
            "final_score",
            "regime_weight",
            "turnover_rate",
            "amount",
            "vol",
        ]
        if c in samples.columns
    ]

    if not feature_cols:
        exclude = {
            "trade_date",
            "next_trade_date",
            "ts_code",
            "name",
            "industry",
            "theme",
            "real_premium_ret",
            "close_2",
            "close_3",
            "risk_liquidity",
            "risk_volatility",
            "risk_crowding",
            "risk_event",
            "confidence",
            "fill_risk_hint",
            "eret_pred_raw",
            "delta_ret",
        }
        num_cols = []
        for c in samples.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(samples[c]):
                num_cols.append(c)
        feature_cols = num_cols

    X_train = _prepare_numeric_matrix(samples, feature_cols)

    y_cls = build_y_from_real_ret(samples["real_premium_ret"], threshold=cfg.up_threshold)

    lr_bundle = fit_lr_classifier(
        X_train,
        y_cls,
        threshold=cfg.up_threshold,
        feature_cols=list(X_train.columns),
    )
    save_lr(lr_bundle, str(cfg.lr_model_path()))

    lgbm_bundle = fit_lgbm_regressor(
        X_train,
        samples["real_premium_ret"],
        feature_cols=list(X_train.columns),
        min_samples=max(30, int(cfg.min_train_days) * 5),
    )
    save_lgbm(lgbm_bundle, str(cfg.lgbm_model_path()))

    # ========== 新 EHX 残差训练链 ==========
    ehx_samples = samples.copy()
    ehx_samples = ehx_samples[
        ehx_samples["real_premium_ret"].notna() & ehx_samples["eret_pred_raw"].notna()
    ].reset_index(drop=True)

    ehx_feature_cols = _build_ehx_feature_cols(ehx_samples)
    ehx_trained = False
    delta_mae = float("nan")
    delta_rmse = float("nan")
    ehx_model_path = cfg.out_root() / "models" / "ehx_delta.joblib"
    ehx_meta_path = cfg.out_root() / "models" / "ehx_meta.json"

    if len(ehx_samples) >= max(30, int(cfg.min_train_days) * 5) and ehx_feature_cols:
        X_ehx = _prepare_numeric_matrix(ehx_samples, ehx_feature_cols)
        y_delta = pd.to_numeric(ehx_samples["delta_ret"], errors="coerce").fillna(0.0)

        ehx_bundle = fit_ehx_delta_regressor(
            X_ehx,
            y_delta,
            feature_cols=list(X_ehx.columns),
        )
        save_ehx(ehx_bundle, str(ehx_model_path))

        # 训练集内粗评估（第一版先做轻评估）
        delta_pred = np.asarray(ehx_bundle.predict(X_ehx), dtype=float)
        delta_true = np.asarray(y_delta, dtype=float)

        delta_mae = _mae(delta_true, delta_pred)
        delta_rmse = _rmse(delta_true, delta_pred)

        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "feature_cols": list(X_ehx.columns),
                "n_samples": int(len(ehx_samples)),
                "n_days": int(ehx_samples["trade_date"].nunique()) if "trade_date" in ehx_samples.columns else 0,
                "delta_mae": delta_mae,
                "delta_rmse": delta_rmse,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
        ehx_trained = True

    # ========== 评估：最后一个 trade_date ==========
    last_td = sorted(samples["trade_date"].unique())[-1]
    df_last = samples[samples["trade_date"] == last_td].reset_index(drop=True)

    X_last = _prepare_numeric_matrix(df_last, list(X_train.columns))
    pred_up = lr_bundle.predict_proba(X_last)
    pred_ret = lgbm_bundle.predict(X_last)
    pred_ev = pred_up * pred_ret

    real = pd.to_numeric(df_last["real_premium_ret"], errors="coerce").values

    k = int(cfg.topk)
    idx = np.argsort(-pred_ev)[:max(1, min(k, len(pred_ev)))]
    real_topk = real[idx]
    hit = float(np.mean(real_topk > 0.0)) if len(real_topk) > 0 else float("nan")
    mean_ret = float(np.nanmean(real_topk)) if len(real_topk) > 0 else float("nan")
    rank_ic = _spearman_rank_ic(pred_ev, real)

    # 新增：最后一天的 EHX 粗验证
    plus_improve_rate = float("nan")
    if ehx_trained:
        df_last_ehx = df_last[df_last["eret_pred_raw"].notna() & df_last["real_premium_ret"].notna()].reset_index(drop=True)
        if not df_last_ehx.empty:
            X_last_ehx = _prepare_numeric_matrix(df_last_ehx, ehx_feature_cols)
            delta_hat_last = np.asarray(ehx_bundle.predict(X_last_ehx), dtype=float)
            raw_ret_last = pd.to_numeric(df_last_ehx["eret_pred_raw"], errors="coerce").values
            real_ret_last = pd.to_numeric(df_last_ehx["real_premium_ret"], errors="coerce").values
            plus_ret_last = raw_ret_last + delta_hat_last

            raw_abs_err = np.abs(real_ret_last - raw_ret_last)
            plus_abs_err = np.abs(real_ret_last - plus_ret_last)
            plus_improve_rate = float(np.mean(plus_abs_err < raw_abs_err)) if len(raw_abs_err) > 0 else float("nan")

    # ========== 追溯 ==========
    run_id = get_run_id()
    sha = get_commit_sha(cfg.repo_root())
    now = utc_now_iso()

    row = {
        "trade_date": str(last_td),
        "next_trade_date": str(df_last["next_trade_date"].dropna().iloc[0]) if df_last["next_trade_date"].notna().any() else pd.NA,
        "n": int(len(df_last)),
        "topk": int(k),
        "hit_rate_at_k": hit,
        "mean_ret_at_k": mean_ret,
        "rank_ic": rank_ic,
        "ehx_trained": int(bool(ehx_trained)),
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "plus_improve_rate": plus_improve_rate,
        "model_version": cfg.model_version,
        "run_id": run_id,
        "commit_sha": sha,
        "created_at_utc": now,
    }
    append_eval_history(cfg, row)

    write_last_run(
        cfg,
        trade_date=str(last_td),
        extra={
            "trained": True,
            "n_samples": n_samples,
            "n_days": n_days,
            "pending_days": stats["pending_days"],
            "ok_days": stats["ok_days"],
            "ehx_trained": bool(ehx_trained),
            "ehx_model_path": str(ehx_model_path) if ehx_trained else "",
            "ehx_meta_path": str(ehx_meta_path) if ehx_trained else "",
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "plus_improve_rate": plus_improve_rate,
        },
    )

    return TrainResult(
        trained=True,
        reason="ok",
        n_samples=n_samples,
        n_days=n_days,
        model_version=cfg.model_version,
    )


__all__ = ["TrainResult", "train_models"]
