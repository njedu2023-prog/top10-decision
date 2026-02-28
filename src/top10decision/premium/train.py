#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Train（训练闭环）

P0 目标：
- 用历史 decision 表 + close 真值表，训练：
  1) LR 分类头：P_win = P(real_premium_ret > 0)
  2) LGBM 回归头：E_ret = E(real_premium_ret)
- 严格 pending：第3日未到（缺 close_3）不训练、不评估
- 输出：
  - outputs/premium/models/premium_lr.joblib
  - outputs/premium/models/premium_lgbm.joblib
  - outputs/premium/learning/premium_eval_history.csv（追加一行）
  - outputs/premium/_last_run.txt（覆盖）

说明：
- trade_date 语义：第2日（decision 表对应日）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PremiumConfig
from .features import build_features_from_decision_df
from .io import (
    append_eval_history,
    get_commit_sha,
    get_run_id,
    load_close_table,
    load_decision_inputs,
    utc_now_iso,
    write_last_run,
)
from .labels import build_premium_labels
from .model_lr import build_y_from_real_ret, fit_lr_classifier, save_lr
from .model_lgbm import fit_lgbm_regressor, save_lgbm


@dataclass(frozen=True)
class TrainResult:
    trained: bool
    reason: str
    n_samples: int
    n_days: int
    model_version: str


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
        # nan 先填极小值，保证可排序
        x2 = np.where(np.isnan(x), -1e18, x)
        order = np.argsort(x2)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x2), dtype=float)
        return r

    ra = rank(a)
    rb = rank(b)
    # pearson
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def collect_training_samples(cfg: PremiumConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    从历史 decision 文件中收集“可打标”的样本。
    返回：
    - samples_df：每个股票一行，包含 X 特征列 + meta + risk + real_premium_ret + trade_date/next_trade_date
    - stats：过程统计信息
    """
    decision_files = load_decision_inputs(cfg)
    close_df = load_close_table(cfg)

    stats = {
        "n_decision_files": len(decision_files),
        "n_close_rows": int(len(close_df)),
        "pending_days": 0,
        "ok_days": 0,
        "skipped_files": 0,
        "notes": [],
    }

    rows = []

    # 为了“训练窗口按天”，先按文件逐个处理
    for item in decision_files:
        df_dec = item.df
        try:
            feat = build_features_from_decision_df(df_dec)
        except Exception as e:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip decision file {item.path.name}: feature error: {e}")
            continue

        trade_date = feat.trade_date

        # 打标：RealPremiumRet(2→3)
        labels_df, meta = build_premium_labels(close_df, trade_date=trade_date)
        if meta.pending:
            stats["pending_days"] += 1
            continue

        # 将标签合并到 meta/risk/X
        # labels_df: trade_date,next_trade_date,ts_code,close_2,close_3,real_premium_ret
        df_join = feat.meta.merge(
            labels_df[["ts_code", "next_trade_date", "real_premium_ret"]],
            on="ts_code",
            how="left",
        )
        df_join = pd.concat([df_join.reset_index(drop=True), feat.risk.reset_index(drop=True)], axis=1)

        # 合并特征 X
        X = feat.X.copy()
        X["ts_code"] = feat.meta["ts_code"].values
        df_all = df_join.merge(X, on="ts_code", how="left")

        stats["ok_days"] += 1
        rows.append(df_all)

    if not rows:
        return pd.DataFrame(), stats

    samples = pd.concat(rows, ignore_index=True)

    # 清理：必须有 trade_date/ts_code/real_premium_ret 才算训练样本
    samples["trade_date"] = samples["trade_date"].astype(str)
    samples["ts_code"] = samples["ts_code"].astype(str)
    samples["real_premium_ret"] = pd.to_numeric(samples["real_premium_ret"], errors="coerce")

    return samples, stats


def _filter_recent_days(samples: pd.DataFrame, cfg: PremiumConfig) -> pd.DataFrame:
    """
    只保留最近 train_window_days 天的样本（按 trade_date 排序）。
    注意：train_window_days 是“天数上限”，不是行数上限。
    """
    if samples.empty:
        return samples
    # 仅保留合法日期
    dates = sorted([d for d in samples["trade_date"].dropna().unique() if str(d).isdigit() and len(str(d)) == 8])
    if not dates:
        return samples
    keep_dates = dates[-int(cfg.train_window_days):]
    return samples[samples["trade_date"].isin(keep_dates)].reset_index(drop=True)


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
    # 只保留 real_premium_ret 非空
    samples = samples[samples["real_premium_ret"].notna()].reset_index(drop=True)

    n_samples = int(len(samples))
    n_days = int(samples["trade_date"].nunique()) if "trade_date" in samples.columns else 0

    if n_days < int(cfg.min_train_days):
        # 不够天数，不训练（避免噪声 + 过拟合）
        last_td = sorted(samples["trade_date"].unique())[-1] if n_days > 0 else "unknown"
        write_last_run(cfg, trade_date=str(last_td), extra={"trained": False, "reason": "min_train_days_not_met"})
        return TrainResult(
            trained=False,
            reason=f"可训练天数不足：n_days={n_days} < min_train_days={cfg.min_train_days}",
            n_samples=n_samples,
            n_days=n_days,
            model_version=cfg.model_version,
        )

    # 训练集特征列：从 X 的列推断（features.py 输出是标准化后的数值列）
    # 我们用 “rank_score/strength_score/...” + auto__* 这些列
    feature_cols = [c for c in samples.columns if c.startswith("auto__")] + [
        c for c in ["rank_score", "strength_score", "theme_boost", "probability", "final_score", "regime_weight",
                    "turnover_rate", "amount", "vol"]
        if c in samples.columns
    ]
    # 如果 feature_cols 为空，尝试兜底：所有 float/int 列（排除 label/meta）
    if not feature_cols:
        exclude = {
            "trade_date", "next_trade_date", "ts_code", "name", "industry", "theme",
            "real_premium_ret", "close_2", "close_3",
            "risk_liquidity", "risk_volatility", "risk_crowding", "risk_event", "confidence",
            "fill_risk_hint",
        }
        num_cols = []
        for c in samples.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(samples[c]):
                num_cols.append(c)
        feature_cols = num_cols

    X_train = samples[feature_cols].copy()
    # 防御性：把非数值强转
    for c in X_train.columns:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)

    # y 分类：real_premium_ret > threshold
    y_cls = build_y_from_real_ret(samples["real_premium_ret"], threshold=cfg.up_threshold)

    # 训练 LR（分类头）
    lr_bundle = fit_lr_classifier(X_train, y_cls, threshold=cfg.up_threshold, feature_cols=list(X_train.columns))
    save_lr(lr_bundle, str(cfg.lr_model_path()))

    # 训练 LGBM（回归头）
    lgbm_bundle = fit_lgbm_regressor(
        X_train,
        samples["real_premium_ret"],
        feature_cols=list(X_train.columns),
        min_samples=max(30, int(cfg.min_train_days) * 5),
    )
    save_lgbm(lgbm_bundle, str(cfg.lgbm_model_path()))

    # 评估：用最后一个 trade_date 的样本做一次 TopK 指标（粗评估）
    last_td = sorted(samples["trade_date"].unique())[-1]
    df_last = samples[samples["trade_date"] == last_td].reset_index(drop=True)
    X_last = df_last[list(X_train.columns)].copy()
    for c in X_last.columns:
        X_last[c] = pd.to_numeric(X_last[c], errors="coerce").fillna(0.0)

    pred_up = lr_bundle.predict_proba(X_last)
    pred_ret = lgbm_bundle.predict(X_last)
    pred_ev = pred_up * pred_ret

    real = pd.to_numeric(df_last["real_premium_ret"], errors="coerce").values
    # 取 TopK
    k = int(cfg.topk)
    idx = np.argsort(-pred_ev)[:max(1, min(k, len(pred_ev)))]
    real_topk = real[idx]
    hit = float(np.mean(real_topk > 0.0)) if len(real_topk) > 0 else float("nan")
    mean_ret = float(np.nanmean(real_topk)) if len(real_topk) > 0 else float("nan")
    rank_ic = _spearman_rank_ic(pred_ev, real)

    # 追溯字段
    run_id = get_run_id()
    sha = get_commit_sha(cfg.repo_root())
    now = utc_now_iso()

    # 写 eval_history
    row = {
        "trade_date": str(last_td),
        "next_trade_date": str(df_last["next_trade_date"].dropna().iloc[0]) if df_last["next_trade_date"].notna().any() else pd.NA,
        "n": int(len(df_last)),
        "topk": int(k),
        "hit_rate_at_k": hit,
        "mean_ret_at_k": mean_ret,
        "rank_ic": rank_ic,
        "model_version": cfg.model_version,
        "run_id": run_id,
        "commit_sha": sha,
        "created_at_utc": now,
    }
    append_eval_history(cfg, row)

    # 写 last_run
    write_last_run(
        cfg,
        trade_date=str(last_td),
        extra={
            "trained": True,
            "n_samples": n_samples,
            "n_days": n_days,
            "pending_days": stats["pending_days"],
            "ok_days": stats["ok_days"],
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
