#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Predict（推理与排序落盘）

职责：
- 读取最新 decision 预测表（第2日表）
- 构建特征
- 加载 LR + LGBM 模型
- 输出 Premium 排序表（CSV + MD）
- 可选：若 close 真值已存在（第3日已到），则附带 real_premium_ret 供验证

注意：
- 本模块不做 P_fill，不考虑买不到，只输出“预测溢价收益排序 + 风险提示”。
- P1：close 真值来源切换到 Market Truth Layer（data/market/daily_YYYYMMDD.csv），并保持 labels.py 的 pending 语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import PremiumConfig
from .features import build_features_from_decision_df
from .io import (
    get_commit_sha,
    get_run_id,
    load_decision_inputs,
    utc_now_iso,
    write_last_run,
    write_rank_csv,
    write_rank_md,
)
from .labels import build_premium_labels
from .market_truth import ensure_daily_cached, load_daily
from .model_lr import LRModelBundle, load_lr
from .model_lgbm import LGBMRegBundle, load_lgbm
from .report_md import render_premium_md
from .schemas import PremiumRankOutputSchema


@dataclass(frozen=True)
class PredictResult:
    ok: bool
    trade_date: str
    rank_csv: Optional[str]
    rank_md: Optional[str]
    reason: str


def _safe_load_lr(path: str) -> Optional[LRModelBundle]:
    try:
        return load_lr(path)
    except Exception:
        return None


def _safe_load_lgbm(path: str) -> Optional[LGBMRegBundle]:
    try:
        return load_lgbm(path)
    except Exception:
        return None


def _constant_lr(feature_cols: list, p: float = 0.5) -> LRModelBundle:
    return LRModelBundle(model=None, pos_rate=float(p), threshold=0.0, feature_cols=list(feature_cols))


def _constant_lgbm(feature_cols: list, m: float = 0.0) -> LGBMRegBundle:
    return LGBMRegBundle(model=None, y_mean=float(m), feature_cols=list(feature_cols))


def _to_yyyymmdd(s: str) -> str:
    s = str(s).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


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


def _build_close_df_for_label(cfg: PremiumConfig, trade_date: str) -> Tuple[pd.DataFrame, str]:
    """
    为 labels.build_premium_labels 构造 close_df（仅包含：trade_date/ts_code/close 的多日表）
    返回：
    - close_df（可能为空）
    - reason
    """
    trade_date = _to_yyyymmdd(trade_date)

    # 第2日必须可用
    r2 = ensure_daily_cached(cfg, trade_date)
    if not r2.ok:
        return pd.DataFrame(), f"第2日 daily 缓存/拉取失败：{r2.reason}"

    # 推断第3日
    next_td = _infer_next_trade_date_by_probe(cfg, trade_date, max_probe_days=10)
    if not next_td:
        return pd.DataFrame(), "找不到 next_trade_date：第3日真实数据尚未到来（正常 pending）"

    # 读第2日/第3日 daily，并拼成 close_df
    df2 = load_daily(cfg, trade_date)[["ts_code", "trade_date", "close"]].copy()
    df3 = load_daily(cfg, next_td)[["ts_code", "trade_date", "close"]].copy()
    close_df = pd.concat([df2, df3], ignore_index=True)

    close_df["trade_date"] = close_df["trade_date"].astype(str)
    close_df["ts_code"] = close_df["ts_code"].astype(str).str.strip()
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")

    return close_df, "ok"


def predict_latest(cfg: Optional[PremiumConfig] = None) -> PredictResult:
    cfg = cfg or PremiumConfig.load()

    decision_files = load_decision_inputs(cfg)
    if not decision_files:
        write_last_run(cfg, trade_date="unknown", extra={"ok": False, "reason": "no_decision_files"})
        return PredictResult(False, "unknown", None, None, "未找到 decision 输入文件")

    # 取最新（按 io.load_decision_inputs 已排序）
    latest = decision_files[-1]
    try:
        feat = build_features_from_decision_df(latest.df)
    except Exception as e:
        write_last_run(cfg, trade_date="unknown", extra={"ok": False, "reason": f"feature_error:{e}"})
        return PredictResult(False, "unknown", None, None, f"特征构建失败：{e}")

    trade_date = _to_yyyymmdd(feat.trade_date)
    X = feat.X.copy()
    meta = feat.meta.copy()
    risk = feat.risk.copy()

    # 加载模型（若不存在则降级常数预测）
    lr = _safe_load_lr(str(cfg.lr_model_path()))
    lgbm = _safe_load_lgbm(str(cfg.lgbm_model_path()))

    feature_cols = list(X.columns)
    if lr is None:
        lr = _constant_lr(feature_cols, p=0.5)
    if lgbm is None:
        lgbm = _constant_lgbm(feature_cols, m=0.0)

    # 推理
    pred_up_prob = lr.predict_proba(X)
    pred_ret_mean = lgbm.predict(X)
    pred_ev = pred_up_prob * pred_ret_mean

    # rank：1 开始
    order = np.argsort(-pred_ev)
    rank_pred_ev = np.empty_like(order, dtype=int)
    rank_pred_ev[order] = np.arange(1, len(order) + 1)

    # 可选：附带真实对照（如果第3日 close 已可用）
    real_ret = pd.Series([pd.NA] * len(meta))
    next_trade_date = pd.Series([pd.NA] * len(meta))

    close_df, close_reason = _build_close_df_for_label(cfg, trade_date)
    if close_df is not None and not close_df.empty:
        labels_df, label_meta = build_premium_labels(close_df, trade_date=trade_date)
        if not label_meta.pending and not labels_df.empty:
            tmp = meta[["ts_code"]].merge(
                labels_df[["ts_code", "next_trade_date", "real_premium_ret"]],
                on="ts_code",
                how="left",
            )
            next_trade_date = tmp["next_trade_date"]
            real_ret = tmp["real_premium_ret"]

    # 追溯字段
    run_id = get_run_id()
    sha = get_commit_sha(cfg.repo_root())
    now = utc_now_iso()

    # 组装输出表
    out = pd.DataFrame({
        "trade_date": meta["trade_date"].astype(str),
        "next_trade_date": next_trade_date,
        "ts_code": meta["ts_code"].astype(str),
        "name": meta.get("name", pd.Series([pd.NA] * len(meta))),
        "decision_rank": meta.get("decision_rank", pd.Series([pd.NA] * len(meta))),
        "strength_score": meta.get("strength_score", pd.Series([pd.NA] * len(meta))),
        "theme_boost": meta.get("theme_boost", pd.Series([pd.NA] * len(meta))),
        "probability": meta.get("probability", pd.Series([pd.NA] * len(meta))),
        "final_score": meta.get("final_score", pd.Series([pd.NA] * len(meta))),
        "pred_up_prob": pred_up_prob,
        "pred_ret_mean": pred_ret_mean,
        "pred_ev": pred_ev,
        "rank_pred_ev": rank_pred_ev,
        "risk_liquidity": risk.get("risk_liquidity", pd.Series([pd.NA] * len(meta))),
        "risk_volatility": risk.get("risk_volatility", pd.Series([pd.NA] * len(meta))),
        "risk_crowding": risk.get("risk_crowding", pd.Series([pd.NA] * len(meta))),
        "risk_event": risk.get("risk_event", pd.Series([pd.NA] * len(meta))),
        "fill_risk_hint": meta.get("fill_risk_hint", pd.Series([pd.NA] * len(meta))),
        "confidence": risk.get("confidence", pd.Series([pd.NA] * len(meta))),
        "real_premium_ret": real_ret,
        "run_id": run_id,
        "commit_sha": sha,
        "model_version": cfg.model_version,
        "data_snapshot": pd.NA,
        "created_at_utc": now,
    })

    # 规范输出列顺序（schemas 定义）
    for c in PremiumRankOutputSchema.COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[list(PremiumRankOutputSchema.COLUMNS)]

    # 排序输出（按 pred_ev）
    out = out.sort_values(by=["pred_ev"], ascending=False).reset_index(drop=True)

    # 写 CSV/MD
    p_csv = write_rank_csv(cfg, trade_date=trade_date, df_rank=out)
    md_text = render_premium_md(out, cfg, trade_date=trade_date)
    p_md = write_rank_md(cfg, trade_date=trade_date, md_text=md_text)

    # last_run
    write_last_run(
        cfg,
        trade_date=trade_date,
        extra={
            "ok": True,
            "rows": int(len(out)),
            "rank_csv": str(p_csv.name),
            "rank_md": str(p_md.name),
            # 记录一下真值可用性（不影响契约）
            "truth_reason": close_reason,
        },
    )

    return PredictResult(True, trade_date, str(p_csv), str(p_md), "ok")


__all__ = ["PredictResult", "predict_latest"]
