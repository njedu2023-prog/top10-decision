#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium 子系统 — Train（训练闭环｜V3.3：E_ret_plus / EHX 残差增强闭环版）

本文件只负责训练链：
- 保留旧 Premium LR / LGBM 训练链；
- 增强 EHX 残差训练链；
- train 不覆盖 outputs/premium/_last_run.txt，避免覆盖 predict 验收摘要；
- train 状态写入 outputs/premium/_last_train.txt；
- 当 decision 样本不足时，允许从 premium_verify_*.csv 回收已验证样本训练 EHX；
- 增加直接执行入口，方便 workflow / 手工命令验证训练结果。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 允许 python src/top10decision/premium/train.py 直接执行。
if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[2]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    __package__ = "top10decision.premium"

import joblib
import numpy as np
import pandas as pd

from .config import PremiumConfig
from .features import build_features_from_decision_df
from .io import (
    get_commit_sha,
    get_run_id,
    load_decision_inputs,
    utc_now_iso,
)
from .labels import build_premium_labels
from .market_truth import ensure_daily_cached, load_daily
from .model_lgbm import fit_lgbm_regressor, save_lgbm
from .model_lr import build_y_from_real_ret, fit_lr_classifier, save_lr


@dataclass(frozen=True)
class TrainResult:
    trained: bool
    reason: str
    n_samples: int
    n_days: int
    model_version: str


# ========= 基础工具 =========

def _spearman_rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    """简易 Spearman（不依赖 scipy）：先 rank，再计算 pearson。"""
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


def _to_yyyymmdd(s: object) -> str:
    s = str(s).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s[:8]


def _safe_numeric_series(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> pd.Series:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = cols.get(str(name).strip().lower())
        if hit is not None:
            return pd.to_numeric(df[hit], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = cols.get(str(name).strip().lower())
        if hit is not None:
            return str(hit)
    return None


def _read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _json_safe(v):
    """递归转成 JSON 可写类型，避免 numpy/pandas/nan 把 meta 写崩。"""
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(float(v)) else None
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _write_train_state(cfg: PremiumConfig, trade_date: str = "unknown", extra: Optional[Dict] = None) -> Path:
    """
    写训练态摘要，不覆盖 predict 生成的 outputs/premium/_last_run.txt。
    """
    extra = extra or {}
    p = cfg.out_root() / "_last_train.txt"
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "trade_date": str(trade_date),
        "model_version": str(getattr(cfg, "model_version", "premium_v2")),
        "run_id": get_run_id(),
        "commit_sha": get_commit_sha(cfg.repo_root()),
        "created_at_utc": utc_now_iso(),
    }
    data.update(extra)

    lines = [f"{k}: {_json_safe(v)}" for k, v in data.items()]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _append_eval_history_keep_extra(cfg: PremiumConfig, row: Dict) -> Path:
    """追加 learning/premium_eval_history.csv，并保留新增 EHX 字段。"""
    p = cfg.eval_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row])
    if p.exists():
        try:
            df_old = _read_csv_smart(p)
            all_cols = list(df_old.columns)
            for c in df_row.columns:
                if c not in all_cols:
                    all_cols.append(c)
            df_old = df_old.reindex(columns=all_cols)
            df_row = df_row.reindex(columns=all_cols)
            df_new = pd.concat([df_old, df_row], ignore_index=True)
        except Exception:
            df_new = df_row
    else:
        df_new = df_row

    df_new.to_csv(p, index=False, encoding="utf-8-sig")
    return p


# ========= 交易日 / close 真值 =========

def _infer_next_trade_date_by_probe(
    cfg: PremiumConfig,
    trade_date: str,
    max_probe_days: int = 10,
) -> Optional[str]:
    """从 trade_date 次日开始探测第一个可缓存/拉取成功的交易日。"""
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
    """为 labels.build_premium_labels 构造 close_df。"""
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
    close_df["trade_date"] = close_df["trade_date"].astype(str).map(_to_yyyymmdd)
    close_df["ts_code"] = close_df["ts_code"].astype(str).str.strip()
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")
    return close_df, next_td, "ok"


# ========= 样本构建 =========

def _extract_raw_eret_from_decision_df(df_dec: pd.DataFrame) -> pd.Series:
    """从 decision / source 表中尽量提取原始 E_ret。"""
    return _safe_numeric_series(
        df_dec,
        [
            "eret_pred_raw",
            "e_ret_pred_raw",
            "raw_eret_pred",
            "raw_e_ret_pred",
            "eret_pred",
            "e_ret_pred",
            "E_ret",
            "e_ret",
            "eret_pred_final",
            "pred_ret",
            "pred_return",
            "premium_ret",
            "pred_premium_ret",
            "pred_ret_mean",
            "eret_plus",
            "e_ret_plus",
            "E_ret_plus",
            "eret_plus_pred",
            "e_ret_plus_pred",
        ],
        default=np.nan,
    )


def _extract_extra_ehx_inputs(df_dec: pd.DataFrame) -> pd.DataFrame:
    """从原始输入中提取 EHX 可能用到的附加输入。"""
    out = pd.DataFrame(index=df_dec.index)
    out["eret_pred_raw"] = _extract_raw_eret_from_decision_df(df_dec)
    out["p_fill_pred"] = _safe_numeric_series(
        df_dec,
        ["p_fill_pred", "p_fill_pred_final", "p_fill", "P_fill", "dec_p_fill"],
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
        ["ev", "EV", "score_ev", "pred_ev", "final_score", "score"],
        default=np.nan,
    )
    out["turnover_rate"] = _safe_numeric_series(df_dec, ["turnover_rate", "换手率"], default=np.nan)
    out["amount"] = _safe_numeric_series(df_dec, ["amount", "成交额"], default=np.nan)
    out["vol"] = _safe_numeric_series(df_dec, ["vol", "volume", "成交量"], default=np.nan)
    out["close"] = _safe_numeric_series(df_dec, ["close", "close_t", "收盘价"], default=np.nan)
    out["pct_chg"] = _safe_numeric_series(df_dec, ["pct_chg", "pct_change", "涨跌幅"], default=np.nan)
    out["amplitude"] = _safe_numeric_series(df_dec, ["amplitude", "range_1d", "振幅"], default=np.nan)
    return out


def collect_training_samples(cfg: PremiumConfig) -> Tuple[pd.DataFrame, Dict]:
    """从历史 decision 文件中收集可打标样本。"""
    decision_files = load_decision_inputs(cfg)
    stats = {
        "n_decision_files": len(decision_files),
        "pending_days": 0,
        "ok_days": 0,
        "skipped_files": 0,
        "notes": [],
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
            stats["notes"].append(f"trade_date={trade_date} pending_or_fail: {reason}")
            continue

        labels_df, meta = build_premium_labels(close_df, trade_date=trade_date)
        if meta.pending:
            stats["pending_days"] += 1
            stats["notes"].append(f"trade_date={trade_date} label_pending: {getattr(meta, 'reason', '')}")
            continue

        df_join = feat.meta.merge(
            labels_df[["ts_code", "next_trade_date", "real_premium_ret"]],
            on="ts_code",
            how="left",
        )
        df_join = pd.concat([df_join.reset_index(drop=True), feat.risk.reset_index(drop=True)], axis=1)

        X = feat.X.copy()
        X["ts_code"] = feat.meta["ts_code"].astype(str).values
        df_all = df_join.merge(X, on="ts_code", how="left")

        extra_raw = _extract_extra_ehx_inputs(df_dec).copy()
        extra_raw["ts_code"] = feat.meta["ts_code"].astype(str).values
        df_all = df_all.merge(extra_raw, on="ts_code", how="left")

        stats["ok_days"] += 1
        rows.append(df_all)

    if not rows:
        return pd.DataFrame(), stats

    samples = pd.concat(rows, ignore_index=True)
    samples["trade_date"] = samples["trade_date"].astype(str).map(_to_yyyymmdd)
    samples["ts_code"] = samples["ts_code"].astype(str).str.strip()
    samples["real_premium_ret"] = pd.to_numeric(samples["real_premium_ret"], errors="coerce")
    samples["eret_pred_raw"] = pd.to_numeric(samples.get("eret_pred_raw"), errors="coerce")
    samples["delta_ret"] = samples["real_premium_ret"] - samples["eret_pred_raw"]
    return samples, stats


def _filter_recent_days(samples: pd.DataFrame, cfg: PremiumConfig) -> pd.DataFrame:
    """只保留最近 train_window_days 天的样本。"""
    if samples.empty:
        return samples

    dates = sorted(
        [d for d in samples["trade_date"].dropna().unique() if str(d).isdigit() and len(str(d)) == 8]
    )
    if not dates:
        return samples

    keep_dates = dates[-int(getattr(cfg, "train_window_days", 60)):]
    return samples[samples["trade_date"].isin(keep_dates)].reset_index(drop=True)


# ========= premium_verify 回收 EHX 样本 =========

def _infer_trade_date_from_path(path: Path) -> str:
    m = re.search(r"(20\d{6})", path.name)
    return m.group(1) if m else "unknown"


def _real_ret_from_verify_df(df: pd.DataFrame) -> pd.Series:
    y = _safe_numeric_series(
        df,
        [
            "real_premium_ret",
            "real_ret",
            "actual_ret",
            "premium_real_ret",
            "ret_real",
            "y_true",
            "truth_ret",
            "t2_ret",
            "close_t2_ret",
            "r_actual",
        ],
        default=np.nan,
    )
    if y.notna().any():
        return y

    c2 = _safe_numeric_series(df, ["close_T", "close_2", "close_t1", "buy_close", "entry_close", "close_buy"], default=np.nan)
    c3 = _safe_numeric_series(df, ["close_T2_actual", "close_3", "close_t2", "sell_close", "exit_close", "close_sell"], default=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = c3 / c2 - 1.0
    return pd.to_numeric(derived, errors="coerce")


def _collect_ehx_samples_from_verify_outputs(cfg: PremiumConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    从 outputs/premium/premium_verify_*.csv 回收 EHX 已验证样本。

    用途：Decision 历史样本无法打标或原始 E_ret 字段缺失时，仍可用已生成的
    premium_verify 文件训练第一版 EHX，而不是一直停留在 ehx:coldstart_v1。
    """
    out_root = cfg.out_root()
    files = sorted(out_root.glob("premium_verify_*.csv"))
    stats = {
        "n_verify_files": len(files),
        "ok_files": 0,
        "skipped_files": 0,
        "notes": [],
    }
    rows: List[pd.DataFrame] = []

    for path in files:
        try:
            df = _read_csv_smart(path)
        except Exception as e:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip {path.name}: read error: {e}")
            continue

        if df.empty:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip {path.name}: empty")
            continue

        raw = _safe_numeric_series(
            df,
            ["eret_pred_raw", "e_ret_pred_raw", "raw_eret_pred", "e_ret", "E_ret", "eret_pred"],
            default=np.nan,
        )
        real = _real_ret_from_verify_df(df)
        if raw.notna().sum() == 0 or real.notna().sum() == 0:
            stats["skipped_files"] += 1
            stats["notes"].append(
                f"skip {path.name}: missing raw/real columns raw_notna={int(raw.notna().sum())} real_notna={int(real.notna().sum())}"
            )
            continue

        out = pd.DataFrame(index=df.index)
        ts_col = _first_existing_col(df, ["ts_code", "code", "symbol"])
        name_col = _first_existing_col(df, ["name", "名称", "stock_name"])
        td_col = _first_existing_col(df, ["trade_date", "target_date", "date"])
        ntd_col = _first_existing_col(df, ["next_trade_date", "verify_date", "target_date", "t2_date"])

        out["trade_date"] = df[td_col].astype(str).map(_to_yyyymmdd) if td_col else _infer_trade_date_from_path(path)
        out["next_trade_date"] = df[ntd_col].astype(str).map(_to_yyyymmdd) if ntd_col else pd.NA
        out["ts_code"] = df[ts_col].astype(str).str.strip() if ts_col else pd.NA
        out["name"] = df[name_col].astype(str) if name_col else pd.NA
        out["real_premium_ret"] = real
        out["eret_pred_raw"] = raw

        # 只保留预测时也可能有的非未来字段，避免 raw_abs_err/plus_abs_err 等未来泄漏。
        feature_candidates = {
            "p_fill_pred": ["p_fill_pred", "p_fill", "P_fill", "dec_p_fill"],
            "cost_total": ["cost_total", "cost", "trade_cost"],
            "risk_penalty_total": ["risk_penalty_total", "risk_penalty", "risk_score"],
            "ev": ["ev", "EV", "score_ev", "pred_ev", "final_score", "score"],
            "turnover_rate": ["turnover_rate", "换手率"],
            "amount": ["amount", "成交额"],
            "vol": ["vol", "volume", "成交量"],
            "close": ["close", "close_t", "收盘价", "close_T", "close_2", "close_t1"],
            "pct_chg": ["pct_chg", "pct_change", "涨跌幅"],
            "amplitude": ["amplitude", "range_1d", "振幅"],
            "rank_eret_plus": ["rank_eret_plus", "rank", "E_ret_plus排名"],
            "eret_plus_conf_score": ["eret_plus_conf_score", "EHX置信分"],
            "r_p50": ["r_p50"],
            "in_p10": ["in_p10"],
            "in_p50": ["in_p50"],
        }
        for target, candidates in feature_candidates.items():
            s = _safe_numeric_series(df, candidates, default=np.nan)
            if s.isna().all() and target in ("in_p10", "in_p50"):
                col = _first_existing_col(df, candidates)
                if col:
                    s = df[col].astype(str).str.lower().isin(["true", "1", "yes", "是"]).astype(float)
            out[target] = s

        out["delta_ret"] = out["real_premium_ret"] - out["eret_pred_raw"]
        out = out[out["real_premium_ret"].notna() & out["eret_pred_raw"].notna()].reset_index(drop=True)
        if out.empty:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip {path.name}: no valid rows after filter")
            continue

        stats["ok_files"] += 1
        rows.append(out)

    if not rows:
        return pd.DataFrame(), stats

    samples = pd.concat(rows, ignore_index=True)
    samples["trade_date"] = samples["trade_date"].astype(str).map(_to_yyyymmdd)
    samples["ts_code"] = samples["ts_code"].astype(str).str.strip()
    return samples, stats


# ========= EHX 模型 =========

def _build_ehx_feature_cols(samples: pd.DataFrame) -> List[str]:
    """第一版 EHX 特征列。"""
    cols: List[str] = []
    cols.extend([c for c in samples.columns if str(c).startswith("auto__")])

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
        "eret_pred_raw",
        "p_fill_pred",
        "cost_total",
        "risk_penalty_total",
        "ev",
        "close",
        "pct_chg",
        "amplitude",
        "rank_eret_plus",
        "eret_plus_conf_score",
        "r_p50",
        "in_p10",
        "in_p50",
    ]:
        if c in samples.columns:
            cols.append(c)

    uniq: List[str] = []
    seen = set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _prepare_numeric_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df.reindex(columns=feature_cols).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_ehx_delta_regressor(
    X_train: pd.DataFrame,
    y_delta: pd.Series,
    feature_cols: List[str],
):
    """第一版 EHX 模型：HistGradientBoostingRegressor。"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = pd.to_numeric(y_delta, errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        min_samples_leaf=10,
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
    meta_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _ehx_min_samples(cfg: PremiumConfig) -> int:
    # 一份 Top30 verify 文件就能启动第一版 model_v1，避免长期停在 coldstart_v1。
    return max(10, min(30, int(getattr(cfg, "min_train_days", 3)) * 5))


def _train_ehx_only(
    cfg: PremiumConfig,
    ehx_samples: pd.DataFrame,
    reason_prefix: str,
    stats: Optional[Dict] = None,
) -> TrainResult:
    """只训练 EHX；用于旧 LR/LGBM 样本不足但 verify 样本可用的场景。"""
    stats = stats or {}
    ehx_samples = ehx_samples.copy()
    ehx_samples = ehx_samples[
        ehx_samples["real_premium_ret"].notna()
        & ehx_samples["eret_pred_raw"].notna()
    ].reset_index(drop=True)

    n_samples = int(len(ehx_samples))
    n_days = int(ehx_samples["trade_date"].nunique()) if "trade_date" in ehx_samples.columns else 0
    ehx_min_samples = _ehx_min_samples(cfg)
    ehx_model_path = cfg.out_root() / "models" / "ehx_delta.joblib"
    ehx_meta_path = cfg.out_root() / "models" / "ehx_meta.json"

    if n_samples < ehx_min_samples:
        ehx_reason = f"ehx_samples_not_enough:{n_samples}<{ehx_min_samples}"
        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "trained": False,
                "reason": ehx_reason,
                "n_samples": n_samples,
                "n_days": n_days,
                "source": reason_prefix,
                "stats": stats,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
        _write_train_state(
            cfg,
            trade_date="unknown",
            extra={
                "trained": False,
                "reason": ehx_reason,
                "ehx_trained": False,
                "ehx_n_samples": n_samples,
                "ehx_min_samples": ehx_min_samples,
                "ehx_meta_path": str(ehx_meta_path),
            },
        )
        return TrainResult(False, ehx_reason, n_samples, n_days, cfg.model_version)

    ehx_feature_cols = _build_ehx_feature_cols(ehx_samples)
    if not ehx_feature_cols:
        ehx_reason = "ehx_feature_cols_empty"
        _write_train_state(cfg, trade_date="unknown", extra={"trained": False, "reason": ehx_reason})
        return TrainResult(False, ehx_reason, n_samples, n_days, cfg.model_version)

    X_ehx = _prepare_numeric_matrix(ehx_samples, ehx_feature_cols)
    y_delta = pd.to_numeric(ehx_samples["delta_ret"], errors="coerce").fillna(0.0)
    ehx_bundle = fit_ehx_delta_regressor(X_ehx, y_delta, feature_cols=list(X_ehx.columns))
    save_ehx(ehx_bundle, str(ehx_model_path))

    delta_pred = np.asarray(ehx_bundle.predict(X_ehx), dtype=float)
    delta_true = np.asarray(y_delta, dtype=float)
    delta_mae = _mae(delta_true, delta_pred)
    delta_rmse = _rmse(delta_true, delta_pred)

    raw_ret = pd.to_numeric(ehx_samples["eret_pred_raw"], errors="coerce").values
    real_ret = pd.to_numeric(ehx_samples["real_premium_ret"], errors="coerce").values
    plus_ret = raw_ret + delta_pred
    raw_abs_err = np.abs(real_ret - raw_ret)
    plus_abs_err = np.abs(real_ret - plus_ret)
    plus_improve_rate = float(np.mean(plus_abs_err < raw_abs_err)) if len(raw_abs_err) > 0 else float("nan")

    last_td = sorted(ehx_samples["trade_date"].dropna().astype(str).unique())[-1] if n_days > 0 else "unknown"

    _save_ehx_meta(
        ehx_meta_path,
        {
            "kind": "ehx_delta_regressor",
            "model_version": cfg.model_version,
            "trained": True,
            "reason": "ok",
            "source": reason_prefix,
            "feature_cols": list(X_ehx.columns),
            "n_samples": n_samples,
            "n_days": n_days,
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "plus_improve_rate": plus_improve_rate,
            "created_at_utc": utc_now_iso(),
            "commit_sha": get_commit_sha(cfg.repo_root()),
            "run_id": get_run_id(),
        },
    )

    row = {
        "trade_date": str(last_td),
        "next_trade_date": pd.NA,
        "n": n_samples,
        "topk": int(getattr(cfg, "topk", getattr(cfg, "top_n", 30))),
        "hit_rate_at_k": float("nan"),
        "mean_ret_at_k": float("nan"),
        "rank_ic": float("nan"),
        "ehx_trained": 1,
        "ehx_reason": f"ok:{reason_prefix}",
        "ehx_n_samples": n_samples,
        "ehx_min_samples": ehx_min_samples,
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "plus_improve_rate": plus_improve_rate,
        "model_version": cfg.model_version,
        "run_id": get_run_id(),
        "commit_sha": get_commit_sha(cfg.repo_root()),
        "created_at_utc": utc_now_iso(),
    }
    _append_eval_history_keep_extra(cfg, row)

    _write_train_state(
        cfg,
        trade_date=str(last_td),
        extra={
            "trained": True,
            "reason": f"ok:{reason_prefix}",
            "n_samples": n_samples,
            "n_days": n_days,
            "ehx_trained": True,
            "ehx_reason": "ok",
            "ehx_n_samples": n_samples,
            "ehx_min_samples": ehx_min_samples,
            "ehx_model_path": str(ehx_model_path),
            "ehx_meta_path": str(ehx_meta_path),
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "plus_improve_rate": plus_improve_rate,
        },
    )
    return TrainResult(True, f"ok:{reason_prefix}", n_samples, n_days, cfg.model_version)


# ========= 主训练入口 =========

def train_models(cfg: Optional[PremiumConfig] = None) -> TrainResult:
    """训练入口（供 scripts / workflow 调用）。"""
    cfg = cfg or PremiumConfig.load()

    samples, stats = collect_training_samples(cfg)
    verify_samples, verify_stats = _collect_ehx_samples_from_verify_outputs(cfg)

    if samples.empty:
        if not verify_samples.empty:
            return _train_ehx_only(
                cfg,
                verify_samples,
                reason_prefix="ehx_from_premium_verify",
                stats={"decision": stats, "verify": verify_stats},
            )

        ehx_meta_path = cfg.out_root() / "models" / "ehx_meta.json"
        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "trained": False,
                "reason": "no_samples",
                "decision_stats": stats,
                "verify_stats": verify_stats,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
        _write_train_state(
            cfg,
            trade_date="unknown",
            extra={
                "trained": False,
                "reason": "no_samples",
                "pending_days": stats.get("pending_days", 0),
                "ok_days": stats.get("ok_days", 0),
                "skipped_files": stats.get("skipped_files", 0),
                "verify_ok_files": verify_stats.get("ok_files", 0),
                "notes_tail": " | ".join(stats.get("notes", [])[-5:]),
                "ehx_meta_path": str(ehx_meta_path),
            },
        )
        return TrainResult(
            trained=False,
            reason=f"没有可用样本（pending_days={stats['pending_days']}，ok_days={stats['ok_days']}）",
            n_samples=0,
            n_days=0,
            model_version=cfg.model_version,
        )

    samples = _filter_recent_days(samples, cfg)
    samples = samples[samples["real_premium_ret"].notna()].reset_index(drop=True)

    n_samples = int(len(samples))
    n_days = int(samples["trade_date"].nunique()) if "trade_date" in samples.columns else 0

    if n_days < int(getattr(cfg, "min_train_days", 3)):
        if not verify_samples.empty:
            return _train_ehx_only(
                cfg,
                verify_samples,
                reason_prefix="ehx_from_premium_verify_min_days_fallback",
                stats={"decision": stats, "verify": verify_stats, "decision_n_days": n_days},
            )

        last_td = sorted(samples["trade_date"].unique())[-1] if n_days > 0 else "unknown"
        _write_train_state(
            cfg,
            trade_date=str(last_td),
            extra={
                "trained": False,
                "reason": "min_train_days_not_met",
                "n_samples": n_samples,
                "n_days": n_days,
                "min_train_days": int(getattr(cfg, "min_train_days", 3)),
                "pending_days": stats.get("pending_days", 0),
                "ok_days": stats.get("ok_days", 0),
            },
        )
        return TrainResult(
            trained=False,
            reason=f"可训练天数不足：n_days={n_days} < min_train_days={getattr(cfg, 'min_train_days', 3)}",
            n_samples=n_samples,
            n_days=n_days,
            model_version=cfg.model_version,
        )

    # ========== 旧 Premium 训练链（保留） ==========
    feature_cols = [c for c in samples.columns if str(c).startswith("auto__")] + [
        c
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
        feature_cols = [
            c for c in samples.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(samples[c])
        ]

    X_train = _prepare_numeric_matrix(samples, feature_cols)

    y_cls = build_y_from_real_ret(samples["real_premium_ret"], threshold=getattr(cfg, "up_threshold", 0.0))
    lr_bundle = fit_lr_classifier(
        X_train,
        y_cls,
        threshold=getattr(cfg, "up_threshold", 0.0),
        feature_cols=list(X_train.columns),
    )
    save_lr(lr_bundle, str(cfg.lr_model_path()))

    lgbm_bundle = fit_lgbm_regressor(
        X_train,
        samples["real_premium_ret"],
        feature_cols=list(X_train.columns),
        min_samples=max(30, int(getattr(cfg, "min_train_days", 3)) * 5),
    )
    save_lgbm(lgbm_bundle, str(cfg.lgbm_model_path()))

    # ========== EHX 残差训练链 ==========
    ehx_samples = samples.copy()
    ehx_samples = ehx_samples[
        ehx_samples["real_premium_ret"].notna()
        & ehx_samples["eret_pred_raw"].notna()
    ].reset_index(drop=True)

    if not verify_samples.empty:
        ehx_samples = pd.concat([ehx_samples, verify_samples], ignore_index=True, sort=False)
        ehx_samples = ehx_samples.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)

    ehx_feature_cols = _build_ehx_feature_cols(ehx_samples)
    ehx_min_samples = _ehx_min_samples(cfg)
    ehx_trained = False
    ehx_reason = "not_run"
    delta_mae = float("nan")
    delta_rmse = float("nan")

    ehx_model_path = cfg.out_root() / "models" / "ehx_delta.joblib"
    ehx_meta_path = cfg.out_root() / "models" / "ehx_meta.json"
    ehx_bundle = None

    if len(ehx_samples) < ehx_min_samples:
        ehx_reason = f"ehx_samples_not_enough:{len(ehx_samples)}<{ehx_min_samples}"
        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "trained": False,
                "reason": ehx_reason,
                "n_samples": int(len(ehx_samples)),
                "n_days": int(ehx_samples["trade_date"].nunique()) if "trade_date" in ehx_samples.columns else 0,
                "decision_stats": stats,
                "verify_stats": verify_stats,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
    elif not ehx_feature_cols:
        ehx_reason = "ehx_feature_cols_empty"
    else:
        X_ehx = _prepare_numeric_matrix(ehx_samples, ehx_feature_cols)
        y_delta = pd.to_numeric(ehx_samples["delta_ret"], errors="coerce").fillna(0.0)
        ehx_bundle = fit_ehx_delta_regressor(
            X_ehx,
            y_delta,
            feature_cols=list(X_ehx.columns),
        )
        save_ehx(ehx_bundle, str(ehx_model_path))

        delta_pred = np.asarray(ehx_bundle.predict(X_ehx), dtype=float)
        delta_true = np.asarray(y_delta, dtype=float)
        delta_mae = _mae(delta_true, delta_pred)
        delta_rmse = _rmse(delta_true, delta_pred)

        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "trained": True,
                "reason": "ok",
                "source": "decision_plus_verify" if not verify_samples.empty else "decision",
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
        ehx_reason = "ok"

    # ========== 评估：最后一个 trade_date ==========
    last_td = sorted(samples["trade_date"].unique())[-1]
    df_last = samples[samples["trade_date"] == last_td].reset_index(drop=True)

    X_last = _prepare_numeric_matrix(df_last, list(X_train.columns))
    pred_up = lr_bundle.predict_proba(X_last)
    pred_ret = lgbm_bundle.predict(X_last)
    pred_ev = pred_up * pred_ret

    real = pd.to_numeric(df_last["real_premium_ret"], errors="coerce").values
    k = int(getattr(cfg, "topk", getattr(cfg, "top_n", 30)))
    idx = np.argsort(-pred_ev)[: max(1, min(k, len(pred_ev)))]
    real_topk = real[idx]
    hit = float(np.mean(real_topk > 0.0)) if len(real_topk) > 0 else float("nan")
    mean_ret = float(np.nanmean(real_topk)) if len(real_topk) > 0 else float("nan")
    rank_ic = _spearman_rank_ic(pred_ev, real)

    plus_improve_rate = float("nan")
    if ehx_trained and ehx_bundle is not None:
        df_last_ehx = df_last[
            df_last["eret_pred_raw"].notna()
            & df_last["real_premium_ret"].notna()
        ].reset_index(drop=True)
        if not df_last_ehx.empty:
            X_last_ehx = _prepare_numeric_matrix(df_last_ehx, ehx_feature_cols)
            delta_hat_last = np.asarray(ehx_bundle.predict(X_last_ehx), dtype=float)
            raw_ret_last = pd.to_numeric(df_last_ehx["eret_pred_raw"], errors="coerce").values
            real_ret_last = pd.to_numeric(df_last_ehx["real_premium_ret"], errors="coerce").values
            plus_ret_last = raw_ret_last + delta_hat_last

            raw_abs_err = np.abs(real_ret_last - raw_ret_last)
            plus_abs_err = np.abs(real_ret_last - plus_ret_last)
            plus_improve_rate = float(np.mean(plus_abs_err < raw_abs_err)) if len(raw_abs_err) > 0 else float("nan")

    run_id = get_run_id()
    sha = get_commit_sha(cfg.repo_root())
    now = utc_now_iso()

    row = {
        "trade_date": str(last_td),
        "next_trade_date": str(df_last["next_trade_date"].dropna().iloc[0])
        if df_last["next_trade_date"].notna().any()
        else pd.NA,
        "n": int(len(df_last)),
        "topk": int(k),
        "hit_rate_at_k": hit,
        "mean_ret_at_k": mean_ret,
        "rank_ic": rank_ic,
        "ehx_trained": int(bool(ehx_trained)),
        "ehx_reason": ehx_reason,
        "ehx_n_samples": int(len(ehx_samples)),
        "ehx_min_samples": int(ehx_min_samples),
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "plus_improve_rate": plus_improve_rate,
        "model_version": cfg.model_version,
        "run_id": run_id,
        "commit_sha": sha,
        "created_at_utc": now,
    }
    _append_eval_history_keep_extra(cfg, row)

    _write_train_state(
        cfg,
        trade_date=str(last_td),
        extra={
            "trained": True,
            "reason": "ok",
            "n_samples": n_samples,
            "n_days": n_days,
            "pending_days": stats["pending_days"],
            "ok_days": stats["ok_days"],
            "skipped_files": stats.get("skipped_files", 0),
            "verify_ok_files": verify_stats.get("ok_files", 0),
            "ehx_trained": bool(ehx_trained),
            "ehx_reason": ehx_reason,
            "ehx_n_samples": int(len(ehx_samples)),
            "ehx_min_samples": int(ehx_min_samples),
            "ehx_model_path": str(ehx_model_path) if ehx_trained else "",
            "ehx_meta_path": str(ehx_meta_path),
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


def _main() -> int:
    parser = argparse.ArgumentParser(description="Train Premium models, including EHX delta regressor.")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式训练结果")
    args = parser.parse_args()

    cfg = PremiumConfig.load()
    result = train_models(cfg)
    payload = {
        "trained": bool(result.trained),
        "reason": result.reason,
        "n_samples": int(result.n_samples),
        "n_days": int(result.n_days),
        "model_version": result.model_version,
        "last_train": str(cfg.out_root() / "_last_train.txt"),
        "ehx_model": str(cfg.out_root() / "models" / "ehx_delta.joblib"),
        "ehx_meta": str(cfg.out_root() / "models" / "ehx_meta.json"),
    }
    if args.json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(
            "[premium][train] "
            f"trained={payload['trained']} "
            f"n_days={payload['n_days']} "
            f"n_samples={payload['n_samples']} "
            f"reason={payload['reason']}"
        )
        print(f"[premium][train] last_train={payload['last_train']}")
        print(f"[premium][train] ehx_model={payload['ehx_model']}")
        print(f"[premium][train] ehx_meta={payload['ehx_meta']}")
    return 0 if result.trained or "不足" in result.reason or "没有可用样本" in result.reason else 1


__all__ = ["TrainResult", "train_models", "collect_training_samples", "fit_ehx_delta_regressor", "save_ehx"]


if __name__ == "__main__":
    raise SystemExit(_main())
