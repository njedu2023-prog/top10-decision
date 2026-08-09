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
import datetime as dt
import hashlib
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
from .limitup_probability_engine import (
    BUNDLE_ARTIFACT_VERSION,
    GATE_VERSION,
    fit_limitup_probability_engine,
    load_bundle as load_limitup_probability_bundle,
    save_bundle as save_limitup_probability_bundle,
)
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


def _d_close_numeric_series(df: pd.DataFrame, default: float = np.nan) -> pd.Series:
    """Read the D-close snapshot without conflating close_T with future close_t."""
    for name in ("close_T", "d_close", "close", "收盘价"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _price_numeric_series(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> pd.Series:
    """解析价格列，兼容 “≤11.97”“冲高至12.78” 这类展示字符串。"""
    col = _first_existing_col(df, candidates)
    if not col:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")

    num = pd.to_numeric(df[col], errors="coerce")
    raw = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
    extracted = raw.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    parsed = pd.to_numeric(extracted, errors="coerce")
    return num.where(num.notna(), parsed).fillna(default)


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
            "t1_close_ret",
            "premium_ret_t1_to_t2",
            "realized_ret_t1_to_t2",
            "real_premium_ret",
            "real_ret",
            "premium_real_ret",
        ],
        default=np.nan,
    )
    if y.notna().any():
        return y

    c2 = _safe_numeric_series(df, ["entry_price_proxy", "open_T_actual", "t_open", "open_t"], default=np.nan)
    c3 = _safe_numeric_series(df, ["close_T2_actual", "close_3", "close_t2", "sell_close", "exit_close", "close_sell"], default=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = c3 / c2 - 1.0
    return pd.to_numeric(derived, errors="coerce")


def _bool_numeric_series(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> pd.Series:
    """把 verify 里的 是/否、true/false、0/1 统一转为 0/1。"""
    col = _first_existing_col(df, candidates)
    if not col:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")

    num = pd.to_numeric(df[col], errors="coerce")
    if num.notna().any():
        return num.clip(lower=0, upper=1)

    raw = df[col].astype(str).str.strip().str.lower()
    true_set = {"1", "true", "yes", "y", "是", "命中", "hit", "up"}
    false_set = {"0", "false", "no", "n", "否", "未命中", "miss", "down"}
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[raw.isin(true_set)] = 1.0
    out.loc[raw.isin(false_set)] = 0.0
    return out.fillna(default)


def _norm_ts_code_for_market(x: object) -> str:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if "." in s:
        left, right = s.split(".", 1)
        return f"{''.join(ch for ch in left if ch.isdigit()).zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
        if digits.startswith(("60", "68", "90")):
            return f"{digits}.SH"
        if digits.startswith(("00", "30", "20")):
            return f"{digits}.SZ"
        if digits.startswith(("43", "83", "87", "88", "92")):
            return f"{digits}.BJ"
        return digits
    return s


def _limit_rate_for_code(ts_code: object) -> float:
    code = _norm_ts_code_for_market(ts_code)
    raw = code.split(".")[0]
    suffix = code.split(".")[-1] if "." in code else ""
    if suffix == "BJ" or raw.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if raw.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def _valid_trade_date(x: object) -> Optional[str]:
    s = _to_yyyymmdd(x)
    return s if re.fullmatch(r"20\d{6}", s) else None


def _first_nonempty_date(df: pd.DataFrame, candidates: List[str], fallback: str = "") -> Optional[str]:
    col = _first_existing_col(df, candidates)
    if col:
        for v in df[col].dropna().astype(str).tolist():
            hit = _valid_trade_date(v)
            if hit:
                return hit
    return _valid_trade_date(fallback) if fallback else None


def _infer_next_market_trade_dates(
    cfg: PremiumConfig,
    trade_date: str,
    max_probe_days: int = 20,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    用真实 daily 数据存在性探测 D 后两个 A 股交易日。

    训练回填阶段用于历史文件修复：只接受有全市场 daily 的日期，避免把周末/节假日误当交易日。
    """
    trade_date = _to_yyyymmdd(trade_date)
    try:
        d0 = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    except Exception:
        return None, None, f"bad_trade_date:{trade_date}"

    found: List[str] = []
    reasons: List[str] = []
    for i in range(1, int(max_probe_days) + 1):
        cand = (d0 + dt.timedelta(days=i)).strftime("%Y%m%d")
        r = ensure_daily_cached(cfg, cand)
        if r.ok:
            found.append(cand)
            if len(found) >= 2:
                return found[0], found[1], "ok:market_daily_probe"
        else:
            if len(reasons) < 3:
                reasons.append(f"{cand}:{r.reason}")
    return (found[0] if found else None), None, "next_trade_dates_not_ready:" + "|".join(reasons)


def _has_complete_limitup_truth(df: pd.DataFrame) -> bool:
    ready = _bool_numeric_series(df, ["t_limitup_verify_ready", "label_matured"], default=np.nan)
    t_up = _bool_numeric_series(df, ["t_up_actual", "t_up_hit"], default=np.nan)
    t_limitup = _bool_numeric_series(df, ["t_limitup_actual", "t_limitup_hit"], default=np.nan)
    t_touch = _bool_numeric_series(df, ["t_touch_limitup_actual", "t_touch_limitup"], default=np.nan)
    close_ret = _t1_close_ret_from_verify_df(df)
    high_ret = _t1_high_ret_from_verify_df(df, close_ret)
    valid = (
        pd.to_numeric(ready, errors="coerce").fillna(0).eq(1)
        & t_up.notna()
        & t_limitup.notna()
        & t_touch.notna()
        & close_ret.notna()
        & high_ret.notna()
    )
    return bool(valid.any())


def _backfill_limitup_truth_from_market(
    cfg: PremiumConfig,
    df: pd.DataFrame,
    path: Path,
) -> Tuple[pd.DataFrame, str]:
    """
    历史 premium_verify 文件没有涨停标签时，用 T/T+1 日频行情补齐训练标签。

    日频条件下，T 日竞价买入收益用 T 日 open 做近似；涨停/触板用 T 日 close/high 与
    D 日 close 推算的制度涨停价校验。
    """
    if df.empty or _has_complete_limitup_truth(df):
        return df, "already_labeled"

    ts_col = _first_existing_col(df, ["ts_code", "code", "symbol"])
    if not ts_col:
        return df, "missing_ts_code"

    d_trade_date = _first_nonempty_date(
        df,
        ["d_analysis_trade_date", "base_date", "trade_date", "date"],
        fallback=_infer_trade_date_from_path(path),
    )
    if not d_trade_date:
        return df, "missing_d_trade_date"

    t_trade_date = _first_nonempty_date(df, ["buy_date", "t_trade_date"])
    t1_trade_date = _first_nonempty_date(df, ["target_date", "t1_trade_date", "next_trade_date"])
    if not t_trade_date or not t1_trade_date:
        t_probe, t1_probe, reason = _infer_next_market_trade_dates(cfg, d_trade_date)
        t_trade_date = t_trade_date or t_probe
        t1_trade_date = t1_trade_date or t1_probe
        if not t_trade_date or not t1_trade_date:
            return df, reason

    r_t = ensure_daily_cached(cfg, t_trade_date)
    if not r_t.ok:
        return df, f"t_daily_not_ready:{t_trade_date}:{r_t.reason}"
    r_t1 = ensure_daily_cached(cfg, t1_trade_date)
    if not r_t1.ok:
        return df, f"t1_daily_not_ready:{t1_trade_date}:{r_t1.reason}"

    try:
        daily_t_raw = load_daily(cfg, t_trade_date)
        daily_t1_raw = load_daily(cfg, t1_trade_date)
        daily_t = daily_t_raw[[c for c in ["ts_code", "open", "high", "low", "close"] if c in daily_t_raw.columns]].copy()
        daily_t1 = daily_t1_raw[[c for c in ["ts_code", "open", "high", "low", "close"] if c in daily_t1_raw.columns]].copy()
    except Exception as e:
        return df, f"daily_load_error:{type(e).__name__}:{e}"

    out = df.copy()
    out["_join_ts_code"] = out[ts_col].map(_norm_ts_code_for_market)
    daily_t["ts_code"] = daily_t["ts_code"].map(_norm_ts_code_for_market)
    daily_t1["ts_code"] = daily_t1["ts_code"].map(_norm_ts_code_for_market)

    daily_t = daily_t.rename(
        columns={
            "ts_code": "_join_ts_code",
            "open": "_bf_open_T_actual",
            "high": "_bf_high_T_actual",
            "low": "_bf_low_T_actual",
            "close": "_bf_close_T_actual",
        }
    )
    daily_t1 = daily_t1.rename(
        columns={
            "ts_code": "_join_ts_code",
            "open": "_bf_open_T2_actual",
            "high": "_bf_high_T2_actual",
            "low": "_bf_low_T2_actual",
            "close": "_bf_close_T2_actual",
        }
    )
    out = out.merge(daily_t, on="_join_ts_code", how="left")
    out = out.merge(daily_t1, on="_join_ts_code", how="left")

    d_close = _d_close_numeric_series(out, default=np.nan)
    t_open = pd.to_numeric(out["_bf_open_T_actual"], errors="coerce")
    t_high = pd.to_numeric(out["_bf_high_T_actual"], errors="coerce")
    t_low = pd.to_numeric(out.get("_bf_low_T_actual", np.nan), errors="coerce")
    t_close = pd.to_numeric(out["_bf_close_T_actual"], errors="coerce")
    t1_open = pd.to_numeric(out.get("_bf_open_T2_actual", np.nan), errors="coerce")
    t1_high = pd.to_numeric(out["_bf_high_T2_actual"], errors="coerce")
    t1_low = pd.to_numeric(out.get("_bf_low_T2_actual", np.nan), errors="coerce")
    t1_close = pd.to_numeric(out["_bf_close_T2_actual"], errors="coerce")

    limit_rates = out["_join_ts_code"].map(_limit_rate_for_code).astype(float)
    limit_price = (d_close * (1.0 + limit_rates)).round(2)
    entry_price = t_open.where(t_open > 0, _price_numeric_series(
        out,
        ["t_max_buy_price", "T日可接受买入价", "entry_price_t1", "entry_price_proxy_t1"],
        default=np.nan,
    ))
    entry_price = entry_price.where(entry_price > 0, t_close)

    ready = t_close.notna() & t_high.notna() & t1_close.notna() & t1_high.notna() & (entry_price > 0) & limit_price.notna()

    out["trade_date"] = d_trade_date
    out["d_analysis_trade_date"] = d_trade_date
    out["buy_date"] = t_trade_date
    out["target_date"] = t1_trade_date
    out["t_trade_date"] = t_trade_date
    out["t1_trade_date"] = t1_trade_date
    out["open_T_actual"] = t_open
    out["high_T_actual"] = t_high
    out["low_T_actual"] = t_low
    out["close_T_actual"] = t_close
    out["open_T2_actual"] = t1_open
    out["high_T2_actual"] = t1_high
    out["low_T2_actual"] = t1_low
    out["close_T2_actual"] = t1_close
    out["t_limit_price_est"] = limit_price
    out["t_open_ret"] = np.where(ready, t_open / d_close - 1.0, np.nan)
    out["t_intraday_ret"] = np.where(ready, t_high / d_close - 1.0, np.nan)
    out["t_low_ret"] = np.where(ready, t_low / d_close - 1.0, np.nan)
    out["t_close_ret"] = np.where(ready, t_close / d_close - 1.0, np.nan)
    out["t_open_up_hit"] = np.where(ready, (t_open > d_close).astype(int), np.nan)
    out["t_up_actual"] = np.where(ready, (t_close > d_close).astype(int), np.nan)
    out["t_high_profit_hit"] = np.where(ready, (out["t_intraday_ret"] >= 0.02).astype(int), np.nan)
    out["t_limitup_actual"] = np.where(ready, (t_close >= limit_price * 0.9985).astype(int), np.nan)
    out["t_touch_limitup_actual"] = np.where(ready, (t_high >= limit_price * 0.9985).astype(int), np.nan)
    out["t_limitup_verify_ready"] = ready.astype(int)
    out["t_limitup_verify_reason"] = np.where(ready, "ok_backfilled_daily", "missing_daily_row")
    out["t_limitup_verify_trade_date"] = t_trade_date
    out["t1_open_ret"] = np.where(ready, t1_open / entry_price - 1.0, np.nan)
    out["t1_close_ret"] = np.where(ready, t1_close / entry_price - 1.0, np.nan)
    out["t1_high_ret"] = np.where(ready, t1_high / entry_price - 1.0, np.nan)
    out["t1_low_ret"] = np.where(ready, t1_low / entry_price - 1.0, np.nan)
    out["t1_up_hit"] = np.where(ready, (out["t1_close_ret"] > 0).astype(int), np.nan)
    out["t1_high_profit_hit"] = np.where(ready, (out["t1_high_ret"] >= 0.02).astype(int), np.nan)
    out["t1_accept_hit"] = np.where(
        ready,
        ((pd.to_numeric(out["t1_high_ret"], errors="coerce") >= 0.015) & (pd.to_numeric(out["t1_close_ret"], errors="coerce") >= -0.015)).astype(int),
        np.nan,
    )
    out["t1_fail_hit"] = np.where(
        ready,
        ((pd.to_numeric(out["t1_high_ret"], errors="coerce") < 0.008) | (pd.to_numeric(out["t1_close_ret"], errors="coerce") <= -0.025)).astype(int),
        np.nan,
    )
    t1_low_ret_num = pd.to_numeric(out["t1_low_ret"], errors="coerce")
    out["t1_big_drawdown_hit"] = np.where(ready & t1_low_ret_num.notna(), (t1_low_ret_num <= -0.04).astype(int), np.nan)
    out["t1_limitdown_risk_hit"] = np.where(ready & t1_low_ret_num.notna(), (t1_low_ret_num <= -0.08).astype(int), np.nan)

    tmp_cols = [c for c in out.columns if str(c).startswith("_bf_")]
    out = out.drop(columns=tmp_cols + ["_join_ts_code"], errors="ignore")
    valid_n = int(pd.to_numeric(out["t_limitup_verify_ready"], errors="coerce").fillna(0).sum())
    return out, f"backfilled:{valid_n}/{len(out)}:{t_trade_date}->{t1_trade_date}"


def _find_prev_market_trade_date(cfg: PremiumConfig, trade_date: str, max_probe_days: int = 15) -> Tuple[Optional[str], str]:
    trade_date = _to_yyyymmdd(trade_date)
    try:
        d0 = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    except Exception:
        return None, f"bad_trade_date:{trade_date}"
    notes: List[str] = []
    for i in range(1, int(max_probe_days) + 1):
        cand = (d0 - dt.timedelta(days=i)).strftime("%Y%m%d")
        r = ensure_daily_cached(cfg, cand)
        if r.ok:
            return cand, "ok"
        if len(notes) < 3:
            notes.append(f"{cand}:{r.reason}")
    return None, "prev_market_daily_not_found:" + "|".join(notes)


def _market_sentiment_features(cfg: PremiumConfig, trade_date: str) -> Dict[str, float]:
    base: Dict[str, float] = {
        "mkt_stock_count": 0.0,
        "mkt_up_ratio": np.nan,
        "mkt_avg_ret": np.nan,
        "mkt_median_ret": np.nan,
        "mkt_strong_count": 0.0,
        "mkt_strong_ratio": np.nan,
        "mkt_touch_strong_count": 0.0,
        "mkt_touch_strong_ratio": np.nan,
        "mkt_amount_sum": np.nan,
        "mkt_emotion_score": np.nan,
    }
    r = ensure_daily_cached(cfg, trade_date)
    if not r.ok:
        return base
    prev_date, _ = _find_prev_market_trade_date(cfg, trade_date)
    if not prev_date:
        return base
    try:
        d = load_daily(cfg, trade_date)[["ts_code", "open", "high", "close", "amount"]].copy()
        p = load_daily(cfg, prev_date)[["ts_code", "close"]].rename(columns={"close": "prev_close"})
        m = d.merge(p, on="ts_code", how="inner")
        for c in ("open", "high", "close", "prev_close", "amount"):
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m[(m["prev_close"] > 0) & m["close"].notna()].copy()
        if m.empty:
            return base
        ret = m["close"] / m["prev_close"] - 1.0
        high_ret = m["high"] / m["prev_close"] - 1.0
        strong = ret >= 0.095
        touch_strong = high_ret >= 0.095
        up_ratio = float((ret > 0).mean())
        avg_ret = float(ret.mean())
        strong_ratio = float(strong.mean())
        touch_ratio = float(touch_strong.mean())
        emotion = (
            0.45 * up_ratio
            + 0.25 * np.clip((avg_ret + 0.02) / 0.06, 0.0, 1.0)
            + 0.20 * np.clip(touch_ratio * 8.0, 0.0, 1.0)
            + 0.10 * np.clip(strong_ratio * 10.0, 0.0, 1.0)
        )
        base.update({
            "mkt_stock_count": float(len(m)),
            "mkt_up_ratio": up_ratio,
            "mkt_avg_ret": avg_ret,
            "mkt_median_ret": float(ret.median()),
            "mkt_strong_count": float(int(strong.sum())),
            "mkt_strong_ratio": strong_ratio,
            "mkt_touch_strong_count": float(int(touch_strong.sum())),
            "mkt_touch_strong_ratio": touch_ratio,
            "mkt_amount_sum": float(pd.to_numeric(m["amount"], errors="coerce").sum()),
            "mkt_emotion_score": float(np.clip(emotion, 0.0, 1.0)),
        })
    except Exception:
        pass
    return base


def _attach_market_sentiment_to_samples(cfg: PremiumConfig, samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty or "d_trade_date" not in samples.columns:
        return samples
    out = samples.copy()
    feature_cache: Dict[str, Dict[str, float]] = {}
    for d in sorted(out["d_trade_date"].astype(str).map(_to_yyyymmdd).dropna().unique()):
        feature_cache[d] = _market_sentiment_features(cfg, d)
    for c in [
        "mkt_stock_count", "mkt_up_ratio", "mkt_avg_ret", "mkt_median_ret",
        "mkt_strong_count", "mkt_strong_ratio", "mkt_touch_strong_count",
        "mkt_touch_strong_ratio", "mkt_amount_sum", "mkt_emotion_score",
    ]:
        out[c] = out["d_trade_date"].astype(str).map(lambda d: feature_cache.get(_to_yyyymmdd(d), {}).get(c, np.nan))
    return out


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
            "ev": ["ev", "EV", "score_ev", "pred_ev"],
            "turnover_rate": ["turnover_rate", "换手率"],
            "amount": ["amount", "成交额"],
            "vol": ["vol", "volume", "成交量"],
            "close": ["close", "close_t", "收盘价", "close_T", "close_2", "close_t1"],
            "pct_chg": ["pct_chg", "pct_change", "涨跌幅"],
            "amplitude": ["amplitude", "range_1d", "振幅"],
        }
        for target, candidates in feature_candidates.items():
            s = _safe_numeric_series(df, candidates, default=np.nan)
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


# ========= premium_verify 回收涨停概率引擎样本 =========

def _t1_close_ret_from_verify_df(df: pd.DataFrame) -> pd.Series:
    """Return T+1 close return using the canonical T-open execution proxy."""
    explicit = _safe_numeric_series(
        df,
        ["t1_close_ret", "premium_ret_t1_to_t2", "realized_ret_t1_to_t2"],
        default=np.nan,
    )
    if explicit.notna().any():
        return explicit

    close_t1 = _safe_numeric_series(df, ["close_T2_actual", "close_t2", "sell_close", "exit_close"], default=np.nan)
    entry = _safe_numeric_series(
        df,
        ["entry_price_proxy", "open_T_actual", "t_open", "open_t"],
        default=np.nan,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close_t1 / entry - 1.0
    return pd.to_numeric(ret, errors="coerce")


def _t1_high_ret_from_verify_df(df: pd.DataFrame, close_ret: pd.Series) -> pd.Series:
    high_ret = _safe_numeric_series(df, ["t1_high_ret", "high_ret_t1", "high_ret_T2"], default=np.nan)
    if high_ret.notna().any():
        return high_ret

    high_t1 = _safe_numeric_series(df, ["high_T2_actual", "high_t2", "high_sell"], default=np.nan)
    entry = _safe_numeric_series(
        df,
        ["entry_price_proxy", "open_T_actual", "t_open", "open_t"],
        default=np.nan,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = high_t1 / entry - 1.0
    derived = pd.to_numeric(derived, errors="coerce")
    return derived.where(derived.notna(), close_ret)


def _collect_limitup_samples_from_verify_outputs(cfg: PremiumConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    从 premium_verify_*.csv 回收涨停/T+1 概率模型样本。

    这些文件只在真实行情到来后补齐验证字段，天然适合做自学习闭环。
    特征列采用白名单，避免 raw_abs_err/actual_ret 等未来字段泄漏进模型。
    """
    out_root = cfg.out_root()
    files = sorted(out_root.glob("premium_verify_*.csv"))
    stats = {
        "n_verify_files": len(files),
        "ok_files": 0,
        "skipped_files": 0,
        "backfilled_files": 0,
        "backfill_failed": 0,
        "notes": [],
    }
    rows: List[pd.DataFrame] = []
    used_sources: List[str] = []

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
        try:
            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            source_sha256 = ""

        df, backfill_reason = _backfill_limitup_truth_from_market(cfg, df, path)
        if str(backfill_reason).startswith("backfilled:"):
            stats["backfilled_files"] += 1
            stats["notes"].append(f"{path.name}: {backfill_reason}")
        elif backfill_reason not in ("already_labeled",):
            stats["backfill_failed"] += 1
            stats["notes"].append(f"{path.name}: backfill skipped: {backfill_reason}")

        ts_col = _first_existing_col(df, ["ts_code", "code", "symbol"])
        if not ts_col:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip {path.name}: missing ts_code")
            continue

        ready_t = _bool_numeric_series(df, ["t_limitup_verify_ready"], default=np.nan)
        ready_t1 = _bool_numeric_series(df, ["t1_verify_ready", "label_matured"], default=np.nan)
        t_up = _bool_numeric_series(df, ["t_up_actual", "t_up_hit"], default=np.nan)
        t_limitup = _bool_numeric_series(df, ["t_limitup_actual", "t_limitup_hit"], default=np.nan)
        t_touch = _bool_numeric_series(df, ["t_touch_limitup_actual", "t_touch_limitup"], default=np.nan)
        close_ret = _t1_close_ret_from_verify_df(df)
        high_ret = _t1_high_ret_from_verify_df(df, close_ret)
        d_close = _d_close_numeric_series(df, default=np.nan)
        t_open_actual = _safe_numeric_series(df, ["open_T_actual", "open_t", "t_open"], default=np.nan)
        t_high_actual = _safe_numeric_series(df, ["high_T_actual", "high_t", "t_high"], default=np.nan)
        t_close_actual = _safe_numeric_series(df, ["close_T_actual", "close_t", "t_close"], default=np.nan)
        t1_open_actual = _safe_numeric_series(df, ["open_T2_actual", "t1_open", "open_t1"], default=np.nan)
        t1_low_actual = _safe_numeric_series(df, ["low_T2_actual", "t1_low", "low_t1"], default=np.nan)
        entry_price = _price_numeric_series(
            df,
            ["entry_price_proxy", "open_T_actual", "t_open", "open_t"],
            default=np.nan,
        )
        entry_price = entry_price.where(entry_price > 0, t_open_actual)
        t_close_ret = _safe_numeric_series(df, ["t_close_ret"], default=np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_close_ret = t_close_ret.where(t_close_ret.notna(), t_close_actual / d_close - 1.0)
            t_intraday_ret = _safe_numeric_series(df, ["t_intraday_ret"], default=np.nan)
            t_intraday_ret = t_intraday_ret.where(t_intraday_ret.notna(), t_high_actual / d_close - 1.0)
            t_open_ret = _safe_numeric_series(df, ["t_open_ret"], default=np.nan)
            t_open_ret = t_open_ret.where(t_open_ret.notna(), t_open_actual / d_close - 1.0)
            t1_open_ret = _safe_numeric_series(df, ["t1_open_ret"], default=np.nan)
            t1_open_ret = t1_open_ret.where(t1_open_ret.notna(), t1_open_actual / entry_price - 1.0)
            t1_low_ret = _safe_numeric_series(df, ["t1_low_ret"], default=np.nan)
            t1_low_ret = t1_low_ret.where(t1_low_ret.notna(), t1_low_actual / entry_price - 1.0)

        out = pd.DataFrame(index=df.index)
        td_col = _first_existing_col(df, ["trade_date", "d_analysis_trade_date", "base_date", "date"])
        buy_col = _first_existing_col(df, ["buy_date", "t_trade_date"])
        target_col = _first_existing_col(df, ["target_date", "t1_trade_date", "next_trade_date"])
        name_col = _first_existing_col(df, ["name", "名称", "stock_name"])

        out["d_trade_date"] = df[td_col].astype(str).map(_to_yyyymmdd) if td_col else _infer_trade_date_from_path(path)
        out["trade_date"] = out["d_trade_date"]
        out["t_trade_date"] = df[buy_col].astype(str).map(_to_yyyymmdd) if buy_col else pd.NA
        out["t1_trade_date"] = df[target_col].astype(str).map(_to_yyyymmdd) if target_col else pd.NA
        out["ts_code"] = df[ts_col].astype(str).str.strip()
        out["name"] = df[name_col].astype(str) if name_col else pd.NA
        out["feature_as_of_date"] = out["d_trade_date"]
        out["feature_known_at"] = "D_CLOSE"
        out["feature_snapshot_source"] = path.name
        out["feature_snapshot_sha256"] = source_sha256

        date_order_ok = (
            out["d_trade_date"].notna()
            & out["t_trade_date"].notna()
            & out["t1_trade_date"].notna()
            & (out["d_trade_date"].astype(str) < out["t_trade_date"].astype(str))
            & (out["t_trade_date"].astype(str) < out["t1_trade_date"].astype(str))
        )
        out["label_matured"] = np.where(ready_t1.notna(), ready_t1, 0.0)
        out["t_verify_ready"] = np.where(ready_t.notna(), ready_t, 0.0)
        out["date_order_valid"] = date_order_ok.astype(float)
        out["t_up_hit"] = t_up
        out["t_open_up_hit"] = (pd.to_numeric(t_open_ret, errors="coerce") > 0).where(pd.to_numeric(t_open_ret, errors="coerce").notna(), np.nan).astype(float)
        out["t_limitup_hit"] = t_limitup
        out["t_touch_limitup"] = t_touch.where(t_touch.notna(), t_limitup)
        out["t_close_ret"] = t_close_ret
        out["t_intraday_ret"] = t_intraday_ret
        out["t_open_ret"] = t_open_ret
        out["t_high_profit_hit"] = (pd.to_numeric(t_intraday_ret, errors="coerce") >= 0.02).where(pd.to_numeric(t_intraday_ret, errors="coerce").notna(), np.nan).astype(float)
        out["t1_open_ret"] = t1_open_ret
        out["t1_close_ret"] = close_ret
        out["t1_high_ret"] = high_ret
        out["t1_low_ret"] = t1_low_ret
        close_ret_num = pd.to_numeric(close_ret, errors="coerce")
        high_ret_num = pd.to_numeric(high_ret, errors="coerce")
        out["t1_up_hit"] = (close_ret_num > 0).where(close_ret_num.notna(), np.nan).astype(float)
        out["t1_high_profit_hit"] = (high_ret_num >= 0.02).where(high_ret_num.notna(), np.nan).astype(float)
        out["t1_accept_hit"] = (
            (pd.to_numeric(high_ret, errors="coerce") >= 0.015)
            & (pd.to_numeric(close_ret, errors="coerce") >= -0.015)
        ).where(pd.to_numeric(high_ret, errors="coerce").notna() & pd.to_numeric(close_ret, errors="coerce").notna(), np.nan).astype(float)
        out["t1_fail_hit"] = (
            (pd.to_numeric(high_ret, errors="coerce") < 0.008)
            | (pd.to_numeric(close_ret, errors="coerce") <= -0.025)
        ).where(pd.to_numeric(high_ret, errors="coerce").notna() & pd.to_numeric(close_ret, errors="coerce").notna(), np.nan).astype(float)
        out["t1_big_drawdown_hit"] = (pd.to_numeric(t1_low_ret, errors="coerce") <= -0.04).where(pd.to_numeric(t1_low_ret, errors="coerce").notna(), np.nan).astype(float)
        out["t1_limitdown_risk_hit"] = (pd.to_numeric(t1_low_ret, errors="coerce") <= -0.08).where(pd.to_numeric(t1_low_ret, errors="coerce").notna(), np.nan).astype(float)

        feature_candidates = {
            "rank": ["rank", "dec_rank"],
            # close_t is intentionally excluded: historical files can use it
            # for the future T-day close instead of the D-day snapshot.
            "close_T": ["close_T", "d_close", "close", "收盘价"],
            "p_premium": ["p_premium", "probability", "pred_prob"],
            "e_premium": ["e_premium", "e_ret", "E_ret", "pred_ret"],
            "score_ev": ["score_ev", "ev", "EV", "final_score", "score"],
            "confidence": ["confidence", "conf"],
            "data_quality": ["data_quality"],
            "dec_rank": ["dec_rank"],
            "dec_weight": ["dec_weight", "weight"],
            "dec_p_fill": ["dec_p_fill", "p_fill_pred", "p_fill"],
            "t_limitup_prob_rule": ["t_limitup_prob_rule", "t_limitup_prob", "T日涨停概率"],
            "t_limitup_strength_rule": ["t_limitup_strength_rule", "t_limitup_strength", "T日涨停强度"],
            "t1_continue_up_rate_rule": ["t1_continue_up_rate_rule", "t1_continue_up_rate", "T+1延续上涨率"],
            "limitup_continuation_score_rule": ["limitup_continuation_score_rule", "limitup_continuation_score", "涨停接力评分"],
            "t_limitup_prob_model": ["t_limitup_prob_model"],
            "t_limitup_prob_engine": ["t_limitup_prob_engine"],
            "t_limit_model_probability_ready": ["t_limit_model_probability_ready"],
            "t1_up_prob_model": ["t1_up_prob_model"],
            "t1_continue_up_rate_engine": ["t1_continue_up_rate_engine"],
            "t1_model_probability_ready": ["t1_model_probability_ready"],
            "eret_pred_raw": ["eret_pred_raw", "e_ret_pred_raw", "raw_eret_pred", "E_ret"],
            "eret_plus_value": ["eret_plus_value", "eret_plus", "E_ret_plus"],
            "eret_plus_delta": ["eret_plus_delta"],
            "t1_up_rate": ["t1_up_rate", "T+1上涨率"],
            "market_score": ["market_score"],
            "mkt_stock_count": ["mkt_stock_count"],
            "mkt_up_ratio": ["mkt_up_ratio"],
            "mkt_avg_ret": ["mkt_avg_ret"],
            "mkt_median_ret": ["mkt_median_ret"],
            "mkt_strong_count": ["mkt_strong_count"],
            "mkt_strong_ratio": ["mkt_strong_ratio"],
            "mkt_touch_strong_count": ["mkt_touch_strong_count"],
            "mkt_touch_strong_ratio": ["mkt_touch_strong_ratio"],
            "mkt_amount_sum": ["mkt_amount_sum"],
            "mkt_emotion_score": ["mkt_emotion_score"],
        }
        for target, candidates in feature_candidates.items():
            out[target] = (
                _d_close_numeric_series(df, default=np.nan)
                if target == "close_T"
                else _safe_numeric_series(df, candidates, default=np.nan)
            )

        point_in_time_factor_cols = [
            "turnover_rate", "turnover_rate_f", "volume_ratio", "circ_mv", "float_mv", "total_mv",
            "pe_ttm", "pb", "returns_1d", "volatility_5d", "volatility_10d", "volatility_20d",
            "max_drawdown_20d", "tail_risk_score", "hot_boards_score", "board_crowding_rank",
            "is_hot_board", "board_rank", "board_limit_up_count", "is_st_like",
            "open_times", "fd_amount", "seal_amount", "limit_touch_count", "open_board_count",
            "max_drawdown_after_limit", "reseal_count", "reseal_minutes_avg", "late_volume_ratio",
            "late_price_weakness", "late_limit_hold_minutes", "late_withdraw_score", "reseal_score",
            "intraday_quality_score", "intraday_confidence_score", "intraday_risk_score",
            "intraday_soft_risk_score", "intraday_hard_risk_flag", "auction_strength_score",
            "auction_amount", "is_limit_up", "break_count_proxy", "limit_up_strength",
            "factor_intraday_available", "factor_auction_available", "factor_intraday_quality",
            "factor_intraday_confidence", "factor_intraday_soft_risk", "factor_intraday_risk",
            "factor_intraday_hard_risk", "factor_late_withdraw", "factor_reseal",
            "factor_open_board_count", "factor_auction_strength", "factor_intraday_attack_edge",
            "factor_intraday_execution_edge", "factor_intraday_risk_penalty",
        ]
        for name in point_in_time_factor_cols:
            out[name] = _safe_numeric_series(df, [name], default=np.nan)

        for c in df.columns:
            name = str(c).strip()
            if re.fullmatch(r"r_p\d{2}", name) or re.fullmatch(r"close_T2_p\d{2}", name):
                out[name] = pd.to_numeric(df[c], errors="coerce")

        out["is_top10"] = (pd.to_numeric(out["rank"], errors="coerce") <= 10).astype(float)
        out["is_top20"] = (pd.to_numeric(out["rank"], errors="coerce") <= 20).astype(float)

        t1_ready_mask = pd.to_numeric(out["label_matured"], errors="coerce").fillna(0).eq(1)
        t1_target_cols = [
            "t1_open_ret", "t1_close_ret", "t1_high_ret", "t1_low_ret",
            "t1_up_hit", "t1_high_profit_hit", "t1_accept_hit", "t1_fail_hit",
            "t1_big_drawdown_hit", "t1_limitdown_risk_hit",
        ]
        out.loc[~t1_ready_mask, t1_target_cols] = np.nan

        t_valid = (
            pd.to_numeric(out["t_verify_ready"], errors="coerce").fillna(0).eq(1)
            & pd.to_numeric(out["date_order_valid"], errors="coerce").fillna(0).eq(1)
            & out["t_up_hit"].notna()
            & out["t_high_profit_hit"].notna()
            & out["t_limitup_hit"].notna()
            & out["t_touch_limitup"].notna()
            & out["t_close_ret"].notna()
            & out["t_intraday_ret"].notna()
        )
        out = out.loc[t_valid].reset_index(drop=True)
        if out.empty:
            stats["skipped_files"] += 1
            stats["notes"].append(f"skip {path.name}: no verified T/T+1 rows")
            continue

        stats["ok_files"] += 1
        used_sources.append(f"{path.name}:{source_sha256}")
        rows.append(out)

    stats["used_source_files"] = int(len(used_sources))
    stats["source_manifest_sha256"] = hashlib.sha256(
        "\n".join(sorted(used_sources)).encode("utf-8")
    ).hexdigest()
    if not rows:
        return pd.DataFrame(), stats

    samples = pd.concat(rows, ignore_index=True, sort=False)
    samples["d_trade_date"] = samples["d_trade_date"].astype(str).map(_to_yyyymmdd)
    samples["trade_date"] = samples["d_trade_date"]
    samples["ts_code"] = samples["ts_code"].astype(str).str.strip()
    samples = samples.drop_duplicates(subset=["d_trade_date", "ts_code"], keep="last").reset_index(drop=True)
    samples = _attach_market_sentiment_to_samples(cfg, samples)
    return samples, stats


def _build_limitup_feature_cols(samples: pd.DataFrame) -> List[str]:
    allow = [
        "close_T",
        "p_premium", "e_premium", "score_ev", "confidence", "data_quality",
        "dec_rank", "dec_weight", "dec_p_fill",
        "t_limitup_prob_rule", "t_limitup_strength_rule", "t1_continue_up_rate_rule",
        "limitup_continuation_score_rule",
        "eret_pred_raw", "eret_plus_value", "eret_plus_delta",
        "t1_up_rate",
        "mkt_stock_count", "mkt_up_ratio", "mkt_avg_ret", "mkt_median_ret",
        "mkt_strong_count", "mkt_strong_ratio", "mkt_touch_strong_count",
        "mkt_touch_strong_ratio", "mkt_amount_sum", "mkt_emotion_score",
        "intraday_available", "intraday_quality_score", "intraday_soft_risk_score",
        "intraday_hard_risk_flag", "intraday_risk_score", "intraday_confidence_score",
        "late_withdraw_score", "reseal_score", "open_board_count", "auction_strength_score",
        "factor_intraday_available", "factor_intraday_quality", "factor_intraday_confidence",
        "factor_intraday_soft_risk", "factor_intraday_risk", "factor_intraday_hard_risk",
        "factor_late_withdraw", "factor_reseal", "factor_open_board_count",
        "factor_auction_strength", "factor_intraday_attack_edge",
        "factor_intraday_execution_edge", "factor_intraday_risk_penalty",
        "intraday_attack_edge", "intraday_execution_edge", "intraday_risk_penalty",
        "turnover_rate", "turnover_rate_f", "volume_ratio", "circ_mv", "float_mv", "total_mv",
        "pe_ttm", "pb", "returns_1d", "volatility_5d", "volatility_10d", "volatility_20d",
        "max_drawdown_20d", "tail_risk_score", "hot_boards_score", "board_crowding_rank",
        "is_hot_board", "board_rank", "board_limit_up_count", "is_st_like",
        "open_times", "fd_amount", "seal_amount", "limit_touch_count", "open_board_count",
        "max_drawdown_after_limit", "reseal_count", "reseal_minutes_avg", "late_volume_ratio",
        "late_price_weakness", "late_limit_hold_minutes", "auction_amount", "is_limit_up",
        "break_count_proxy", "limit_up_strength", "factor_auction_available",
    ]
    allow += [c for c in samples.columns if re.fullmatch(r"r_p\d{2}", str(c)) or re.fullmatch(r"close_T2_p\d{2}", str(c))]
    cols = []
    seen = set()
    for c in allow:
        if c in samples.columns and c not in seen:
            s = pd.to_numeric(samples[c], errors="coerce")
            if s.notna().sum() >= max(5, int(len(samples) * 0.05)):
                cols.append(c)
                seen.add(c)
    return cols


def _limitup_probability_min_samples(cfg: PremiumConfig) -> int:
    return max(20, int(getattr(cfg, "min_train_days", 20)))


def _save_limitup_meta(meta_path: Path, payload: Dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _train_limitup_probability_from_verify(
    cfg: PremiumConfig,
    samples: pd.DataFrame,
    stats: Dict,
) -> Dict:
    model_path = cfg.out_root() / "models" / "limitup_probability_engine.joblib"
    candidate_path = cfg.out_root() / "models" / "limitup_probability_engine_candidate.joblib"
    metrics_path = cfg.out_root() / "models" / "limitup_probability_engine_metrics.csv"
    meta_path = cfg.out_root() / "models" / "limitup_probability_engine_meta.json"
    trainset_path = cfg.out_learning_dir() / "limitup_probability_training_samples.csv"

    base_state = {
        "limitup_trained": False,
        "limitup_reason": "not_run",
        "limitup_n_samples": 0,
        "limitup_n_days": 0,
        "limitup_min_samples": _limitup_probability_min_samples(cfg),
        "limitup_model_path": "",
        "limitup_metrics_path": "",
        "limitup_meta_path": str(meta_path),
    }

    n_samples = int(len(samples))
    n_days = int(samples["d_trade_date"].nunique()) if (not samples.empty and "d_trade_date" in samples.columns) else 0
    state = dict(base_state)
    state.update({"limitup_n_samples": n_samples, "limitup_n_days": n_days})

    if samples.empty:
        state["limitup_reason"] = "no_limitup_verify_samples"
        _save_limitup_meta(meta_path, {"kind": "limitup_probability_engine", "trained": False, "reason": state["limitup_reason"], "stats": stats, "created_at_utc": utc_now_iso(), "commit_sha": get_commit_sha(cfg.repo_root()), "run_id": get_run_id()})
        return state

    if n_days < 4:
        state["limitup_reason"] = f"limitup_days_not_enough:{n_days}<4"
        _save_limitup_meta(meta_path, {"kind": "limitup_probability_engine", "trained": False, "reason": state["limitup_reason"], "n_samples": n_samples, "n_days": n_days, "stats": stats, "created_at_utc": utc_now_iso(), "commit_sha": get_commit_sha(cfg.repo_root()), "run_id": get_run_id()})
        return state

    feature_cols = _build_limitup_feature_cols(samples)
    if not feature_cols:
        state["limitup_reason"] = "limitup_feature_cols_empty"
        _save_limitup_meta(meta_path, {"kind": "limitup_probability_engine", "trained": False, "reason": state["limitup_reason"], "n_samples": n_samples, "n_days": n_days, "stats": stats, "created_at_utc": utc_now_iso(), "commit_sha": get_commit_sha(cfg.repo_root()), "run_id": get_run_id()})
        return state

    try:
        cfg.out_learning_dir().mkdir(parents=True, exist_ok=True)
        samples.to_csv(trainset_path, index=False, encoding="utf-8-sig")
        bundle = fit_limitup_probability_engine(
            samples,
            feature_cols=feature_cols,
            valid_ratio=0.30,
            min_samples=_limitup_probability_min_samples(cfg),
        )
        save_limitup_probability_bundle(bundle, candidate_path)
        promoted = bool(getattr(bundle, "model_can_rank", False))
        if promoted:
            save_limitup_probability_bundle(bundle, model_path)
        elif model_path.exists():
            try:
                active_bundle = load_limitup_probability_bundle(model_path)
                active_is_current = bool(
                    str(getattr(active_bundle, "gate_version", "")) == GATE_VERSION
                    and int(getattr(active_bundle, "artifact_version", 0) or 0)
                    == BUNDLE_ARTIFACT_VERSION
                )
            except Exception:
                active_is_current = False
            if not active_is_current:
                model_path.unlink()
        bundle.metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        state.update({
            "limitup_trained": promoted,
            "limitup_candidate_trained": True,
            "limitup_promoted": promoted,
            "limitup_reason": "ok_promoted" if promoted else str(bundle.gate_reason),
            "limitup_model_path": str(model_path) if promoted else "",
            "limitup_candidate_path": str(candidate_path),
            "limitup_metrics_path": str(metrics_path),
            "limitup_trainset_path": str(trainset_path),
            "limitup_feature_n": int(len(feature_cols)),
            "limitup_train_end_date": str(bundle.train_end_date),
            "limitup_valid_start_date": str(bundle.valid_start_date),
            "limitup_gate_version": str(getattr(bundle, "gate_version", GATE_VERSION)),
            "limitup_artifact_version": int(
                getattr(bundle, "artifact_version", BUNDLE_ARTIFACT_VERSION)
            ),
            "limitup_target_gate_status": dict(getattr(bundle, "target_gate_status", {}) or {}),
            "limitup_target_probability_status": dict(
                getattr(bundle, "target_probability_status", {}) or {}
            ),
            "limitup_validation_mode": str(getattr(bundle, "validation_mode", "")),
            "limitup_walk_forward_folds": int(getattr(bundle, "walk_forward_folds", 0) or 0),
            "limitup_data_fingerprint": str(getattr(bundle, "data_fingerprint", "")),
            "limitup_feature_fingerprint": str(getattr(bundle, "feature_fingerprint", "")),
            "limitup_feature_contract_version": str(
                getattr(bundle, "feature_contract_version", "")
            ),
        })
        _save_limitup_meta(
            meta_path,
            {
                "kind": "limitup_probability_engine",
                "model_version": cfg.model_version,
                "trained": True,
                "reason": "ok_promoted" if promoted else str(bundle.gate_reason),
                "promoted": promoted,
                "model_can_rank": promoted,
                "gate_reason": str(bundle.gate_reason),
                "gate_version": str(getattr(bundle, "gate_version", GATE_VERSION)),
                "artifact_version": int(
                    getattr(bundle, "artifact_version", BUNDLE_ARTIFACT_VERSION)
                ),
                "target_gate_status": dict(getattr(bundle, "target_gate_status", {}) or {}),
                "target_gate_reasons": dict(getattr(bundle, "target_gate_reasons", {}) or {}),
                "target_probability_status": dict(
                    getattr(bundle, "target_probability_status", {}) or {}
                ),
                "target_probability_reasons": dict(
                    getattr(bundle, "target_probability_reasons", {}) or {}
                ),
                "source": "premium_verify",
                "feature_cols": feature_cols,
                "n_samples": n_samples,
                "n_days": n_days,
                "min_samples": _limitup_probability_min_samples(cfg),
                "train_end_date": bundle.train_end_date,
                "valid_start_date": bundle.valid_start_date,
                "validation_days": int(bundle.validation_days),
                "validation_samples": int(bundle.validation_samples),
                "validation_mode": str(getattr(bundle, "validation_mode", "")),
                "walk_forward_folds": int(getattr(bundle, "walk_forward_folds", 0) or 0),
                "embargo_days": int(getattr(bundle, "embargo_days", 0) or 0),
                "feature_contract_version": str(
                    getattr(bundle, "feature_contract_version", "")
                ),
                "data_fingerprint": str(getattr(bundle, "data_fingerprint", "")),
                "feature_fingerprint": str(getattr(bundle, "feature_fingerprint", "")),
                "point_in_time_audit": dict(
                    getattr(bundle, "point_in_time_audit", {}) or {}
                ),
                "feature_availability": dict(
                    getattr(bundle, "feature_availability", {}) or {}
                ),
                "target_train_end_dates": dict(
                    getattr(bundle, "target_train_end_dates", {}) or {}
                ),
                "fold_boundaries": list(getattr(bundle, "fold_boundaries", []) or []),
                "source_manifest_sha256": str(stats.get("source_manifest_sha256", "")),
                "used_source_files": int(stats.get("used_source_files", 0) or 0),
                "candidate_model_path": str(candidate_path),
                "active_model_path": str(model_path) if model_path.exists() else "",
                "metrics": bundle.metrics.to_dict(orient="records"),
                "stats": stats,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
        return state
    except Exception as e:
        state["limitup_reason"] = f"limitup_train_error:{type(e).__name__}"
        _save_limitup_meta(
            meta_path,
            {
                "kind": "limitup_probability_engine",
                "model_version": cfg.model_version,
                "trained": False,
                "reason": state["limitup_reason"],
                "error": str(e),
                "n_samples": n_samples,
                "n_days": n_days,
                "feature_cols": feature_cols,
                "stats": stats,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )
        return state


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
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "intraday_hard_risk_flag",
        "intraday_risk_score",
        "intraday_confidence_score",
        "late_withdraw_score",
        "reseal_score",
        "open_board_count",
        "auction_strength_score",
        "factor_intraday_quality",
        "factor_intraday_confidence",
        "factor_intraday_soft_risk",
        "factor_intraday_risk",
        "factor_intraday_hard_risk",
        "factor_late_withdraw",
        "factor_reseal",
        "factor_open_board_count",
        "factor_auction_strength",
        "factor_intraday_attack_edge",
        "factor_intraday_execution_edge",
        "factor_intraday_risk_penalty",
        "intraday_attack_edge",
        "intraday_execution_edge",
        "intraday_risk_penalty",
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


def _daily_ehx_rank_ic(frame: pd.DataFrame, pred_col: str) -> float:
    values: List[float] = []
    for _, day in frame.groupby("trade_date", sort=True):
        if len(day) < 3:
            continue
        ic = _spearman_rank_ic(
            pd.to_numeric(day[pred_col], errors="coerce").to_numpy(),
            pd.to_numeric(day["real_premium_ret"], errors="coerce").to_numpy(),
        )
        if np.isfinite(ic):
            values.append(float(ic))
    return float(np.mean(values)) if values else float("nan")


def _fit_validated_ehx(
    samples: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Optional[object], object, Dict[str, object]]:
    """Train EHX with a purged date holdout and return only a validated model."""
    data = samples.copy()
    data["trade_date"] = data["trade_date"].astype(str).map(_to_yyyymmdd)
    dates = sorted(d for d in data["trade_date"].dropna().unique() if len(str(d)) == 8)
    if len(dates) < 8:
        raise ValueError(f"ehx_validation_days_not_enough:{len(dates)}<8")

    cut = max(1, int(len(dates) * 0.70))
    cut = min(cut, len(dates) - 1)
    train_cut = max(1, cut - 2)
    train_dates = set(dates[:train_cut])
    valid_dates = set(dates[cut:])
    train = data[data["trade_date"].isin(train_dates)].copy()
    valid = data[data["trade_date"].isin(valid_dates)].copy()
    if len(train) < 30 or len(valid) < 30:
        raise ValueError(f"ehx_validation_samples_not_enough:train={len(train)};valid={len(valid)}")

    X_train = _prepare_numeric_matrix(train, feature_cols)
    y_train = pd.to_numeric(train["delta_ret"], errors="coerce").fillna(0.0)
    candidate = fit_ehx_delta_regressor(X_train, y_train, feature_cols=list(X_train.columns))

    X_valid = _prepare_numeric_matrix(valid, feature_cols)
    delta_hat = np.asarray(candidate.predict(X_valid), dtype=float)
    real = pd.to_numeric(valid["real_premium_ret"], errors="coerce").to_numpy(dtype=float)
    raw = pd.to_numeric(valid["eret_pred_raw"], errors="coerce").to_numpy(dtype=float)
    plus = raw + delta_hat
    raw_mae = _mae(real, raw)
    plus_mae = _mae(real, plus)
    delta_true = real - raw
    delta_mae = _mae(delta_true, delta_hat)
    delta_rmse = _rmse(delta_true, delta_hat)
    improve_rate = float(np.mean(np.abs(real - plus) < np.abs(real - raw)))

    eval_frame = valid[["trade_date", "real_premium_ret"]].copy()
    eval_frame["raw_pred"] = raw
    eval_frame["plus_pred"] = plus
    raw_daily_ic = _daily_ehx_rank_ic(eval_frame, "raw_pred")
    plus_daily_ic = _daily_ehx_rank_ic(eval_frame, "plus_pred")
    bias = float(abs(np.nanmean(plus - real)))
    raw_bias = float(abs(np.nanmean(raw - real)))
    validation_days = int(valid["trade_date"].nunique())
    validation_samples = int(len(valid))

    rank_ok = np.isfinite(plus_daily_ic) and plus_daily_ic > 0.02
    if np.isfinite(raw_daily_ic):
        rank_ok = rank_ok and plus_daily_ic >= raw_daily_ic
    validation_pass = bool(
        validation_days >= 5
        and validation_samples >= 30
        and np.isfinite(raw_mae)
        and np.isfinite(plus_mae)
        and plus_mae <= raw_mae * 0.99
        and improve_rate >= 0.52
        and rank_ok
        and bias <= raw_bias + 0.005
    )
    reason = (
        "validation_pass"
        if validation_pass
        else (
            "validation_not_pass:"
            f"days={validation_days};samples={validation_samples};"
            f"raw_mae={raw_mae};plus_mae={plus_mae};improve_rate={improve_rate};"
            f"raw_daily_ic={raw_daily_ic};plus_daily_ic={plus_daily_ic};bias={bias}"
        )
    )
    metrics: Dict[str, object] = {
        "validation_pass": validation_pass,
        "validation_reason": reason,
        "validation_days": validation_days,
        "validation_samples": validation_samples,
        "train_end_date": max(train_dates),
        "valid_start_date": min(valid_dates),
        "raw_mae": raw_mae,
        "plus_mae": plus_mae,
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "plus_improve_rate": improve_rate,
        "raw_daily_rank_ic": raw_daily_ic,
        "plus_daily_rank_ic": plus_daily_ic,
        "plus_bias": bias,
    }

    if not validation_pass:
        return None, candidate, metrics

    X_full = _prepare_numeric_matrix(data, feature_cols)
    y_full = pd.to_numeric(data["delta_ret"], errors="coerce").fillna(0.0)
    promoted = fit_ehx_delta_regressor(X_full, y_full, feature_cols=list(X_full.columns))
    return promoted, candidate, metrics


def _train_ehx_only(
    cfg: PremiumConfig,
    ehx_samples: pd.DataFrame,
    reason_prefix: str,
    stats: Optional[Dict] = None,
    extra_train_state: Optional[Dict] = None,
) -> TrainResult:
    """只训练 EHX；用于旧 LR/LGBM 样本不足但 verify 样本可用的场景。"""
    stats = stats or {}
    extra_train_state = extra_train_state or {}
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
                **extra_train_state,
            },
        )
        return TrainResult(False, ehx_reason, n_samples, n_days, cfg.model_version)

    ehx_feature_cols = _build_ehx_feature_cols(ehx_samples)
    if not ehx_feature_cols:
        ehx_reason = "ehx_feature_cols_empty"
        _write_train_state(
            cfg,
            trade_date="unknown",
            extra={"trained": False, "reason": ehx_reason, **extra_train_state},
        )
        return TrainResult(False, ehx_reason, n_samples, n_days, cfg.model_version)

    candidate_path = cfg.out_root() / "models" / "ehx_delta_candidate.joblib"
    try:
        ehx_bundle, candidate_bundle, validation = _fit_validated_ehx(ehx_samples, ehx_feature_cols)
        save_ehx(candidate_bundle, str(candidate_path))
    except Exception as exc:
        ehx_bundle = None
        validation = {
            "validation_pass": False,
            "validation_reason": f"validation_error:{type(exc).__name__}:{exc}",
        }

    validation_pass = bool(validation.get("validation_pass", False))
    if validation_pass and ehx_bundle is not None:
        save_ehx(ehx_bundle, str(ehx_model_path))
    delta_mae = float(validation.get("delta_mae", np.nan))
    delta_rmse = float(validation.get("delta_rmse", np.nan))
    plus_improve_rate = float(validation.get("plus_improve_rate", np.nan))
    ehx_reason = "ok_validated" if validation_pass else str(validation.get("validation_reason", "validation_not_pass"))

    last_td = sorted(ehx_samples["trade_date"].dropna().astype(str).unique())[-1] if n_days > 0 else "unknown"

    _save_ehx_meta(
        ehx_meta_path,
        {
            "kind": "ehx_delta_regressor",
            "model_version": cfg.model_version,
            "trained": validation_pass,
            "validation_pass": validation_pass,
            "reason": ehx_reason,
            "source": reason_prefix,
            "feature_cols": list(ehx_feature_cols),
            "n_samples": n_samples,
            "n_days": n_days,
            "candidate_model_path": str(candidate_path),
            "active_model_path": str(ehx_model_path) if ehx_model_path.exists() else "",
            **validation,
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
        "ehx_trained": int(validation_pass),
        "ehx_reason": ehx_reason,
        "ehx_n_samples": n_samples,
        "ehx_min_samples": ehx_min_samples,
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "plus_improve_rate": plus_improve_rate,
        "ehx_validation_pass": int(bool(validation.get("validation_pass", False))),
        "ehx_plus_daily_rank_ic": validation.get("plus_daily_rank_ic", np.nan),
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
            "trained": validation_pass,
            "reason": ehx_reason,
            "n_samples": n_samples,
            "n_days": n_days,
            "ehx_trained": validation_pass,
            "ehx_reason": ehx_reason,
            "ehx_n_samples": n_samples,
            "ehx_min_samples": ehx_min_samples,
            "ehx_model_path": str(ehx_model_path) if validation_pass else "",
            "ehx_candidate_path": str(candidate_path),
            "ehx_meta_path": str(ehx_meta_path),
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "plus_improve_rate": plus_improve_rate,
            "ehx_validation_pass": bool(validation.get("validation_pass", False)),
            "ehx_plus_daily_rank_ic": validation.get("plus_daily_rank_ic", np.nan),
            **extra_train_state,
        },
    )
    return TrainResult(validation_pass, ehx_reason, n_samples, n_days, cfg.model_version)


# ========= 主训练入口 =========

def train_models(cfg: Optional[PremiumConfig] = None) -> TrainResult:
    """训练入口（供 scripts / workflow 调用）。"""
    cfg = cfg or PremiumConfig.load()

    samples, stats = collect_training_samples(cfg)
    verify_samples, verify_stats = _collect_ehx_samples_from_verify_outputs(cfg)
    limitup_samples, limitup_stats = _collect_limitup_samples_from_verify_outputs(cfg)
    limitup_state = _train_limitup_probability_from_verify(cfg, limitup_samples, limitup_stats)

    if samples.empty:
        if not verify_samples.empty:
            return _train_ehx_only(
                cfg,
                verify_samples,
                reason_prefix="ehx_from_premium_verify",
                stats={"decision": stats, "verify": verify_stats},
                extra_train_state=limitup_state,
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
                **limitup_state,
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
                extra_train_state=limitup_state,
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
                **limitup_state,
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
            "intraday_quality_score",
            "intraday_soft_risk_score",
            "intraday_hard_risk_flag",
            "intraday_risk_score",
            "intraday_confidence_score",
            "late_withdraw_score",
            "reseal_score",
            "open_board_count",
            "auction_strength_score",
            "factor_intraday_attack_edge",
            "factor_intraday_execution_edge",
            "factor_intraday_risk_penalty",
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
    ehx_candidate_path = cfg.out_root() / "models" / "ehx_delta_candidate.joblib"
    ehx_meta_path = cfg.out_root() / "models" / "ehx_meta.json"
    ehx_bundle = None
    ehx_validation: Dict[str, object] = {}

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
        try:
            ehx_bundle, candidate_bundle, ehx_validation = _fit_validated_ehx(ehx_samples, ehx_feature_cols)
            save_ehx(candidate_bundle, str(ehx_candidate_path))
        except Exception as exc:
            ehx_bundle = None
            ehx_validation = {
                "validation_pass": False,
                "validation_reason": f"validation_error:{type(exc).__name__}:{exc}",
            }
        ehx_trained = bool(ehx_validation.get("validation_pass", False) and ehx_bundle is not None)
        ehx_reason = "ok_validated" if ehx_trained else str(ehx_validation.get("validation_reason", "validation_not_pass"))
        delta_mae = float(ehx_validation.get("delta_mae", np.nan))
        delta_rmse = float(ehx_validation.get("delta_rmse", np.nan))
        if ehx_trained and ehx_bundle is not None:
            save_ehx(ehx_bundle, str(ehx_model_path))

        _save_ehx_meta(
            ehx_meta_path,
            {
                "kind": "ehx_delta_regressor",
                "model_version": cfg.model_version,
                "trained": ehx_trained,
                "validation_pass": ehx_trained,
                "reason": ehx_reason,
                "source": "decision_plus_verify" if not verify_samples.empty else "decision",
                "feature_cols": list(ehx_feature_cols),
                "n_samples": int(len(ehx_samples)),
                "n_days": int(ehx_samples["trade_date"].nunique()) if "trade_date" in ehx_samples.columns else 0,
                "candidate_model_path": str(ehx_candidate_path),
                "active_model_path": str(ehx_model_path) if ehx_model_path.exists() else "",
                **ehx_validation,
                "created_at_utc": utc_now_iso(),
                "commit_sha": get_commit_sha(cfg.repo_root()),
                "run_id": get_run_id(),
            },
        )

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

    plus_improve_rate = float(ehx_validation.get("plus_improve_rate", np.nan))
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
        "limitup_trained": int(bool(limitup_state.get("limitup_trained", False))),
        "limitup_reason": limitup_state.get("limitup_reason", ""),
        "limitup_n_samples": int(limitup_state.get("limitup_n_samples", 0) or 0),
        "limitup_n_days": int(limitup_state.get("limitup_n_days", 0) or 0),
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
            "ehx_candidate_path": str(ehx_candidate_path),
            "ehx_meta_path": str(ehx_meta_path),
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "plus_improve_rate": plus_improve_rate,
            "ehx_validation_pass": bool(ehx_validation.get("validation_pass", False)),
            "ehx_plus_daily_rank_ic": ehx_validation.get("plus_daily_rank_ic", np.nan),
            **limitup_state,
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
        "limitup_model": str(cfg.out_root() / "models" / "limitup_probability_engine.joblib"),
        "limitup_meta": str(cfg.out_root() / "models" / "limitup_probability_engine_meta.json"),
        "limitup_metrics": str(cfg.out_root() / "models" / "limitup_probability_engine_metrics.csv"),
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
        print(f"[premium][train] limitup_model={payload['limitup_model']}")
        print(f"[premium][train] limitup_meta={payload['limitup_meta']}")
        print(f"[premium][train] limitup_metrics={payload['limitup_metrics']}")
    return 0 if result.trained or "不足" in result.reason or "没有可用样本" in result.reason else 1


__all__ = ["TrainResult", "train_models", "collect_training_samples", "fit_ehx_delta_regressor", "save_ehx"]


if __name__ == "__main__":
    raise SystemExit(_main())
