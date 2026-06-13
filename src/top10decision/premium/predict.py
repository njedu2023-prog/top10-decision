#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Predict（V3.6：涨停接力专业概率引擎接线版）

当前锁死契约（Premium Contract V3）：

- 主目标不再是“好看报告”，而是围绕 E_ret 生成更强的 E_ret_plus。
- 当前只动 premium 线，不动 decision 主线文件。
- 旧 V2 分位链保留，但中心值优先围绕 eret_plus_value 展开。
- 必须保留 raw / plus 后验误差对比，支撑闭环验收。

本版关键修复 / 增强：

- 与 train.py 对齐原始 E_ret 字段识别口径，避免训练能识别、推理识别不到。
- EHX 推理优先加载 outputs/premium/models/ehx_delta.joblib。
- EHX 加载/预测失败不阻断主流程，但会在 eret_plus_src 与 _last_run.txt 中追溯失败原因。
- eret_pred_raw / eret_plus_value / eret_plus_delta 全链路落盘，供 report / verify / 后续 decision 接线使用。
- 修复 _write_last_run 前 eret_plus_src 兜底 Series 长度不匹配问题。
- 修复 _zscore 全空输入触发 RuntimeWarning 的问题。
- 新增实盘执行字段，先进入 premium_top30 / premium_full / premium_verify CSV：
  buy_date、T日涨停概率、T日涨停强度、T+1延续上涨率、涨停接力评分、T+1建议买入方式。
- V3.6 新增：接入 limitup_probability_engine.py 专业概率模型。
  有模型时输出模型概率并参与排序；无模型/加载失败时自动回退 V3.5 规则评分，不阻断 Premium 主流程。

主输入：

- data/pred/pred_source_latest.csv（a-top10 全量源表，候选=全量，不过滤）

输出：

- outputs/premium/premium_top30_{T}.csv
- outputs/premium/premium_full_{T}.csv
- outputs/premium/premium_verify_{T}.csv
- docs/reports/premium_{T}.md
- docs/reports/premium_latest.md
- outputs/premium/_last_run.txt（每次覆盖）
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .audit import make_audit_block_md, make_audit_kv
from .config import PremiumConfig
from .factor_builders import build_features_by_packs
from .factor_registry import detect_factor_packs
from .market_truth import ensure_daily_cached, load_daily
from .premium_views import add_rank_groups, attach_limitup_validation, render_premium_report_html
from .report_md import render_premium_report_md

try:
    from .limitup_probability_engine import load_bundle as _load_limitup_probability_bundle
except Exception:  # pragma: no cover
    _load_limitup_probability_bundle = None  # type: ignore

_TD_RE = re.compile(r"^\d{8}$")
_PREMIUM_REPORT_RE = re.compile(r"^premium_(20\d{6})\.html$")


@dataclass(frozen=True)
class PredictResult:
    ok: bool
    trade_date: str
    target_date: Optional[str]
    pending: bool
    reason: str
    out_top10: Optional[str] = None
    out_top20: Optional[str] = None
    out_top30: Optional[str] = None
    out_full: Optional[str] = None
    out_verify: Optional[str] = None
    report_md: Optional[str] = None
    report_html: Optional[str] = None


# ========= 通用工具 =========

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_run_id() -> str:
    rid = os.getenv("GITHUB_RUN_ID", "").strip()
    rno = os.getenv("GITHUB_RUN_NUMBER", "").strip()
    if rid:
        return f"gh_{rid}"
    if rno:
        return f"ghno_{rno}"
    return datetime.now(timezone.utc).strftime("local_%Y%m%d%H%M%S")


def _get_commit_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12] if out else "unknown"
    except Exception:
        return "unknown"


def _to_yyyymmdd(x: object) -> str:
    s = str(x).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "")
    return s[:8]


def _read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        return pd.read_csv(path)
    return pd.read_csv(path)


def _ensure_dirs(cfg: PremiumConfig) -> None:
    cfg.out_root().mkdir(parents=True, exist_ok=True)
    cfg.reports_root().mkdir(parents=True, exist_ok=True)
    try:
        cfg.out_learning_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_last_run(cfg: PremiumConfig, trade_date: str, extra: Dict[str, object]) -> None:
    _ensure_dirs(cfg)
    lines = [
        f"trade_date: {trade_date}",
        f"run_id: {_get_run_id()}",
        f"commit_sha: {_get_commit_sha(cfg.repo_root())}",
        f"model_version: {getattr(cfg, 'model_version', 'unknown')}",
        f"created_at_utc: {_utc_now_iso()}",
    ]
    for k, v in extra.items():
        lines.append(f"{k}: {v}")
    _write_text(cfg.out_last_run_path(), "\n".join(lines) + "\n")


def _list_report_dates(cfg: PremiumConfig, current_trade_date: str) -> List[str]:
    dates = {str(current_trade_date)}
    try:
        for p in cfg.reports_root().glob("premium_*.html"):
            m = _PREMIUM_REPORT_RE.match(p.name)
            if m:
                dates.add(m.group(1))
    except Exception:
        pass
    return sorted(d for d in dates if _TD_RE.match(str(d)))


def _rate_from_hits(hits: int, total: int) -> float:
    return float(hits) / float(total) if int(total) > 0 else float("nan")


def _bool_like_series(df: pd.DataFrame, names: List[str], default: float = np.nan) -> pd.Series:
    c = _first_existing_col(df, *names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    num = pd.to_numeric(df[c], errors="coerce")
    if num.notna().any():
        return num.clip(lower=0, upper=1)
    raw = df[c].astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[raw.isin({"1", "true", "yes", "y", "是", "命中", "hit", "up"})] = 1.0
    out.loc[raw.isin({"0", "false", "no", "n", "否", "未命中", "miss", "down"})] = 0.0
    return out.fillna(default)


def _historical_limitup_stats_from_df(df: pd.DataFrame, source: str) -> Dict[str, object]:
    """Summarize cumulative historical TOP10/TOP20 limit-up hit rates."""
    if df is None or df.empty:
        return {"ready": False, "reason": "history_empty", "source": source}

    rank = _num_series(df, "rank", "dec_rank", default=np.nan)
    actual = _bool_like_series(df, ["t_limitup_hit", "t_limitup_actual"], default=np.nan)
    ready = _bool_like_series(df, ["label_matured", "t_limitup_verify_ready"], default=np.nan)
    if ready.notna().any():
        valid = ready.fillna(0).eq(1) & actual.notna() & rank.notna()
    else:
        valid = actual.notna() & rank.notna()

    if not valid.any():
        return {"ready": False, "reason": "no_ready_history_rows", "source": source}

    def calc(n: int) -> Tuple[int, int, float]:
        m = valid & (rank <= n)
        total = int(m.sum())
        hits = int(actual[m].eq(1).sum())
        return total, hits, _rate_from_hits(hits, total)

    top10_total, top10_hits, top10_rate = calc(10)
    top20_total, top20_hits, top20_rate = calc(20)

    date_col = _first_existing_col(df, "d_trade_date", "trade_date", "base_date", "d_analysis_trade_date")
    n_days = 0
    if date_col is not None:
        dates = df.loc[valid, date_col].astype(str).map(_to_yyyymmdd)
        n_days = int(dates[dates.str.match(r"^20\d{6}$", na=False)].nunique())

    return {
        "ready": bool(top10_total > 0 or top20_total > 0),
        "reason": "ok",
        "source": source,
        "n_days": n_days,
        "top10_total": top10_total,
        "top10_hits": top10_hits,
        "top10_hit_rate": top10_rate,
        "top20_total": top20_total,
        "top20_hits": top20_hits,
        "top20_hit_rate": top20_rate,
    }


def _collect_historical_limitup_stats(cfg: PremiumConfig) -> Dict[str, object]:
    """
    Prefer the limit-up training sample set because it includes backfilled historical labels.
    Fall back to raw premium_verify files when the training set has not been generated yet.
    """
    trainset_path = cfg.out_learning_dir() / "limitup_probability_training_samples.csv"
    if trainset_path.exists():
        try:
            df = _read_csv_smart(trainset_path)
            stats = _historical_limitup_stats_from_df(df, "limitup_probability_training_samples.csv")
            if stats.get("ready"):
                return stats
        except Exception as e:
            return {"ready": False, "reason": f"trainset_read_error:{type(e).__name__}", "source": str(trainset_path.name)}

    rows: List[pd.DataFrame] = []
    for p in sorted(cfg.out_root().glob("premium_verify_*.csv")):
        try:
            d = _read_csv_smart(p)
            if not d.empty:
                if "d_trade_date" not in d.columns:
                    m = re.search(r"(20\d{6})", p.name)
                    d["d_trade_date"] = m.group(1) if m else ""
                rows.append(d)
        except Exception:
            continue
    if not rows:
        return {"ready": False, "reason": "no_history_verify_files", "source": "premium_verify_*.csv"}
    return _historical_limitup_stats_from_df(pd.concat(rows, ignore_index=True, sort=False), "premium_verify_*.csv")


def _rebuild_rank_front(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank" in df.columns:
        df = df.drop(columns=["rank"], errors="ignore")
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if len(x) == 0 or x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def _norm_ppf(q: float) -> float:
    """标准正态分位点近似，避免引入 scipy 依赖。"""
    import math

    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1 - plow

    if q <= 0.0 or q >= 1.0:
        return float("nan")
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    elif q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        x = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    else:
        r = q - 0.5
        s = r * r
        x = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
    return float(x)


def _norm_cdf(x: object) -> float:
    """标准正态分布 CDF，避免引入 scipy 依赖。"""
    import math

    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(v / math.sqrt(2.0))))


def _first_existing_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        hit = cols.get(str(n).strip().lower())
        if hit is not None:
            return hit
    return None


def _num_series(df: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    c = _first_existing_col(df, *names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce")


# ========= 交易日推进（必须用交易日历，不得用未来行情探测） =========

def _get_tushare_token() -> str:
    return (os.getenv("TUSHARE_TOKEN", "") or "").strip()


def _tushare_trade_cal_open_days(token: str, start_date: str, end_date: str) -> Optional[List[str]]:
    """返回 [YYYYMMDD, ...] 交易日列表（is_open=1）。"""
    try:
        import requests  # type: ignore
    except Exception:
        return None

    payload = {
        "api_name": "trade_cal",
        "token": token,
        "params": {"exchange": "SSE", "start_date": start_date, "end_date": end_date, "is_open": "1"},
        "fields": "cal_date,is_open",
    }
    try:
        r = requests.post("https://api.tushare.pro", json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, dict):
            return None
        data = j.get("data") or {}
        items = data.get("items") or []
        fields = data.get("fields") or []
        if not items or not fields:
            return None
        df = pd.DataFrame(items, columns=fields)
        if "cal_date" not in df.columns:
            return None
        df["cal_date"] = df["cal_date"].astype(str).map(_to_yyyymmdd)
        return sorted([d for d in df["cal_date"].tolist() if _TD_RE.match(str(d) or "")])
    except Exception:
        return None


def _advance_trade_days_by_trade_cal(trade_date: str, steps: int) -> Tuple[Optional[str], str]:
    import datetime as dt

    td = _to_yyyymmdd(trade_date)
    if not _TD_RE.match(td):
        return None, "bad_trade_date"

    token = _get_tushare_token()
    if not token:
        return None, "missing_TUSHARE_TOKEN"

    d0 = dt.datetime.strptime(td, "%Y%m%d").date()
    end = (d0 + dt.timedelta(days=120)).strftime("%Y%m%d")
    days = _tushare_trade_cal_open_days(token, td, end)
    if not days:
        return None, "trade_cal_empty"

    if td not in days:
        after = [x for x in days if x > td]
        if not after:
            return None, "trade_cal_no_future_open_day"
        td = after[0]
        steps = max(0, int(steps) - 1)

    try:
        i0 = days.index(td)
    except ValueError:
        return None, "trade_cal_index_fail"

    i1 = i0 + int(steps)
    if i1 >= len(days):
        return None, "trade_cal_out_of_range"
    return days[i1], "ok"


def _advance_trade_days_fallback_business_day(trade_date: str, steps: int) -> Tuple[str, str]:
    import datetime as dt

    td = _to_yyyymmdd(trade_date)
    d = dt.datetime.strptime(td, "%Y%m%d").date()
    n = 0
    while n < int(steps):
        d = d + dt.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d.strftime("%Y%m%d"), "fallback_business_day"


def _advance_trade_days(cfg: PremiumConfig, trade_date: str, steps: int) -> Tuple[str, str]:
    td, reason = _advance_trade_days_by_trade_cal(trade_date, steps)
    if td:
        return td, "trade_cal_ok"
    return "", f"strict_a_share_trade_cal_failed:{reason}"


# ========= pred_source_latest 读取与字段兜底 =========

def _normalize_pred_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols_map:
                return cols_map[n.lower()]
        return None

    c_date = pick("trade_date", "date", "dt", "交易日期", "日期")
    c_code = pick("ts_code", "code", "symbol", "ticker", "股票代码", "代码")
    c_name = pick("name", "stock_name", "股票名称", "名称")

    if c_date:
        df["trade_date"] = df[c_date].astype(str).map(_to_yyyymmdd)
    if c_code:
        df["ts_code"] = df[c_code].astype(str).str.strip()
    if c_name:
        df["name"] = df[c_name].astype(str).str.strip()
    return df


def _infer_trade_date(df: pd.DataFrame) -> str:
    if "trade_date" in df.columns:
        s = df["trade_date"].dropna().astype(str).map(_to_yyyymmdd)
        s = s[s.str.match(r"^\d{8}$", na=False)]
        if not s.empty:
            return sorted(s.unique().tolist())[-1]
    for c in df.columns:
        s = df[c].dropna().astype(str).map(_to_yyyymmdd)
        s = s[s.str.match(r"^\d{8}$", na=False)]
        if not s.empty:
            return sorted(s.unique().tolist())[-1]
    return "unknown"


def _extract_raw_eret(df: pd.DataFrame) -> pd.Series:
    """与 train.py 对齐的原始 E_ret 字段识别口径。"""
    return _num_series(
        df,
        "eret_pred_raw",
        "e_ret_pred_raw",
        "raw_eret_pred",
        "raw_e_ret_pred",
        "eret_pred",
        "e_ret_pred",
        "E_ret",
        "e_ret",
        "eret_pred_final",
        "e_premium",
        "pred_ret",
        "pred_return",
        "ret",
        "premium_ret",
        "pred_premium_ret",
        "pred_ret_mean",
        "eret_plus",
        "e_ret_plus",
        "E_ret_plus",
        "eret_plus_pred",
        "e_ret_plus_pred",
        default=np.nan,
    )


def _pick_pred_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    c_prob = _first_existing_col(df, "p_premium", "up_prob", "probability", "prob", "p")
    c_score = _first_existing_col(df, "score_ev", "ev", "final_score", "score", "pred_ev")
    c_conf = _first_existing_col(df, "confidence", "conf")
    c_dq = _first_existing_col(df, "data_quality", "dq")
    c_risk = _first_existing_col(df, "risk_flags", "risk", "warning", "risk_hint", "fill_risk_hint")

    p = pd.to_numeric(df[c_prob], errors="coerce") if c_prob else pd.Series([np.nan] * len(df), index=df.index)
    p = p.fillna(0.5).clip(0.0, 1.0)
    e = _extract_raw_eret(df).fillna(0.0)
    if c_score:
        s = pd.to_numeric(df[c_score], errors="coerce").fillna(p * e)
    else:
        s = (p * e).astype(float)
    conf = pd.to_numeric(df[c_conf], errors="coerce") if c_conf else pd.Series([pd.NA] * len(df), index=df.index)
    dq = pd.to_numeric(df[c_dq], errors="coerce") if c_dq else pd.Series([pd.NA] * len(df), index=df.index)
    risk = df[c_risk].astype(str) if c_risk else pd.Series([""] * len(df), index=df.index)
    return p, e, s, conf, dq, risk


# ========= decision merge（仅标签，不过滤） =========

def _load_decision_merge(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    repo_root = cfg.repo_root()
    paths = [Path(p).resolve() for p in glob.glob(str((repo_root / cfg.decision_glob).resolve()))]
    if not paths:
        return pd.DataFrame()

    hit = []
    for p in sorted(paths):
        try:
            d = _read_csv_smart(p)
            if d is None or d.empty:
                continue
            d = _normalize_pred_source(d)
            if "trade_date" not in d.columns or "ts_code" not in d.columns:
                continue
            if (d["trade_date"].astype(str) == str(trade_date)).any():
                hit.append(d)
        except Exception:
            continue

    if not hit:
        return pd.DataFrame()

    dec = pd.concat(hit, ignore_index=True)
    dec["trade_date"] = dec["trade_date"].astype(str).map(_to_yyyymmdd)
    dec["ts_code"] = dec["ts_code"].astype(str).str.strip()
    if "name" in dec.columns:
        dec["name"] = dec["name"].astype(str).str.strip()

    out = pd.DataFrame({"trade_date": dec["trade_date"].astype(str), "ts_code": dec["ts_code"].astype(str)})
    if "name" in dec.columns:
        out["name"] = dec["name"]

    for out_col, names in {
        "dec_rank": ("dec_rank", "decision_rank", "rank", "决策排名"),
        "dec_weight": ("dec_weight", "weight", "target_weight", "决策权重"),
        "dec_can_buy": ("dec_can_buy", "can_buy", "可买提示"),
        "dec_p_fill": ("dec_p_fill", "p_fill", "p_fill_pred", "p_fill_pred_final"),
        "dec_reason": ("dec_reason", "reason", "label", "决策原因", "决策标签"),
    }.items():
        c = _first_existing_col(dec, *names)
        out[out_col] = dec[c] if c else pd.NA

    return out.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)


# ========= EHX（优先加载训练模型，失败再回退冷启动） =========

def _ehx_model_path(cfg: PremiumConfig) -> Path:
    return cfg.out_root() / "models" / "ehx_delta.joblib"


def _ehx_meta_path(cfg: PremiumConfig) -> Path:
    return cfg.out_root() / "models" / "ehx_meta.json"


def _load_ehx_bundle(cfg: PremiumConfig) -> Tuple[Optional[dict], str]:
    path = _ehx_model_path(cfg)
    if not path.exists():
        return None, "ehx_model_missing"
    try:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            return None, "ehx_model_bad_bundle"
        model = obj.get("model")
        feature_cols = obj.get("feature_cols")
        if model is None or not isinstance(feature_cols, (list, tuple)) or not feature_cols:
            return None, "ehx_model_missing_model_or_features"
        meta = {}
        mp = _ehx_meta_path(cfg)
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return {"model": model, "feature_cols": list(feature_cols), "path": str(path), "meta": meta}, "ok"
    except Exception as e:
        return None, f"ehx_model_load_error:{type(e).__name__}"


def _build_ehx_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["eret_pred_raw"] = _extract_raw_eret(df).fillna(0.0)
    out["p_fill_pred"] = _num_series(df, "p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill", default=np.nan).fillna(0.5)
    out["cost_total"] = _num_series(df, "cost_total", "cost", "cost_value", "cost_all", "trade_cost", default=np.nan).fillna(0.0)
    out["risk_penalty_total"] = _num_series(df, "risk_penalty_total", "risk_penalty", "riskpenalty", "risk_penalty_score", "risk_score", default=np.nan).fillna(0.0)
    out["ev"] = _num_series(df, "score_ev", "ev", "pred_ev", default=np.nan).fillna(0.0)
    out["turnover_rate"] = _num_series(df, "turnover_rate", "换手率", default=np.nan).fillna(0.0)
    out["amount"] = _num_series(df, "amount", "成交额", default=np.nan).fillna(0.0)
    out["vol"] = _num_series(df, "vol", "volume", "成交量", default=np.nan).fillna(0.0)
    out["close"] = _num_series(df, "close", "close_T", "收盘价", default=np.nan).fillna(0.0)
    out["pct_chg"] = _num_series(df, "pct_chg", "pct_change", "涨跌幅", default=np.nan).fillna(0.0)
    out["amplitude"] = _num_series(df, "amplitude", "range_1d", "振幅", default=np.nan).fillna(0.0)

    for c in feature_cols:
        if c in out.columns:
            continue
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = 0.0

    out = out.reindex(columns=list(feature_cols), fill_value=0.0)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _infer_conf_from_inputs(out: pd.DataFrame, eret_raw: pd.Series) -> Tuple[pd.Series, pd.Series]:
    p_fill = _num_series(out, "p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill", default=np.nan).fillna(0.5).clip(0.0, 1.0)
    ev = _num_series(out, "score_ev", "ev", "pred_ev", default=np.nan).fillna(0.0)
    cost = _num_series(out, "cost_total", "cost", "cost_value", "cost_all", "trade_cost", default=np.nan).fillna(0.0)
    risk_pen = _num_series(out, "risk_penalty_total", "risk_penalty", "riskpenalty", "risk_penalty_score", "risk_score", default=np.nan).fillna(0.0)
    ret_5d = _num_series(out, "ret_5d", default=np.nan).fillna(0.0)
    vol_10d = _num_series(out, "vol_10d", default=np.nan).fillna(0.0)
    range_1d = _num_series(out, "range_1d", default=np.nan).fillna(0.0)
    amount_z_5d = _num_series(out, "amount_z_5d", default=np.nan).fillna(1.0)
    close_pos_n = _num_series(out, "close_pos_n", "close_pos_10d", "close_pos_20d", default=np.nan).fillna(0.5)

    amount_dev = (amount_z_5d - 1.0).abs().clip(0.0, 3.0)
    crowded_penalty = (close_pos_n - 0.70).clip(lower=0.0)
    input_cols = pd.DataFrame({
        "eret_raw": eret_raw,
        "p_fill": p_fill,
        "ev": ev,
        "cost": cost,
        "risk_pen": risk_pen,
        "ret_5d": ret_5d,
        "vol_10d": vol_10d,
        "range_1d": range_1d,
    })
    completeness = input_cols.notna().mean(axis=1).astype(float)
    stability = (
        1.0
        - 0.45 * vol_10d.clip(lower=0.0, upper=0.25)
        - 0.30 * range_1d.clip(lower=0.0, upper=0.20)
        - 0.10 * amount_dev.clip(lower=0.0, upper=1.0)
        - 0.10 * crowded_penalty.clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)
    conf_score = (0.55 * completeness + 0.45 * stability).clip(lower=0.0, upper=1.0)

    def to_conf_label(x: float) -> str:
        if x >= 0.72:
            return "high"
        if x >= 0.50:
            return "mid"
        return "low"

    return conf_score.apply(lambda x: to_conf_label(float(x)) if pd.notna(x) else "low"), conf_score.round(6)


def _build_ehx_v1(cfg: PremiumConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """优先加载训练后的 EHX 残差模型；失败则回退冷启动增强器。"""
    out = df.copy()
    eret_raw = _extract_raw_eret(out).fillna(0.0)
    trace: Dict[str, object] = {"ehx_mode": "coldstart", "ehx_reason": "not_run", "ehx_model_path": str(_ehx_model_path(cfg))}

    bundle, load_reason = _load_ehx_bundle(cfg)
    trace["ehx_load_reason"] = load_reason

    if bundle is not None:
        try:
            X_ehx = _build_ehx_feature_frame(out, bundle["feature_cols"])
            delta_hat = pd.Series(bundle["model"].predict(X_ehx), index=out.index, dtype="float64")
            delta_hat = pd.to_numeric(delta_hat, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.12, upper=0.12)
            eret_plus = (eret_raw + delta_hat).clip(lower=-0.95, upper=2.0)
            conf_label, conf_score = _infer_conf_from_inputs(out, eret_raw)
            eps = 0.002
            direction = pd.Series(np.where(delta_hat > eps, "up", np.where(delta_hat < -eps, "down", "flat")), index=out.index, dtype="object")

            out["eret_pred_raw"] = eret_raw
            out["eret_plus_value"] = eret_plus
            out["eret_plus_delta"] = delta_hat
            out["eret_plus_direction"] = direction
            out["eret_plus_conf"] = conf_label
            out["eret_plus_conf_score"] = conf_score
            out["eret_plus_src"] = "ehx:model_v1"
            trace.update({
                "ehx_mode": "model_v1",
                "ehx_reason": "ok",
                "ehx_feature_n": len(bundle.get("feature_cols", [])),
                "ehx_meta_n_samples": (bundle.get("meta") or {}).get("n_samples", ""),
                "ehx_meta_delta_mae": (bundle.get("meta") or {}).get("delta_mae", ""),
                "ehx_meta_delta_rmse": (bundle.get("meta") or {}).get("delta_rmse", ""),
            })
            return out, trace
        except Exception as e:
            trace.update({"ehx_mode": "coldstart", "ehx_reason": f"ehx_predict_error:{type(e).__name__}"})
    else:
        trace.update({"ehx_mode": "coldstart", "ehx_reason": load_reason})

    # 冷启动增强器：保留旧逻辑，作为 EHX 模型缺失/失败时的安全兜底。
    p_prob = pd.to_numeric(out.get("p_premium", pd.Series([0.5] * len(out), index=out.index)), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    p_fill = _num_series(out, "p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill", default=np.nan).fillna(0.5).clip(0.0, 1.0)
    ev = _num_series(out, "score_ev", "ev", "pred_ev", default=np.nan).fillna(0.0)
    cost = _num_series(out, "cost_total", "cost", "cost_value", "cost_all", "trade_cost", default=np.nan).fillna(0.0)
    risk_pen = _num_series(out, "risk_penalty_total", "risk_penalty", "riskpenalty", "risk_penalty_score", "risk_score", default=np.nan).fillna(0.0)
    ret_5d = _num_series(out, "ret_5d", default=np.nan).fillna(0.0)
    vol_10d = _num_series(out, "vol_10d", default=np.nan).fillna(0.0)
    range_1d = _num_series(out, "range_1d", default=np.nan).fillna(0.0)
    amount_z_5d = _num_series(out, "amount_z_5d", default=np.nan).fillna(1.0)
    f_strength = _num_series(out, "f_strength", default=np.nan).fillna(0.0)
    f_theme = _num_series(out, "f_theme", default=np.nan).fillna(0.0)
    close_pos_n = _num_series(out, "close_pos_n", "close_pos_10d", "close_pos_20d", default=np.nan).fillna(0.5)

    liquidity_edge = (p_fill - 0.5) * 2.0
    amount_dev = (amount_z_5d - 1.0).abs().clip(0.0, 3.0)
    crowded_penalty = (close_pos_n - 0.70).clip(lower=0.0)
    delta = (
        0.0100 * _zscore(ev)
        + 0.0055 * _zscore(p_prob)
        + 0.0075 * _zscore(ret_5d)
        + 0.0040 * _zscore(f_strength)
        + 0.0025 * _zscore(f_theme)
        + 0.0070 * liquidity_edge
        - 0.0100 * _zscore(cost)
        - 0.0120 * _zscore(risk_pen)
        - 0.0060 * _zscore(vol_10d)
        - 0.0040 * _zscore(range_1d)
        - 0.0040 * crowded_penalty
        - 0.0020 * amount_dev
    )
    delta += eret_raw.clip(lower=-0.20, upper=0.20) * 0.08
    delta = pd.to_numeric(delta, errors="coerce").fillna(0.0).clip(lower=-0.08, upper=0.08)
    eret_plus = (eret_raw + delta).clip(lower=-0.95, upper=2.0)
    conf_label, conf_score = _infer_conf_from_inputs(out, eret_raw)
    eps = 0.002
    direction = pd.Series(np.where(delta > eps, "up", np.where(delta < -eps, "down", "flat")), index=out.index, dtype="object")

    out["eret_pred_raw"] = eret_raw
    out["eret_plus_value"] = eret_plus
    out["eret_plus_delta"] = delta
    out["eret_plus_direction"] = direction
    out["eret_plus_conf"] = conf_label
    out["eret_plus_conf_score"] = conf_score
    out["eret_plus_src"] = "ehx:coldstart_v1"
    return out, trace


# ========= V2/V3 分布预测 =========

def _build_mu_sigma(cfg: PremiumConfig, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    base_sigma = float(getattr(cfg, "base_sigma", 0.05))
    score_scale = float(getattr(cfg, "score_scale", 0.012))
    e_plus = pd.to_numeric(df.get("eret_plus_value", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    e_raw = _extract_raw_eret(df)
    e_core = e_plus.where(e_plus.notna(), e_raw)
    mu_from_e = np.log1p(e_core.clip(lower=-0.99))
    z_prob = _zscore(df.get("f_prob", df.get("p_premium", pd.Series([np.nan] * len(df), index=df.index))))
    z_strength = _zscore(df.get("f_strength", pd.Series([np.nan] * len(df), index=df.index)))
    z_theme = _zscore(df.get("f_theme", pd.Series([np.nan] * len(df), index=df.index)))
    z_mom = _zscore(df.get("ret_5d", pd.Series([np.nan] * len(df), index=df.index)))
    mu_pack = score_scale * (0.55 * z_prob + 0.30 * z_strength + 0.15 * z_theme + 0.20 * z_mom)
    mu = mu_from_e.where(np.isfinite(mu_from_e.values), mu_pack)
    mu = pd.to_numeric(mu, errors="coerce").fillna(0.0)

    vol10 = pd.to_numeric(df.get("vol_10d", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    range1 = pd.to_numeric(df.get("range_1d", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    az5 = pd.to_numeric(df.get("amount_z_5d", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    az_dev = (az5 - 1.0).abs()
    sigma = base_sigma * (
        1.0
        + 2.0 * vol10.fillna(0.0).clip(lower=0.0, upper=0.25)
        + 1.2 * range1.fillna(0.0).clip(lower=0.0, upper=0.20)
        + 0.6 * az_dev.fillna(0.0).clip(lower=0.0, upper=3.0)
    )
    sigma = sigma.clip(lower=1e-6, upper=0.5)
    return mu.astype(float), sigma.astype(float)


def _compute_quantile_returns(cfg: PremiumConfig, df: pd.DataFrame) -> pd.DataFrame:
    qs = tuple(getattr(cfg, "quantiles", (0.05, 0.25, 0.50, 0.75, 0.95)))
    mu, sigma = _build_mu_sigma(cfg, df)
    out = df.copy()

    # T+1上涨率使用同一套 D→T+1 分布口径计算：P(r_target > 0)。
    # 不再直接沿用 p_premium，避免出现“预期收益和价格区间明显上涨，但上涨率很低”的口径冲突。
    z_up = (mu / sigma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["t1_up_rate"] = z_up.map(_norm_cdf).clip(lower=0.0, upper=1.0).fillna(0.5)
    out["T+1上涨率"] = out["t1_up_rate"]

    for q in qs:
        z = _norm_ppf(float(q))
        out[f"r_p{int(round(q * 100)):02d}"] = mu + sigma * z

    close_T = pd.to_numeric(out.get("close_T", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    for q in qs:
        key = f"r_p{int(round(q * 100)):02d}"
        # 底层保留 close_T2_* 旧列名兼容历史代码；报告层展示为 T+1 预测到期价格。
        out[f"close_T2_p{int(round(q * 100)):02d}"] = close_T * np.exp(pd.to_numeric(out[key], errors="coerce"))
    return out


# ========= 实盘执行层 =========

def _fmt_price_value(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0:
        return ""
    if v >= 100:
        return f"{v:.2f}"
    return f"{v:.2f}"


def _rank_pct(s: pd.Series, neutral: float = 0.50) -> pd.Series:
    """把任意连续因子转成 0~1 横截面分位分。全空/常数时回到 neutral。"""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(x) == 0 or x.notna().sum() == 0:
        return pd.Series([neutral] * len(x), index=x.index, dtype="float64")
    if x.nunique(dropna=True) <= 1:
        return pd.Series([neutral] * len(x), index=x.index, dtype="float64")
    return x.rank(method="average", pct=True).fillna(neutral).clip(0.0, 1.0).astype(float)


def _sigmoid_series(s: pd.Series) -> pd.Series:
    """稳定 sigmoid，输出 0~1。"""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)
    return (1.0 / (1.0 + np.exp(-x))).clip(0.0, 1.0)


def _build_limitup_continuation_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Premium 涨停接力增强层 V1。

    目标：从“D 日分析 -> T 日竞价买入 -> T+1 卖出”的实盘路径出发，
    优先筛出 T 日具备涨停攻击性、且 T+1 仍有延续上涨能力的标的。

    说明：规则评分层保留为模型缺失/加载失败时的安全兜底。
    """
    out = df.copy()
    idx = out.index

    eret_plus = pd.to_numeric(out.get("eret_plus_value", pd.Series([0.0] * len(out), index=idx)), errors="coerce").fillna(0.0)
    t1_up = pd.to_numeric(out.get("t1_up_rate", out.get("p_premium", pd.Series([0.5] * len(out), index=idx))), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    old_prob = pd.to_numeric(out.get("p_premium", pd.Series([0.5] * len(out), index=idx)), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    conf_score = pd.to_numeric(out.get("eret_plus_conf_score", pd.Series([0.5] * len(out), index=idx)), errors="coerce").fillna(0.5).clip(0.0, 1.0)

    f_strength = _num_series(out, "f_strength", "strength", "momentum_score", default=np.nan)
    f_theme = _num_series(out, "f_theme", "theme_score", default=np.nan)
    ret_5d = _num_series(out, "ret_5d", "return_5d", default=np.nan)
    pct_chg = _num_series(out, "pct_chg", "pct_change", "涨跌幅", default=np.nan)
    amount_z_5d = _num_series(out, "amount_z_5d", "amount_z", default=np.nan)
    close_pos = _num_series(out, "close_pos_n", "close_pos_10d", "close_pos_20d", default=np.nan)
    p_fill = _num_series(out, "p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill", default=np.nan).fillna(0.5).clip(0.0, 1.0)
    risk_pen = _num_series(out, "risk_penalty_total", "risk_penalty", "risk_score", default=np.nan).fillna(0.0)
    cost_total = _num_series(out, "cost_total", "cost", "trade_cost", default=np.nan).fillna(0.0)
    vol_10d = _num_series(out, "vol_10d", default=np.nan).fillna(0.0)
    range_1d = _num_series(out, "range_1d", "amplitude", default=np.nan).fillna(0.0)

    eret_rank = _rank_pct(eret_plus)
    strength_rank = _rank_pct(f_strength)
    theme_rank = _rank_pct(f_theme)
    ret_rank = _rank_pct(ret_5d)
    amount_rank = _rank_pct(amount_z_5d)
    close_pos_n = pd.to_numeric(close_pos, errors="coerce").fillna(_rank_pct(close_pos).median() if len(close_pos) else 0.5).clip(0.0, 1.0)

    attack_logit = (
        -1.10
        + 1.25 * _zscore(old_prob).fillna(0.0).clip(-3, 3)
        + 1.15 * _zscore(eret_plus).fillna(0.0).clip(-3, 3)
        + 0.85 * _zscore(f_strength).fillna(0.0).clip(-3, 3)
        + 0.45 * _zscore(ret_5d).fillna(0.0).clip(-3, 3)
        + 0.35 * _zscore(amount_z_5d).fillna(0.0).clip(-3, 3)
        + 0.25 * (close_pos_n - 0.50)
        - 0.65 * pd.to_numeric(vol_10d, errors="coerce").fillna(0.0).clip(0.0, 0.35)
        - 0.45 * pd.to_numeric(range_1d, errors="coerce").fillna(0.0).clip(0.0, 0.30)
        - 0.80 * pd.to_numeric(risk_pen, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    t_limitup_prob = _sigmoid_series(attack_logit).clip(0.01, 0.99)

    t_limitup_strength_ratio = (
        0.30 * t_limitup_prob
        + 0.22 * strength_rank
        + 0.14 * theme_rank
        + 0.14 * ret_rank
        + 0.10 * amount_rank
        + 0.10 * close_pos_n
    ).clip(0.0, 1.0)

    continuation_core = (
        0.52 * t1_up
        + 0.18 * eret_rank
        + 0.12 * strength_rank
        + 0.10 * theme_rank
        + 0.08 * conf_score
    )
    crowding_penalty = ((close_pos_n - 0.82).clip(lower=0.0) * 0.25 + pd.to_numeric(range_1d, errors="coerce").fillna(0.0).clip(0.0, 0.30) * 0.35)
    t1_continue_up_rate = (continuation_core - crowding_penalty).clip(0.01, 0.99)

    exec_safety = (
        0.55 * p_fill
        + 0.25 * conf_score
        + 0.20 * (1.0 - _rank_pct(cost_total, neutral=0.35))
        - 0.25 * pd.to_numeric(risk_pen, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)

    score_ratio = (
        0.40 * t_limitup_prob
        + 0.25 * t_limitup_strength_ratio
        + 0.25 * t1_continue_up_rate
        + 0.10 * exec_safety
    ).clip(0.0, 1.0)

    out["t_limitup_prob_rule"] = t_limitup_prob.round(6)
    out["t_limitup_strength_rule"] = (t_limitup_strength_ratio * 100.0).round(4)
    out["t1_continue_up_rate_rule"] = t1_continue_up_rate.round(6)
    out["limitup_continuation_score_rule"] = (score_ratio * 100.0).round(4)

    out["t_limitup_prob"] = out["t_limitup_prob_rule"]
    out["T日涨停概率"] = out["t_limitup_prob"]
    out["t_limitup_strength"] = out["t_limitup_strength_rule"]
    out["T日涨停强度"] = out["t_limitup_strength"]
    out["t1_continue_up_rate"] = out["t1_continue_up_rate_rule"]
    out["T+1延续上涨率"] = out["t1_continue_up_rate"]
    out["limitup_continuation_score"] = out["limitup_continuation_score_rule"]
    out["涨停接力评分"] = out["limitup_continuation_score"]
    return out


# ========= V3.6 涨停接力专业概率引擎 =========

def _limitup_model_candidate_paths(cfg: PremiumConfig) -> List[Path]:
    """兼容多个可能的模型落盘名，避免工作流命名差异导致模型接不上。"""
    root = cfg.out_root()
    return [
        root / "models" / "limitup_probability_engine.joblib",
        root / "models" / "limitup_model.joblib",
        root / "models" / "limitup_probability_model.joblib",
        root / "limitup_probability_engine.joblib",
    ]


def _find_limitup_model_path(cfg: PremiumConfig) -> Tuple[Optional[Path], str]:
    for p in _limitup_model_candidate_paths(cfg):
        if p.exists():
            return p, "ok"
    return None, "limitup_model_missing"


def _apply_limitup_probability_engine(cfg: PremiumConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    接入 limitup_probability_engine.py。

    安全原则：
    1）模型存在且可预测：模型概率参与最终排序字段；
    2）模型缺失或失败：保留 V3.5 规则评分，不阻断主流程；
    3）所有模型字段落盘，便于验收“是否真正用起来”。
    """
    out = df.copy()
    model_path, path_reason = _find_limitup_model_path(cfg)
    trace: Dict[str, object] = {
        "limitup_model_mode": "rule_fallback",
        "limitup_model_reason": path_reason,
        "limitup_model_path": str(model_path) if model_path is not None else "",
    }

    model_cols = [
        "t_limitup_prob_model",
        "t_touch_limitup_prob_model",
        "t1_up_prob_model",
        "t1_high_profit_prob_model",
        "t1_close_ret_pred",
        "t1_high_ret_pred",
        "limitup_model_score",
    ]
    for c in model_cols:
        if c not in out.columns:
            out[c] = pd.NA

    if _load_limitup_probability_bundle is None:
        out["limitup_model_src"] = "rule_fallback:import_failed"
        trace["limitup_model_reason"] = "limitup_engine_import_failed"
        return out, trace

    if model_path is None:
        out["limitup_model_src"] = "rule_fallback:model_missing"
        return out, trace

    try:
        bundle = _load_limitup_probability_bundle(model_path)
        pred = bundle.predict(out)
        for c in model_cols:
            if c in pred.columns:
                out[c] = pred[c]

        rule_t_limitup = pd.to_numeric(out.get("t_limitup_prob_rule", out.get("t_limitup_prob")), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        rule_t_strength = pd.to_numeric(out.get("t_limitup_strength_rule", out.get("t_limitup_strength")), errors="coerce").fillna(50.0).clip(0.0, 100.0)
        rule_t1_continue = pd.to_numeric(out.get("t1_continue_up_rate_rule", out.get("t1_continue_up_rate")), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        rule_score = pd.to_numeric(out.get("limitup_continuation_score_rule", out.get("limitup_continuation_score")), errors="coerce").fillna(50.0).clip(0.0, 100.0)

        m_t_limitup = pd.to_numeric(out["t_limitup_prob_model"], errors="coerce").fillna(rule_t_limitup).clip(0.0, 1.0)
        m_touch = pd.to_numeric(out["t_touch_limitup_prob_model"], errors="coerce").fillna(m_t_limitup).clip(0.0, 1.0)
        m_t1_up = pd.to_numeric(out["t1_up_prob_model"], errors="coerce").fillna(rule_t1_continue).clip(0.0, 1.0)
        m_high_profit = pd.to_numeric(out["t1_high_profit_prob_model"], errors="coerce").fillna(m_t1_up).clip(0.0, 1.0)
        m_score = pd.to_numeric(out["limitup_model_score"], errors="coerce").fillna(
            0.35 * m_t_limitup + 0.25 * m_touch + 0.25 * m_t1_up + 0.15 * m_high_profit
        ).clip(0.0, 1.0)

        # 最终生产字段：模型为主、规则为辅，避免模型早期样本不足时突然漂移。
        out["t_limitup_prob"] = (0.70 * m_t_limitup + 0.30 * rule_t_limitup).clip(0.01, 0.99).round(6)
        out["T日涨停概率"] = out["t_limitup_prob"]
        out["t_limitup_strength"] = (0.60 * (100.0 * m_touch) + 0.40 * rule_t_strength).clip(0.0, 100.0).round(4)
        out["T日涨停强度"] = out["t_limitup_strength"]
        out["t1_continue_up_rate"] = (0.70 * m_t1_up + 0.30 * rule_t1_continue).clip(0.01, 0.99).round(6)
        out["T+1延续上涨率"] = out["t1_continue_up_rate"]
        out["limitup_continuation_score"] = (0.70 * (100.0 * m_score) + 0.30 * rule_score).clip(0.0, 100.0).round(4)
        out["涨停接力评分"] = out["limitup_continuation_score"]
        out["limitup_model_src"] = "limitup_probability_engine:model_v1"

        trace.update({
            "limitup_model_mode": "model_v1_blend_rule",
            "limitup_model_reason": "ok",
            "limitup_model_feature_n": len(getattr(bundle, "feature_cols", []) or []),
            "limitup_model_train_end_date": getattr(bundle, "train_end_date", ""),
            "limitup_model_valid_start_date": getattr(bundle, "valid_start_date", ""),
        })
        return out, trace
    except Exception as e:
        out["limitup_model_src"] = f"rule_fallback:predict_error:{type(e).__name__}"
        trace["limitup_model_reason"] = f"limitup_model_predict_error:{type(e).__name__}"
        return out, trace


def _build_execution_fields(df: pd.DataFrame) -> pd.DataFrame:
    """基于 D 日收盘、D→T+1收益/价格区间/上涨率，生成 T 日买入与 T+1 卖出执行字段。"""
    out = df.copy()

    close_t = pd.to_numeric(out.get("close_T", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    eret_plus = pd.to_numeric(out.get("eret_plus_value", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    p_up = pd.to_numeric(out.get("t1_continue_up_rate", out.get("t1_up_rate", out.get("p_premium", pd.Series([0.5] * len(out), index=out.index)))), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    conf_score = pd.to_numeric(out.get("eret_plus_conf_score", pd.Series([0.5] * len(out), index=out.index)), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    limitup_prob = pd.to_numeric(out.get("t_limitup_prob", pd.Series([0.5] * len(out), index=out.index)), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    continuation_score = pd.to_numeric(out.get("limitup_continuation_score", pd.Series([50.0] * len(out), index=out.index)), errors="coerce").fillna(50.0).clip(0.0, 100.0)
    p25 = pd.to_numeric(out.get("close_T2_p25", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    p50 = pd.to_numeric(out.get("close_T2_p50", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    p75 = pd.to_numeric(out.get("close_T2_p75", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")

    edge = eret_plus.fillna(0.0)
    max_open_premium = (0.18 * edge + 0.035 * (p_up - 0.5) + 0.025 * (conf_score - 0.5)).clip(lower=-0.03, upper=0.08)
    max_buy_px = close_t * (1.0 + max_open_premium)

    methods = []
    max_buy_labels = []
    sell_plans = []

    for idx in out.index:
        ct = close_t.loc[idx]
        ep = edge.loc[idx]
        pu = p_up.loc[idx]
        cf = conf_score.loc[idx]
        lp = limitup_prob.loc[idx]
        cs = continuation_score.loc[idx]
        mp = max_open_premium.loc[idx]
        mb = max_buy_px.loc[idx]
        q25 = p25.loc[idx]
        q50 = p50.loc[idx]
        q75 = p75.loc[idx]

        if pd.isna(ct) or ct <= 0:
            methods.append("只观察不追")
            max_buy_labels.append("")
            sell_plans.append("缺少T日收盘价，先不生成价格计划")
            continue

        if ep <= -0.005 or pu < 0.48 or lp < 0.35 or cs < 42:
            methods.append("放弃")
            max_buy_labels.append("不建议买入")
        elif ep < 0.015 or pu < 0.56 or lp < 0.48 or cs < 55 or cf < 0.50:
            methods.append("只观察不追")
            max_buy_labels.append(f"≤{_fmt_price_value(min(mb, ct * 1.01))}")
        elif ep < 0.035 or pu < 0.64 or lp < 0.60 or cs < 68 or cf < 0.62:
            methods.append("限价竞价")
            max_buy_labels.append(f"≤{_fmt_price_value(mb)}")
        else:
            methods.append("市价竞价")
            if mp >= 0.05:
                max_buy_labels.append(f"≤{_fmt_price_value(mb)}；高开>5%谨慎追")
            else:
                max_buy_labels.append(f"≤{_fmt_price_value(mb)}")

        q25_s = _fmt_price_value(q25)
        q50_s = _fmt_price_value(q50)
        q75_s = _fmt_price_value(q75)
        stop_px = _fmt_price_value(min(q25 if pd.notna(q25) and q25 > 0 else ct * 0.97, ct * 0.985))

        if ep <= -0.005 or pu < 0.48:
            sell_plans.append("未建议买入；若误买，弱于T日收盘价立即减仓")
        elif q50_s and q75_s:
            sell_plans.append(f"冲高至{q50_s}附近优先兑现；强势放量再看{q75_s}；弱于{stop_px}止损")
        elif q50_s:
            sell_plans.append(f"接近{q50_s}优先兑现；开盘不强则减仓；弱于{stop_px}止损")
        else:
            sell_plans.append(f"按盘中强弱分批卖；弱于{stop_px}止损")

    out["T日建议买入方式"] = pd.Series(methods, index=out.index, dtype="object")
    out["T日可接受买入价"] = pd.Series(max_buy_labels, index=out.index, dtype="object")
    out["T+1卖出计划"] = pd.Series(sell_plans, index=out.index, dtype="object")

    # Backward-compatible aliases; the canonical contract is D analysis -> T buy -> T+1 sell.
    out["T+1建议买入方式"] = out["T日建议买入方式"]
    out["T+1可接受买入价"] = out["T日可接受买入价"]
    out["t_buy_method"] = out["T日建议买入方式"]
    out["t_max_buy_price"] = out["T日可接受买入价"]
    out["t1_buy_method"] = out["T日建议买入方式"]
    out["t1_max_buy_price"] = out["T日可接受买入价"]
    out["t1_sell_plan"] = out["T+1卖出计划"]
    out["T+2卖出计划"] = out["T+1卖出计划"]
    out["t2_sell_plan"] = out["T+1卖出计划"]
    return out


# ========= 主入口 =========

def predict_latest(cfg: Optional[PremiumConfig] = None) -> PredictResult:
    cfg = cfg or PremiumConfig.load()
    _ensure_dirs(cfg)

    pred_path = cfg.pred_source_latest_path()
    if not pred_path.exists():
        _write_last_run(cfg, "unknown", {"ok": False, "reason": f"pred_source_latest_not_found: {pred_path}"})
        return PredictResult(False, "unknown", None, True, f"未找到主输入：{pred_path}")

    df0 = _read_csv_smart(pred_path)
    if df0 is None or df0.empty:
        _write_last_run(cfg, "unknown", {"ok": False, "reason": "pred_source_latest_empty"})
        return PredictResult(False, "unknown", None, True, "pred_source_latest 为空")

    df0 = _normalize_pred_source(df0)
    trade_date = _infer_trade_date(df0)
    if trade_date == "unknown":
        _write_last_run(cfg, "unknown", {"ok": False, "reason": "cannot_infer_trade_date"})
        return PredictResult(False, "unknown", None, True, "无法从 pred_source_latest 推断 trade_date")

    if "trade_date" not in df0.columns:
        df0["trade_date"] = trade_date
    else:
        df0["trade_date"] = df0["trade_date"].astype(str).map(_to_yyyymmdd)

    if "ts_code" not in df0.columns:
        _write_last_run(cfg, trade_date, {"ok": False, "reason": "missing_ts_code"})
        return PredictResult(False, trade_date, None, True, "pred_source_latest 缺少 ts_code（或别名 code）")

    df0["ts_code"] = df0["ts_code"].astype(str).str.strip()
    if "name" not in df0.columns:
        df0["name"] = pd.NA

    pack_status = detect_factor_packs(cfg, trade_date)
    r_t = ensure_daily_cached(cfg, trade_date)
    if not r_t.ok:
        pending_truth_T = True
        pending_truth_reason_T = f"truth_T_not_ready: {r_t.reason}"
        d0 = pd.DataFrame(columns=["ts_code", "close_T"])
    else:
        pending_truth_T = False
        pending_truth_reason_T = "ok"
        d0 = load_daily(cfg, trade_date)[["ts_code", "close"]].rename(columns={"close": "close_T"})

    buy_date, buy_reason = _advance_trade_days(cfg, trade_date, 1)
    target_date, td_reason = _advance_trade_days(cfg, trade_date, int(cfg.horizon_trade_days))
    if not buy_date or not target_date:
        reason = f"strict_a_share_calendar_unavailable: buy_date={buy_reason}; target_date={td_reason}"
        _write_last_run(
            cfg,
            trade_date,
            {
                "ok": False,
                "reason": reason,
                "buy_date": buy_date,
                "target_date": target_date,
                "pending": True,
                "calendar_contract": "strict_a_share_trade_calendar_only",
            },
        )
        return PredictResult(False, trade_date, target_date or None, True, reason)
    pending = (td_reason != "trade_cal_ok") or (buy_reason != "trade_cal_ok") or pending_truth_T
    pending_reason = f"buy_date:{buy_reason};target_date:{td_reason}"

    dec = _load_decision_merge(cfg, trade_date)
    df = df0.copy()
    if not dec.empty:
        m = df.merge(dec, on=["trade_date", "ts_code"], how="left", suffixes=("", "_dec"))
        if "name_dec" in m.columns:
            m["name"] = m["name"].where(m["name"].notna() & (m["name"].astype(str).str.strip() != ""), m["name_dec"])
            m = m.drop(columns=["name_dec"])
        df = m
    else:
        for c in ("dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason"):
            df[c] = pd.NA

    p, e, s, conf, dq, risk = _pick_pred_fields(df)
    df["p_premium"] = p
    df["e_premium"] = e
    df["score_ev"] = s
    df["confidence"] = conf
    df["data_quality"] = dq
    df["risk_flags"] = risk

    df = df.merge(d0, on="ts_code", how="left")
    df["base_date"] = trade_date
    df["buy_date"] = buy_date
    df["target_date"] = target_date

    feats = build_features_by_packs(cfg, trade_date, df0, pack_status.packs_used)
    if feats is not None and not feats.empty:
        df = df.merge(feats, on="ts_code", how="left")

    df, ehx_trace = _build_ehx_v1(cfg, df)
    df = _compute_quantile_returns(cfg, df)
    df = _build_limitup_continuation_fields(df)
    df, limitup_trace = _apply_limitup_probability_engine(cfg, df)

    qs = tuple(getattr(cfg, "quantiles", (0.05, 0.25, 0.50, 0.75, 0.95)))
    q_mid = min(qs, key=lambda x: abs(float(x) - 0.50))
    mid_key = f"r_p{int(round(float(q_mid) * 100)):02d}"

    df["rank_eret_plus"] = pd.NA
    df["rank_r_p50"] = pd.NA
    df["rank_limitup_continuation"] = pd.NA

    if "eret_plus_value" in df.columns and pd.to_numeric(df["eret_plus_value"], errors="coerce").notna().any():
        df["rank_eret_plus"] = pd.to_numeric(df["eret_plus_value"], errors="coerce").rank(method="first", ascending=False).astype("Int64")
    if mid_key in df.columns and pd.to_numeric(df[mid_key], errors="coerce").notna().any():
        df["rank_r_p50"] = pd.to_numeric(df[mid_key], errors="coerce").rank(method="first", ascending=False).astype("Int64")
    if "limitup_continuation_score" in df.columns and pd.to_numeric(df["limitup_continuation_score"], errors="coerce").notna().any():
        df["rank_limitup_continuation"] = pd.to_numeric(df["limitup_continuation_score"], errors="coerce").rank(method="first", ascending=False).astype("Int64")
        df = df.sort_values(
            by=["limitup_continuation_score", "t_limitup_prob", "t1_continue_up_rate", "t_limitup_strength"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    elif "eret_plus_value" in df.columns and pd.to_numeric(df["eret_plus_value"], errors="coerce").notna().any():
        df = df.sort_values(by=["eret_plus_value"], ascending=False, na_position="last").reset_index(drop=True)
    elif mid_key in df.columns and pd.to_numeric(df[mid_key], errors="coerce").notna().any():
        df = df.sort_values(by=[mid_key], ascending=False, na_position="last").reset_index(drop=True)
    elif "p_premium" in df.columns and pd.to_numeric(df["p_premium"], errors="coerce").notna().any():
        df = df.sort_values(by=["p_premium"], ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = _rebuild_rank_front(df)
    df = _build_execution_fields(df)
    df = add_rank_groups(df)

    topn = int(cfg.top_n)
    df_top = df.head(topn).copy()
    df_full = df.copy()

    v_cols = [f"r_p{int(round(float(q) * 100)):02d}" for q in qs]
    v_cols += [f"close_T2_p{int(round(float(q) * 100)):02d}" for q in qs]

    exec_cols = [
        "T日建议买入方式",
        "T日可接受买入价",
        "T+1建议买入方式",
        "T+1可接受买入价",
        "T+1卖出计划",
        "T+2卖出计划",
        "t_buy_method",
        "t_max_buy_price",
        "t1_buy_method",
        "t1_max_buy_price",
        "t1_sell_plan",
        "t2_sell_plan",
    ]

    limitup_model_cols = [
        "t_limitup_prob_rule", "t_limitup_strength_rule", "t1_continue_up_rate_rule", "limitup_continuation_score_rule",
        "t_limitup_prob_model", "t_touch_limitup_prob_model", "t1_up_prob_model", "t1_high_profit_prob_model",
        "t1_close_ret_pred", "t1_high_ret_pred", "limitup_model_score", "limitup_model_src",
    ]

    out_cols = [
        "rank", "trade_date", "base_date", "buy_date", "target_date", "ts_code", "name", "close_T",
        "rank_group", "is_top10", "is_top20", "榜单分组",
        *exec_cols,
        "t_limitup_prob", "T日涨停概率", "t_limitup_strength", "T日涨停强度",
        "t1_continue_up_rate", "T+1延续上涨率", "limitup_continuation_score", "涨停接力评分",
        *limitup_model_cols,
        "eret_pred_raw", "eret_plus_value", "eret_plus_delta", "eret_plus_direction", "eret_plus_conf", "eret_plus_conf_score", "eret_plus_src",
        *v_cols,
        "t1_up_rate", "T+1上涨率",
        "rank_limitup_continuation", "rank_eret_plus", "rank_r_p50", "p_premium", "e_premium", "score_ev", "risk_flags", "confidence", "data_quality",
        "dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason",
    ]

    for c in out_cols:
        if c not in df_top.columns:
            df_top[c] = pd.NA
        if c not in df_full.columns:
            df_full[c] = pd.NA

    out_top = df_top[out_cols].copy()
    out_top10 = df.head(10).copy()
    out_top20 = df.head(20).copy()
    for c in out_cols:
        if c not in out_top10.columns:
            out_top10[c] = pd.NA
        if c not in out_top20.columns:
            out_top20[c] = pd.NA
    out_top10 = out_top10[out_cols].copy()
    out_top20 = out_top20[out_cols].copy()
    out_full = df_full[out_cols].copy()

    p_top10 = _write_csv(cfg.out_top10_csv(trade_date), out_top10)
    p_top20 = _write_csv(cfg.out_top20_csv(trade_date), out_top20)
    p_top = _write_csv(cfg.out_top30_csv(trade_date), out_top)
    p_full = _write_csv(cfg.out_full_csv(trade_date), out_full)

    verify_pending = True
    verify_reason = "pending"
    verify_cols = [
        "rank", "trade_date", "base_date", "buy_date", "target_date", "ts_code", "name", "close_T",
        "rank_group", "is_top10", "is_top20", "榜单分组",
        "T日建议买入方式", "T日可接受买入价",
        "T+1建议买入方式", "T+1可接受买入价", "T+1卖出计划", "T+2卖出计划",
        "t_buy_method", "t_max_buy_price",
        "t1_buy_method", "t1_max_buy_price", "t1_sell_plan", "t2_sell_plan",
        "t_limitup_prob", "T日涨停概率", "t_limitup_strength", "T日涨停强度",
        "t1_continue_up_rate", "T+1延续上涨率", "limitup_continuation_score", "涨停接力评分",
        *limitup_model_cols,
        "t1_up_rate", "T+1上涨率",
        "close_T2_actual", "r_actual", mid_key,
        "eret_pred_raw", "eret_plus_value", "eret_plus_delta", "eret_plus_direction", "eret_plus_conf",
        "in_p10", "in_p50", "err_r_p50", "err_close_p50", "actual_ret", "raw_abs_err", "plus_abs_err", "improve_flag", "hit_up",
        "open_T_actual", "high_T_actual", "close_T_actual", "t_limit_price_est",
        "t_limitup_actual", "t_touch_limitup_actual", "t_limitup_verify_ready",
        "t_limitup_verify_reason", "t_limitup_verify_trade_date", "d_analysis_trade_date",
    ]

    df_verify = out_top[[c for c in verify_cols if c in out_top.columns]].copy()
    for c in verify_cols:
        if c not in df_verify.columns:
            df_verify[c] = pd.NA
    df_verify = df_verify[verify_cols].copy()

    r_t2 = ensure_daily_cached(cfg, target_date)
    if (not pending_truth_T) and r_t2.ok:
        d2 = load_daily(cfg, target_date)[["ts_code", "close"]].rename(columns={"close": "close_T2_actual"})
        tmp = out_top.merge(d2, on="ts_code", how="left")
        tmp["close_T2_actual"] = pd.to_numeric(tmp["close_T2_actual"], errors="coerce")
        tmp["close_T"] = pd.to_numeric(tmp["close_T"], errors="coerce")
        tmp["r_actual"] = np.log(tmp["close_T2_actual"] / tmp["close_T"])

        lo10 = f"r_p{int(round(float(min(qs)) * 100)):02d}"
        hi10 = f"r_p{int(round(float(max(qs)) * 100)):02d}"
        q25 = min(qs, key=lambda x: abs(float(x) - 0.25))
        q75 = min(qs, key=lambda x: abs(float(x) - 0.75))
        lo50 = f"r_p{int(round(float(q25) * 100)):02d}"
        hi50 = f"r_p{int(round(float(q75) * 100)):02d}"

        tmp["in_p10"] = (pd.to_numeric(tmp["r_actual"], errors="coerce") >= pd.to_numeric(tmp.get(lo10), errors="coerce")) & (pd.to_numeric(tmp["r_actual"], errors="coerce") <= pd.to_numeric(tmp.get(hi10), errors="coerce"))
        tmp["in_p50"] = (pd.to_numeric(tmp["r_actual"], errors="coerce") >= pd.to_numeric(tmp.get(lo50), errors="coerce")) & (pd.to_numeric(tmp["r_actual"], errors="coerce") <= pd.to_numeric(tmp.get(hi50), errors="coerce"))
        tmp["err_r_p50"] = pd.to_numeric(tmp["r_actual"], errors="coerce") - pd.to_numeric(tmp.get(mid_key), errors="coerce")
        mid_price_key = f"close_T2_p{int(round(float(q_mid) * 100)):02d}"
        tmp["err_close_p50"] = pd.to_numeric(tmp["close_T2_actual"], errors="coerce") - pd.to_numeric(tmp.get(mid_price_key), errors="coerce")
        tmp["actual_ret"] = tmp["close_T2_actual"] / tmp["close_T"] - 1
        tmp["raw_abs_err"] = (pd.to_numeric(tmp["actual_ret"], errors="coerce") - pd.to_numeric(tmp.get("eret_pred_raw"), errors="coerce")).abs()
        tmp["plus_abs_err"] = (pd.to_numeric(tmp["actual_ret"], errors="coerce") - pd.to_numeric(tmp.get("eret_plus_value"), errors="coerce")).abs()
        tmp["improve_flag"] = np.where(pd.to_numeric(tmp["plus_abs_err"], errors="coerce") < pd.to_numeric(tmp["raw_abs_err"], errors="coerce"), 1, 0)
        tmp["hit_up"] = tmp["actual_ret"].apply(lambda x: "是" if pd.notna(x) and float(x) > 0 else ("否" if pd.notna(x) else ""))

        keep = [c for c in verify_cols if c in tmp.columns]
        df_verify = tmp[keep].copy()
        verify_pending = False
        verify_reason = "ok"
    else:
        verify_pending = True
        verify_reason = f"truth_not_ready: T_ok={not pending_truth_T} T2_ok={r_t2.ok} ({pending_reason})"

    r_buy = ensure_daily_cached(cfg, buy_date)
    daily_buy = load_daily(cfg, buy_date) if r_buy.ok else None
    df_verify, limitup_validation = attach_limitup_validation(
        df_verify=df_verify,
        daily_t=daily_buy,
        trade_date=trade_date,
        buy_date=buy_date,
    )
    if r_buy.ok:
        limitup_truth_reason = "ok"
    else:
        limitup_truth_reason = f"t_truth_not_ready:{r_buy.reason}"

    for c in verify_cols:
        if c not in df_verify.columns:
            df_verify[c] = pd.NA
    df_verify = df_verify[verify_cols].copy()

    p_verify = _write_csv(cfg.out_verify_csv(trade_date), df_verify)

    gen_ts = _utc_now_iso()
    md = render_premium_report_md(
        trade_date=trade_date,
        target_date=target_date,
        df_top30=out_top,
        df_verify=df_verify,
        verify_pending=verify_pending,
        verify_reason=verify_reason,
        gen_ts=gen_ts,
    )
    md += make_audit_block_md(
        packs_used=pack_status.packs_used,
        packs_missing=pack_status.packs_missing,
        degrade_mode=pack_status.degrade_mode,
        missing_fields=pack_status.missing_fields,
        notes=pack_status.notes,
    )

    p_md = _write_text(cfg.report_md_path(trade_date), md)
    _write_text(cfg.report_latest_md_path(), md)

    historical_limitup_stats = _collect_historical_limitup_stats(cfg)
    html_report = render_premium_report_html(
        trade_date=trade_date,
        buy_date=buy_date,
        target_date=target_date,
        df_top=out_top,
        df_verify=df_verify,
        verify_pending=verify_pending,
        verify_reason=verify_reason,
        gen_ts=gen_ts,
        model_version=str(getattr(cfg, "model_version", "-")),
        audit_notes=[
            f"factor_degrade_mode={pack_status.degrade_mode}",
            f"limitup_truth={limitup_truth_reason}",
        ],
        report_dates=_list_report_dates(cfg, trade_date),
        historical_limitup_stats=historical_limitup_stats,
    )
    p_html = _write_text(cfg.report_html_path(trade_date), html_report)
    _write_text(cfg.report_latest_html_path(), html_report)

    audit_kv = make_audit_kv(
        extra_prefix="factor",
        packs_used=pack_status.packs_used,
        packs_missing=pack_status.packs_missing,
        degrade_mode=pack_status.degrade_mode,
        missing_fields=pack_status.missing_fields,
        notes=pack_status.notes,
    )

    if len(df) > 0 and "eret_plus_src" in df.columns and df["eret_plus_src"].notna().any():
        eret_plus_src = str(df["eret_plus_src"].dropna().iloc[0])
    else:
        eret_plus_src = "ehx_unknown"

    if len(df) > 0 and "limitup_model_src" in df.columns and df["limitup_model_src"].notna().any():
        limitup_model_src = str(df["limitup_model_src"].dropna().iloc[0])
    else:
        limitup_model_src = "rule_fallback:unknown"

    _write_last_run(
        cfg,
        trade_date,
        {
            "ok": True,
            "buy_date": buy_date,
            "target_date": target_date,
            "pending": bool(pending or verify_pending),
            "pending_reason": pending_reason,
            "truth_T_ok": (not pending_truth_T),
            "truth_T_reason": pending_truth_reason_T,
            "verify_pending": bool(verify_pending),
            "verify_reason": verify_reason,
            "eret_plus_src": eret_plus_src,
            **ehx_trace,
            "limitup_model_src": limitup_model_src,
            **limitup_trace,
            "exec_fields": "buy_date|T日涨停概率|T日涨停强度|T+1延续上涨率|涨停接力评分|T日建议买入方式|T日可接受买入价|T+1卖出计划",
            "limitup_model_fields": "t_limitup_prob_model|t_touch_limitup_prob_model|t1_up_prob_model|t1_high_profit_prob_model|t1_close_ret_pred|t1_high_ret_pred|limitup_model_score",
            "rank_groups": "TOP10|TOP20",
            "limitup_truth_reason": limitup_truth_reason,
            **limitup_validation.as_dict(),
            "history_limitup_source": historical_limitup_stats.get("source", ""),
            "history_limitup_reason": historical_limitup_stats.get("reason", ""),
            "history_limitup_days": historical_limitup_stats.get("n_days", 0),
            "history_top10_limitup_total": historical_limitup_stats.get("top10_total", 0),
            "history_top10_limitup_hits": historical_limitup_stats.get("top10_hits", 0),
            "history_top10_limitup_hit_rate": (
                "" if not historical_limitup_stats.get("ready") else round(float(historical_limitup_stats.get("top10_hit_rate", float("nan"))), 6)
            ),
            "history_top20_limitup_total": historical_limitup_stats.get("top20_total", 0),
            "history_top20_limitup_hits": historical_limitup_stats.get("top20_hits", 0),
            "history_top20_limitup_hit_rate": (
                "" if not historical_limitup_stats.get("ready") else round(float(historical_limitup_stats.get("top20_hit_rate", float("nan"))), 6)
            ),
            "out_top10": str(p_top10),
            "out_top20": str(p_top20),
            "out_top30": str(p_top),
            "out_full": str(p_full),
            "out_verify": str(p_verify),
            "report_md": str(p_md),
            "report_html": str(p_html),
            **audit_kv,
        },
    )

    return PredictResult(
        ok=True,
        trade_date=trade_date,
        target_date=target_date,
        pending=bool(pending or verify_pending),
        reason="pending" if (pending or verify_pending) else "ok",
        out_top10=str(p_top10),
        out_top20=str(p_top20),
        out_top30=str(p_top),
        out_full=str(p_full),
        out_verify=str(p_verify),
        report_md=str(p_md),
        report_html=str(p_html),
    )


__all__ = ["PredictResult", "predict_latest"]
