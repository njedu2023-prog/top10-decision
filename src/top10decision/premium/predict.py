#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Predict（V3：E_ret_plus 主线，端到端落盘）

当前锁死契约（Premium Contract V3）：
- 主目标不再是“好看报告”，而是围绕 E_ret 生成更强的 E_ret_plus
- 当前第一刀只动 premium 线，不动 decision 主线文件
- 当前先落 EHX 冷启动版骨架，不假装已经完成二级残差学习终版
- 旧 V2 分位链保留，但其中心值优先改为围绕 eret_plus_value 展开
- 必须新增 raw / plus 后验误差对比，支撑闭环验收

主输入：
- data/pred/pred_source_latest.csv（a-top10 全量源表，候选=全量，不过滤）

decision 产物：
- 仅用于字段合并/标签（不得过滤），输出字段需带 dec_ 前缀

时间口径：
- horizon = 2 个交易日（T -> T+2）
- 非交易日顺延（必须用交易日历推进；不得用未来行情探测）

行情真值：
- data/market/daily_YYYYMMDD.csv（由 Market Truth Layer 拉取并缓存）

输出：
- outputs/premium/premium_top30_{T}.csv
- outputs/premium/premium_full_{T}.csv
- outputs/premium/premium_verify_{T}.csv
- docs/reports/premium_{T}.md
- docs/reports/premium_latest.md
- outputs/premium/_last_run.txt（每次覆盖）

当前版本说明：
- EHX-V1 先采用冷启动增强器（非训练终版）
- 产出：
  - eret_pred_raw
  - eret_plus_value
  - eret_plus_delta
  - eret_plus_direction
  - eret_plus_conf
  - eret_plus_conf_score
  - eret_plus_src
- 验证新增：
  - raw_abs_err
  - plus_abs_err
  - improve_flag
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import PremiumConfig
from .market_truth import ensure_daily_cached, load_daily
from .report_md import render_premium_report_md

from .factor_registry import detect_factor_packs
from .factor_builders import build_features_by_packs
from .audit import make_audit_block_md, make_audit_kv


_TD_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class PredictResult:
    ok: bool
    trade_date: str
    target_date: Optional[str]
    pending: bool
    reason: str
    out_top30: Optional[str] = None
    out_full: Optional[str] = None
    out_verify: Optional[str] = None
    report_md: Optional[str] = None


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


def _rebuild_rank_front(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank" in df.columns:
        df = df.drop(columns=["rank"], errors="ignore")
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def _norm_ppf(q: float) -> float:
    """
    标准正态分位点近似（避免引入 scipy 依赖）
    """
    import math

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if q <= 0.0 or q >= 1.0:
        return float("nan")

    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        x = (
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    elif q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        x = -(
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    else:
        r = q - 0.5
        s = r * r
        x = (
            (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r
            / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
        )
    return x


def _first_existing_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def _num_series(df: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    c = _first_existing_col(df, *names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce")


def _str_series(df: pd.DataFrame, *names: str, default: str = "") -> pd.Series:
    c = _first_existing_col(df, *names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[c].astype(str)


# ========= 交易日推进（✅ 必须用交易日历，不得用未来行情探测）=========

def _get_tushare_token() -> str:
    return (os.getenv("TUSHARE_TOKEN", "") or "").strip()


def _tushare_trade_cal_open_days(token: str, start_date: str, end_date: str) -> Optional[list]:
    """
    返回 [YYYYMMDD, ...] 交易日列表（is_open=1）。
    不引入 tushare 包，直接请求 api.tushare.pro。
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    payload = {
        "api_name": "trade_cal",
        "token": token,
        "params": {
            "exchange": "SSE",
            "start_date": start_date,
            "end_date": end_date,
            "is_open": "1",
        },
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
        days = sorted([d for d in df["cal_date"].tolist() if _TD_RE.match(str(d) or "")])
        return days
    except Exception:
        return None


def _advance_trade_days_by_trade_cal(trade_date: str, steps: int) -> Tuple[Optional[str], str]:
    """
    用 Tushare trade_cal 推进 steps 个交易日。
    返回：(target_date, reason)；reason='ok' 表示成功。
    """
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
    """
    兜底：按工作日（周一~周五）推进 steps 天。
    ⚠️ 不是严格交易日，但保证不会退回 T，避免验证“同日假命中”。
    """
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
    """
    ✅ 正确口径：
    - 先用 trade_cal 推进（真实交易日）
    - 失败则兜底工作日推进（保证 target_date 不会等于 T）
    返回：(target_date, reason)
    """
    td, reason = _advance_trade_days_by_trade_cal(trade_date, steps)
    if td:
        return td, "trade_cal_ok"
    td2, reason2 = _advance_trade_days_fallback_business_day(trade_date, steps)
    return td2, f"{reason}|{reason2}"


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
            u = sorted(s.unique().tolist())
            return u[-1]
    for c in df.columns:
        s = df[c].dropna().astype(str).map(_to_yyyymmdd)
        s = s[s.str.match(r"^\d{8}$", na=False)]
        if not s.empty:
            u = sorted(s.unique().tolist())
            return u[-1]
    return "unknown"


def _pick_pred_fields(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_prob = col("p_premium", "up_prob", "probability", "prob", "p")
    c_ret = col(
        "e_premium",
        "pred_ret",
        "pred_return",
        "ret",
        "premium_ret",
        "pred_premium_ret",
        "pred_ret_mean",
        "eret_pred",
        "e_ret_pred",
    )
    c_score = col("score_ev", "ev", "final_score", "score", "pred_ev")
    c_conf = col("confidence", "conf")
    c_dq = col("data_quality", "dq")
    c_risk = col("risk_flags", "risk", "warning", "risk_hint", "fill_risk_hint")

    p = pd.to_numeric(df[c_prob], errors="coerce") if c_prob else pd.Series([np.nan] * len(df))
    p = p.fillna(0.5).clip(0.0, 1.0)

    e = pd.to_numeric(df[c_ret], errors="coerce") if c_ret else pd.Series([np.nan] * len(df))
    e = e.fillna(0.0)

    if c_score:
        s = pd.to_numeric(df[c_score], errors="coerce").fillna(p * e)
    else:
        s = (p * e).astype(float)

    conf = pd.to_numeric(df[c_conf], errors="coerce") if c_conf else pd.Series([pd.NA] * len(df))
    dq = pd.to_numeric(df[c_dq], errors="coerce") if c_dq else pd.Series([pd.NA] * len(df))
    risk = df[c_risk].astype(str) if c_risk else pd.Series([""] * len(df))

    return p, e, s, conf, dq, risk


# ========= decision merge（仅标签）=========

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

    cols = {str(c).strip().lower(): c for c in dec.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    out = pd.DataFrame(
        {
            "trade_date": dec["trade_date"].astype(str),
            "ts_code": dec["ts_code"].astype(str),
        }
    )
    if "name" in dec.columns:
        out["name"] = dec["name"]

    m_rank = pick("dec_rank", "decision_rank", "rank", "决策排名")
    m_w = pick("dec_weight", "weight", "target_weight", "决策权重")
    m_can = pick("dec_can_buy", "can_buy", "可买提示")
    m_pf = pick("dec_p_fill", "p_fill", "p_fill_pred", "p_fill_pred_final")
    m_reason = pick("dec_reason", "reason", "label", "决策原因", "决策标签")

    out["dec_rank"] = dec[m_rank] if m_rank else pd.NA
    out["dec_weight"] = dec[m_w] if m_w else pd.NA
    out["dec_can_buy"] = dec[m_can] if m_can else pd.NA
    out["dec_p_fill"] = dec[m_pf] if m_pf else pd.NA
    out["dec_reason"] = dec[m_reason] if m_reason else pd.NA

    out = out.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)
    return out


# ========= EHX（优先加载训练模型，失败再回退冷启动）=========

def _ehx_model_path(cfg: PremiumConfig) -> Path:
    return cfg.out_root() / "models" / "ehx_delta.joblib"


def _load_ehx_bundle(cfg: PremiumConfig) -> Optional[dict]:
    path = _ehx_model_path(cfg)
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            return None
        model = obj.get("model")
        feature_cols = obj.get("feature_cols")
        if model is None or not isinstance(feature_cols, (list, tuple)) or not feature_cols:
            return None
        return {"model": model, "feature_cols": list(feature_cols), "path": str(path)}
    except Exception:
        return None


def _build_ehx_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["eret_pred_raw"] = pd.to_numeric(df.get("e_premium", pd.Series([np.nan] * len(df))), errors="coerce").fillna(0.0)
    out["p_fill_pred"] = _num_series(df, "p_fill_pred", "p_fill_pred_final", "p_fill", "dec_p_fill", default=np.nan).fillna(0.5)
    out["cost_total"] = _num_series(df, "cost_total", "cost", "cost_value", "cost_all", "trade_cost", default=np.nan).fillna(0.0)
    out["risk_penalty_total"] = _num_series(df, "risk_penalty_total", "risk_penalty", "riskpenalty", "risk_penalty_score", "risk_score", default=np.nan).fillna(0.0)
    out["ev"] = _num_series(df, "score_ev", "ev", "pred_ev", default=np.nan).fillna(0.0)
    out["turnover_rate"] = _num_series(df, "turnover_rate", default=np.nan).fillna(0.0)
    out["amount"] = _num_series(df, "amount", default=np.nan).fillna(0.0)
    out["vol"] = _num_series(df, "vol", "volume", default=np.nan).fillna(0.0)
    out["close"] = _num_series(df, "close", "close_T", default=np.nan).fillna(0.0)
    out["pct_chg"] = _num_series(df, "pct_chg", "pct_change", default=np.nan).fillna(0.0)
    out["amplitude"] = _num_series(df, "amplitude", "range_1d", default=np.nan).fillna(0.0)

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


def _infer_conf_from_inputs(out: pd.DataFrame, eret_raw: pd.Series) -> tuple[pd.Series, pd.Series]:
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

    conf_label = conf_score.apply(lambda x: to_conf_label(float(x)) if pd.notna(x) else "low")
    return conf_label, conf_score.round(6)


def _build_ehx_v1(cfg: PremiumConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    EHX-V1（优先加载训练模型，失败再回退冷启动）
    ------------------------------------------------
    优先逻辑：
    - 若 outputs/premium/models/ehx_delta.joblib 存在且可读，则优先加载真 EHX 残差模型
    - 若模型不存在 / 读取失败 / 预测失败，则自动回退到冷启动增强器
    - 这样不会阻塞主线，同时保持训练链与推理链可衔接

    输出：
    - eret_pred_raw
    - eret_plus_value
    - eret_plus_delta
    - eret_plus_direction
    - eret_plus_conf
    - eret_plus_conf_score
    - eret_plus_src
    """
    out = df.copy()

    eret_raw = pd.to_numeric(out.get("e_premium", pd.Series([np.nan] * len(out))), errors="coerce").fillna(0.0)

    # 1) 优先尝试真实 EHX 残差模型
    bundle = _load_ehx_bundle(cfg)
    if bundle is not None:
        try:
            X_ehx = _build_ehx_feature_frame(out, bundle["feature_cols"])
            delta_hat = pd.Series(bundle["model"].predict(X_ehx), index=out.index, dtype="float64")
            delta_hat = pd.to_numeric(delta_hat, errors="coerce").fillna(0.0).clip(lower=-0.12, upper=0.12)
            eret_plus = (eret_raw + delta_hat).clip(lower=-0.95, upper=2.0)
            conf_label, conf_score = _infer_conf_from_inputs(out, eret_raw)
            eps = 0.002
            direction = pd.Series(
                np.where(delta_hat > eps, "up", np.where(delta_hat < -eps, "down", "flat")),
                index=out.index,
                dtype="object",
            )

            out["eret_pred_raw"] = eret_raw
            out["eret_plus_value"] = eret_plus
            out["eret_plus_delta"] = delta_hat
            out["eret_plus_direction"] = direction
            out["eret_plus_conf"] = conf_label
            out["eret_plus_conf_score"] = conf_score
            out["eret_plus_src"] = "ehx:model_v1"
            return out
        except Exception:
            pass

    # 2) 回退：冷启动增强器
    p_prob = pd.to_numeric(out.get("p_premium", pd.Series([0.5] * len(out))), errors="coerce").fillna(0.5).clip(0.0, 1.0)

    p_fill = _num_series(
        out,
        "p_fill_pred",
        "p_fill_pred_final",
        "p_fill",
        "dec_p_fill",
        default=np.nan,
    ).fillna(0.5).clip(0.0, 1.0)

    ev = _num_series(out, "score_ev", "ev", "pred_ev", default=np.nan).fillna(0.0)
    cost = _num_series(
        out,
        "cost_total",
        "cost",
        "cost_value",
        "cost_all",
        "trade_cost",
        default=np.nan,
    ).fillna(0.0)

    risk_pen = _num_series(
        out,
        "risk_penalty_total",
        "risk_penalty",
        "riskpenalty",
        "risk_penalty_score",
        "risk_score",
        default=np.nan,
    ).fillna(0.0)

    ret_5d = _num_series(out, "ret_5d", default=np.nan).fillna(0.0)
    vol_10d = _num_series(out, "vol_10d", default=np.nan).fillna(0.0)
    range_1d = _num_series(out, "range_1d", default=np.nan).fillna(0.0)
    amount_z_5d = _num_series(out, "amount_z_5d", default=np.nan).fillna(1.0)
    f_strength = _num_series(out, "f_strength", default=np.nan).fillna(0.0)
    f_theme = _num_series(out, "f_theme", default=np.nan).fillna(0.0)
    close_pos_n = _num_series(out, "close_pos_n", "close_pos_10d", "close_pos_20d", default=np.nan).fillna(0.5)

    z_ev = _zscore(ev)
    z_cost = _zscore(cost)
    z_risk = _zscore(risk_pen)
    z_mom = _zscore(ret_5d)
    z_vol = _zscore(vol_10d)
    z_range = _zscore(range_1d)
    z_strength = _zscore(f_strength)
    z_theme = _zscore(f_theme)
    z_prob = _zscore(p_prob)

    liquidity_edge = (p_fill - 0.5) * 2.0
    amount_dev = (amount_z_5d - 1.0).abs().clip(0.0, 3.0)
    crowded_penalty = (close_pos_n - 0.70).clip(lower=0.0)

    # 冷启动 delta：
    # 正向：EV / 概率 / 动量 / 强度 / 题材 / 可兑现性
    # 负向：成本 / 风险 / 波动 / 日内振幅 / 价格拥挤 / 量能异常
    delta = (
        0.0100 * z_ev
        + 0.0055 * z_prob
        + 0.0075 * z_mom
        + 0.0040 * z_strength
        + 0.0025 * z_theme
        + 0.0070 * liquidity_edge
        - 0.0100 * z_cost
        - 0.0120 * z_risk
        - 0.0060 * z_vol
        - 0.0040 * z_range
        - 0.0040 * crowded_penalty
        - 0.0020 * amount_dev
    )

    # 对原始收益的轻微放大/抑制，避免 delta 过度脱锚
    delta += eret_raw.clip(lower=-0.20, upper=0.20) * 0.08

    delta = pd.to_numeric(delta, errors="coerce").fillna(0.0).clip(lower=-0.08, upper=0.08)
    eret_plus = (eret_raw + delta).clip(lower=-0.95, upper=2.0)

    # 可信度：先做基础分数，后续再升级成真正模型化 conf
    input_cols = pd.DataFrame(
        {
            "eret_raw": eret_raw,
            "p_fill": p_fill,
            "ev": ev,
            "cost": cost,
            "risk_pen": risk_pen,
            "ret_5d": ret_5d,
            "vol_10d": vol_10d,
            "range_1d": range_1d,
        }
    )
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

    eps = 0.002
    direction = pd.Series(
        np.where(delta > eps, "up", np.where(delta < -eps, "down", "flat")),
        index=out.index,
        dtype="object",
    )
    conf_label = conf_score.apply(lambda x: to_conf_label(float(x)) if pd.notna(x) else "low")

    out["eret_pred_raw"] = eret_raw
    out["eret_plus_value"] = eret_plus
    out["eret_plus_delta"] = delta
    out["eret_plus_direction"] = direction
    out["eret_plus_conf"] = conf_label
    out["eret_plus_conf_score"] = conf_score.round(6)
    out["eret_plus_src"] = "ehx:coldstart_v1"

    return out


# ========= V2/V3 分布预测 =========

def _build_mu_sigma(cfg: PremiumConfig, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    V3 口径：
    - 分位中心优先围绕 eret_plus_value 构造
    - 若没有 eret_plus_value，再回退到 e_premium
    - sigma 仍优先用 Pack0 的波动/振幅/量能构造
    """
    base_sigma = float(getattr(cfg, "base_sigma", 0.05))
    score_scale = float(getattr(cfg, "score_scale", 0.012))

    e_plus = pd.to_numeric(df.get("eret_plus_value", pd.Series([np.nan] * len(df))), errors="coerce")
    e_raw = pd.to_numeric(df.get("e_premium", pd.Series([np.nan] * len(df))), errors="coerce")

    e_core = e_plus.where(e_plus.notna(), e_raw)
    mu_from_e = np.log1p(e_core.clip(lower=-0.99))

    z_prob = _zscore(df.get("f_prob", df.get("p_premium", pd.Series([np.nan] * len(df)))))
    z_strength = _zscore(df.get("f_strength", pd.Series([np.nan] * len(df))))
    z_theme = _zscore(df.get("f_theme", pd.Series([np.nan] * len(df))))
    z_mom = _zscore(df.get("ret_5d", pd.Series([np.nan] * len(df))))

    mu_pack = score_scale * (0.55 * z_prob + 0.30 * z_strength + 0.15 * z_theme + 0.20 * z_mom)

    mu = mu_from_e.copy()
    need = ~np.isfinite(mu.values)
    mu = mu.where(~need, mu_pack)
    mu = pd.to_numeric(mu, errors="coerce").fillna(0.0)

    vol10 = pd.to_numeric(df.get("vol_10d", pd.Series([np.nan] * len(df))), errors="coerce")
    range1 = pd.to_numeric(df.get("range_1d", pd.Series([np.nan] * len(df))), errors="coerce")
    az5 = pd.to_numeric(df.get("amount_z_5d", pd.Series([np.nan] * len(df))), errors="coerce")
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
    for q in qs:
        z = _norm_ppf(float(q))
        out[f"r_p{int(round(q * 100)):02d}"] = mu + sigma * z

    close_T = pd.to_numeric(out.get("close_T", pd.Series([np.nan] * len(out))), errors="coerce")
    for q in qs:
        key = f"r_p{int(round(q * 100)):02d}"
        out[f"close_T2_p{int(round(q * 100)):02d}"] = close_T * np.exp(pd.to_numeric(out[key], errors="coerce"))
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

    # === Factor Packs：自动启用/降级 ===
    pack_status = detect_factor_packs(cfg, trade_date)

    # 先确保 T 日真值缓存落盘（T 日应当可拿到；拿不到也不阻塞）
    r_t = ensure_daily_cached(cfg, trade_date)
    if not r_t.ok:
        pending_truth_T = True
        pending_truth_reason_T = f"truth_T_not_ready: {r_t.reason}"
        d0 = pd.DataFrame(columns=["ts_code", "close_T"])
    else:
        pending_truth_T = False
        pending_truth_reason_T = "ok"
        d0 = load_daily(cfg, trade_date)[["ts_code", "close"]].rename(columns={"close": "close_T"})

    # 正确推进到 T+2：只依赖交易日历，不依赖未来行情
    target_date, td_reason = _advance_trade_days(cfg, trade_date, int(cfg.horizon_trade_days))

    pending = (td_reason != "trade_cal_ok") or pending_truth_T
    pending_reason = td_reason

    # decision merge（仅标签，不过滤）
    dec = _load_decision_merge(cfg, trade_date)
    df = df0.copy()
    if not dec.empty:
        m = df.merge(dec, on=["trade_date", "ts_code"], how="left", suffixes=("", "_dec"))
        if "name_dec" in m.columns:
            m["name"] = m["name"].where(
                m["name"].notna() & (m["name"].astype(str).str.strip() != ""),
                m["name_dec"],
            )
            m = m.drop(columns=["name_dec"])
        df = m
    else:
        for c in ("dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason"):
            df[c] = pd.NA

    # 旧口径字段保留（就地取材 + 兜底）
    p, e, s, conf, dq, risk = _pick_pred_fields(df)
    df["p_premium"] = p
    df["e_premium"] = e
    df["score_ev"] = s
    df["confidence"] = conf
    df["data_quality"] = dq
    df["risk_flags"] = risk

    # merge close_T（用于价格还原）
    df = df.merge(d0, on="ts_code", how="left")
    df["target_date"] = target_date

    # 计算并合并 Factor Packs 特征（至少 Pack0）
    feats = build_features_by_packs(cfg, trade_date, df0, pack_status.packs_used)
    if feats is not None and not feats.empty:
        df = df.merge(feats, on="ts_code", how="left")

    # === 新主线：EHX 冷启动骨架 ===
    df = _build_ehx_v1(cfg, df)

    # V3 分布预测字段（中心优先围绕 eret_plus_value）
    df = _compute_quantile_returns(cfg, df)

    # 排序：优先 eret_plus_value；再退到 r_p50；再退到 p_premium；再退到源顺序
    qs = tuple(getattr(cfg, "quantiles", (0.05, 0.25, 0.50, 0.75, 0.95)))
    q_mid = min(qs, key=lambda x: abs(float(x) - 0.50))
    mid_key = f"r_p{int(round(float(q_mid) * 100)):02d}"

    df["rank_eret_plus"] = pd.NA
    df["rank_r_p50"] = pd.NA

    if "eret_plus_value" in df.columns and pd.to_numeric(df["eret_plus_value"], errors="coerce").notna().any():
        df = df.sort_values(by=["eret_plus_value"], ascending=False, na_position="last").reset_index(drop=True)
        df["rank_eret_plus"] = np.arange(1, len(df) + 1)
    elif mid_key in df.columns and pd.to_numeric(df[mid_key], errors="coerce").notna().any():
        df = df.sort_values(by=[mid_key], ascending=False, na_position="last").reset_index(drop=True)
        df["rank_r_p50"] = np.arange(1, len(df) + 1)
    elif "p_premium" in df.columns and pd.to_numeric(df["p_premium"], errors="coerce").notna().any():
        df = df.sort_values(by=["p_premium"], ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if pd.isna(df["rank_r_p50"]).all() and mid_key in df.columns and pd.to_numeric(df[mid_key], errors="coerce").notna().any():
        rk = (
            pd.to_numeric(df[mid_key], errors="coerce")
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
        df["rank_r_p50"] = rk

    df = _rebuild_rank_front(df)

    # top30 + full
    topn = int(cfg.top_n)
    df_top = df.head(topn).copy()
    df_full = df.copy()

    v_cols = []
    for q in qs:
        v_cols.append(f"r_p{int(round(float(q) * 100)):02d}")
    for q in qs:
        v_cols.append(f"close_T2_p{int(round(float(q) * 100)):02d}")

    out_cols = [
        "rank",
        "trade_date",
        "target_date",
        "ts_code",
        "name",
        "close_T",
        "eret_pred_raw",
        "eret_plus_value",
        "eret_plus_delta",
        "eret_plus_direction",
        "eret_plus_conf",
        "eret_plus_conf_score",
        "eret_plus_src",
        *v_cols,
        "rank_eret_plus",
        "rank_r_p50",
        "p_premium",
        "e_premium",
        "score_ev",
        "risk_flags",
        "confidence",
        "data_quality",
        "dec_rank",
        "dec_weight",
        "dec_can_buy",
        "dec_p_fill",
        "dec_reason",
    ]

    for c in out_cols:
        if c not in df_top.columns:
            df_top[c] = pd.NA
        if c not in df_full.columns:
            df_full[c] = pd.NA

    out_top = df_top[out_cols].copy()
    out_full = df_full[out_cols].copy()

    p_top = _write_csv(cfg.out_top30_csv(trade_date), out_top)
    p_full = _write_csv(cfg.out_full_csv(trade_date), out_full)

    # verify：只有当 T+2 真值真的可拿到时才做；否则保持 PENDING
    verify_pending = True
    verify_reason = "pending"

    verify_cols = [
        "rank",
        "trade_date",
        "target_date",
        "ts_code",
        "name",
        "close_T",
        "close_T2_actual",
        "r_actual",
        mid_key,
        "eret_pred_raw",
        "eret_plus_value",
        "eret_plus_delta",
        "eret_plus_direction",
        "eret_plus_conf",
        "in_p10",
        "in_p50",
        "err_r_p50",
        "err_close_p50",
        "actual_ret",
        "raw_abs_err",
        "plus_abs_err",
        "improve_flag",
        "hit_up",
    ]
    df_verify = out_top[["rank", "trade_date", "target_date", "ts_code", "name"]].copy()
    for c in verify_cols:
        if c not in df_verify.columns:
            df_verify[c] = pd.NA

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

        tmp["in_p10"] = (
            (pd.to_numeric(tmp["r_actual"], errors="coerce") >= pd.to_numeric(tmp.get(lo10), errors="coerce"))
            & (pd.to_numeric(tmp["r_actual"], errors="coerce") <= pd.to_numeric(tmp.get(hi10), errors="coerce"))
        )
        tmp["in_p50"] = (
            (pd.to_numeric(tmp["r_actual"], errors="coerce") >= pd.to_numeric(tmp.get(lo50), errors="coerce"))
            & (pd.to_numeric(tmp["r_actual"], errors="coerce") <= pd.to_numeric(tmp.get(hi50), errors="coerce"))
        )

        tmp["err_r_p50"] = (
            pd.to_numeric(tmp["r_actual"], errors="coerce")
            - pd.to_numeric(tmp.get(mid_key), errors="coerce")
        )
        mid_price_key = f"close_T2_p{int(round(float(q_mid) * 100)):02d}"
        tmp["err_close_p50"] = (
            pd.to_numeric(tmp["close_T2_actual"], errors="coerce")
            - pd.to_numeric(tmp.get(mid_price_key), errors="coerce")
        )

        tmp["actual_ret"] = tmp["close_T2_actual"] / tmp["close_T"] - 1
        tmp["raw_abs_err"] = (
            pd.to_numeric(tmp["actual_ret"], errors="coerce")
            - pd.to_numeric(tmp.get("eret_pred_raw"), errors="coerce")
        ).abs()
        tmp["plus_abs_err"] = (
            pd.to_numeric(tmp["actual_ret"], errors="coerce")
            - pd.to_numeric(tmp.get("eret_plus_value"), errors="coerce")
        ).abs()
        tmp["improve_flag"] = np.where(
            pd.to_numeric(tmp["plus_abs_err"], errors="coerce")
            < pd.to_numeric(tmp["raw_abs_err"], errors="coerce"),
            1,
            0,
        )
        tmp["hit_up"] = tmp["actual_ret"].apply(
            lambda x: "是" if pd.notna(x) and float(x) > 0 else ("否" if pd.notna(x) else "")
        )

        keep = [c for c in verify_cols if c in tmp.columns]
        df_verify = tmp[keep].copy()
        verify_pending = False
        verify_reason = "ok"
    else:
        verify_pending = True
        verify_reason = f"truth_not_ready: T_ok={not pending_truth_T} T2_ok={r_t2.ok} ({pending_reason})"

    p_verify = _write_csv(cfg.out_verify_csv(trade_date), df_verify)

    # 渲染报告 md（保留现有 report_md 接口，避免本轮直接扩散）
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

    audit_kv = make_audit_kv(
        extra_prefix="factor",
        packs_used=pack_status.packs_used,
        packs_missing=pack_status.packs_missing,
        degrade_mode=pack_status.degrade_mode,
        missing_fields=pack_status.missing_fields,
        notes=pack_status.notes,
    )

    _write_last_run(
        cfg,
        trade_date,
        {
            "ok": True,
            "target_date": target_date,
            "pending": bool(pending or verify_pending),
            "pending_reason": pending_reason,
            "truth_T_ok": (not pending_truth_T),
            "truth_T_reason": pending_truth_reason_T,
            "verify_pending": bool(verify_pending),
            "verify_reason": verify_reason,
            "eret_plus_src": str(df.get("eret_plus_src", pd.Series(["ehx:unknown"])) .iloc[0]) if "eret_plus_src" in df.columns and len(df) > 0 else "ehx:unknown",
            "out_top30": str(p_top),
            "out_full": str(p_full),
            "out_verify": str(p_verify),
            "report_md": str(p_md),
            **audit_kv,
        },
    )

    return PredictResult(
        ok=True,
        trade_date=trade_date,
        target_date=target_date,
        pending=bool(pending or verify_pending),
        reason="pending" if (pending or verify_pending) else "ok",
        out_top30=str(p_top),
        out_full=str(p_full),
        out_verify=str(p_verify),
        report_md=str(p_md),
    )


__all__ = ["PredictResult", "predict_latest"]
