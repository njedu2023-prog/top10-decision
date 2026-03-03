#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Predict（V2：Close[T+2] 分布预测，端到端落盘）

锁死契约（来自 Premium.md / ANCHOR.md）：
- 主输入：data/pred/pred_source_latest.csv（a-top10 全量源表，候选=全量，不过滤）
- decision 产物：仅用于字段合并/标签（不得过滤），输出字段需带 dec_ 前缀
- horizon = 2 个交易日（T -> T+2），非交易日顺延（✅ 必须用交易日历推进；不得用未来行情探测）
- 行情真值：data/market/daily_YYYYMMDD.csv（由 Market Truth Layer 拉取并缓存）
- 行情未到：pending（不得报错卡死）
- 验证表顺序必须与预测表 Top30 完全一致（不可重新排序）
- 输出：
  outputs/premium/premium_top30_{T}.csv
  outputs/premium/premium_full_{T}.csv
  outputs/premium/premium_verify_{T}.csv
  docs/reports/premium_{T}.md
  docs/reports/premium_latest.md
  outputs/premium/_last_run.txt（每次覆盖）

V2 核心：
- 预测 r = ln(Close[T+2]/Close[T]) 的分位数：r_p05/r_p25/r_p50/r_p75/r_p95
- 还原价格分位数：close_T2_pXX = close_T * exp(r_pXX)
- 默认排序：r_p50 降序；若缺则退化到 p_premium；再缺则保持源顺序（不得随机）

新增（核心开发阶段）：
- Factor Packs 机制一次性写死：Pack0/1/2 自动启用/自动降级 + 审计落盘
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

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    plow = 0.02425
    phigh = 1 - plow

    if q <= 0.0 or q >= 1.0:
        return float("nan")

    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    elif q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        x = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    else:
        r = q - 0.5
        s = r * r
        x = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
            (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
    return x


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
    end = (d0 + dt.timedelta(days=120)).strftime("%Y%m%d")  # 给足长假窗口

    days = _tushare_trade_cal_open_days(token, td, end)
    if not days:
        return None, "trade_cal_empty"

    # trade_date 如果不是交易日：取其后的第一个交易日作为基准，再推进 steps-1
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


def _pick_pred_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_prob = col("p_premium", "up_prob", "probability", "prob", "p")
    c_ret = col("e_premium", "pred_ret", "pred_return", "ret", "premium_ret", "pred_premium_ret", "pred_ret_mean")
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

    out = pd.DataFrame({
        "trade_date": dec["trade_date"].astype(str),
        "ts_code": dec["ts_code"].astype(str),
    })
    if "name" in dec.columns:
        out["name"] = dec["name"]

    m_rank = pick("dec_rank", "decision_rank", "rank", "决策排名")
    m_w = pick("dec_weight", "weight", "target_weight", "决策权重")
    m_can = pick("dec_can_buy", "can_buy", "可买提示")
    m_pf = pick("dec_p_fill", "p_fill")
    m_reason = pick("dec_reason", "reason", "label", "决策原因", "决策标签")

    out["dec_rank"] = dec[m_rank] if m_rank else pd.NA
    out["dec_weight"] = dec[m_w] if m_w else pd.NA
    out["dec_can_buy"] = dec[m_can] if m_can else pd.NA
    out["dec_p_fill"] = dec[m_pf] if m_pf else pd.NA
    out["dec_reason"] = dec[m_reason] if m_reason else pd.NA

    out = out.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)
    return out


# ========= V2 分布预测（冷启动版本：无模型也可跑）=========

def _build_mu_sigma(cfg: PremiumConfig, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    新口径（Factor Packs 驱动）：
    - mu（中心）优先用 Pack0 的先验 + 动量构造
    - sigma（宽度）优先用 Pack0 的波动/振幅/量能构造
    - 仍保留兜底：如果特征缺失，则退回旧逻辑（保证永远可跑）
    """
    base_sigma = float(getattr(cfg, "base_sigma", 0.05))
    score_scale = float(getattr(cfg, "score_scale", 0.012))

    # 旧口径 e_premium 作为辅助
    e = pd.to_numeric(df.get("e_premium", pd.Series([np.nan] * len(df))), errors="coerce")
    mu_from_e = np.log1p(e.clip(lower=-0.99))

    # Pack0 先验
    z_prob = _zscore(df.get("f_prob", df.get("p_premium", pd.Series([np.nan] * len(df)))))
    z_strength = _zscore(df.get("f_strength", pd.Series([np.nan] * len(df))))
    z_theme = _zscore(df.get("f_theme", pd.Series([np.nan] * len(df))))
    z_mom = _zscore(df.get("ret_5d", pd.Series([np.nan] * len(df))))

    # 组合中心：主要由先验 + 动量决定
    mu_pack = score_scale * (0.55 * z_prob + 0.30 * z_strength + 0.15 * z_theme + 0.20 * z_mom)

    mu = mu_from_e.copy()
    need = ~np.isfinite(mu.values)
    mu = mu.where(~need, mu_pack)

    # 如果 mu 仍有大量缺失，最终兜底为 0
    mu = pd.to_numeric(mu, errors="coerce").fillna(0.0)

    # 宽度：波动 + 振幅 + 量能异常
    vol10 = pd.to_numeric(df.get("vol_10d", pd.Series([np.nan] * len(df))), errors="coerce")
    range1 = pd.to_numeric(df.get("range_1d", pd.Series([np.nan] * len(df))), errors="coerce")
    az5 = pd.to_numeric(df.get("amount_z_5d", pd.Series([np.nan] * len(df))), errors="coerce")

    # 量能异常：偏离 1 越大越不确定
    az_dev = (az5 - 1.0).abs()

    # sigma 组合，全部缺失则回到 base_sigma
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

    # === Factor Packs：自动启用/降级（一次性写死机制）===
    pack_status = detect_factor_packs(cfg, trade_date)

    # 2) 先确保 T 日真值缓存落盘（T 日应当可拿到；拿不到也不阻塞）
    r_t = ensure_daily_cached(cfg, trade_date)
    if not r_t.ok:
        pending_truth_T = True
        pending_truth_reason_T = f"truth_T_not_ready: {r_t.reason}"
        d0 = pd.DataFrame(columns=["ts_code", "close_T"])
    else:
        pending_truth_T = False
        pending_truth_reason_T = "ok"
        d0 = load_daily(cfg, trade_date)[["ts_code", "close"]].rename(columns={"close": "close_T"})

    # 3) ✅ 正确推进到 T+2：只依赖交易日历（trade_cal），不依赖未来行情
    target_date, td_reason = _advance_trade_days(cfg, trade_date, int(cfg.horizon_trade_days))

    # pending：只要 trade_cal 不是真正 ok，就标记 pending（但 target_date 仍非空）
    pending = (td_reason != "trade_cal_ok") or pending_truth_T
    pending_reason = td_reason

    # 4) decision merge（仅标签，不过滤）
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

    # 5) 旧口径字段保留（就地取材 + 兜底）
    p, e, s, conf, dq, risk = _pick_pred_fields(df)
    df["p_premium"] = p
    df["e_premium"] = e
    df["score_ev"] = s
    df["confidence"] = conf
    df["data_quality"] = dq
    df["risk_flags"] = risk

    # 6) merge close_T（用于价格分位数还原）
    df = df.merge(d0, on="ts_code", how="left")
    df["target_date"] = target_date

    # 6.1) 计算并合并 Factor Packs 特征（至少 Pack0）
    feats = build_features_by_packs(cfg, trade_date, df0, pack_status.packs_used)
    if feats is not None and not feats.empty:
        df = df.merge(feats, on="ts_code", how="left")

    # 7) V2 分布预测字段（r_pXX / close_T2_pXX）
    df = _compute_quantile_returns(cfg, df)

    # 8) 排序（锁死）：r_p50 降序；缺则 p_premium；再缺保持源顺序
    qs = tuple(getattr(cfg, "quantiles", (0.05, 0.25, 0.50, 0.75, 0.95)))
    q_mid = min(qs, key=lambda x: abs(float(x) - 0.50))
    mid_key = f"r_p{int(round(float(q_mid) * 100)):02d}"
    df["rank_r_p50"] = pd.NA

    if mid_key in df.columns and pd.to_numeric(df[mid_key], errors="coerce").notna().any():
        df = df.sort_values(by=[mid_key], ascending=False, na_position="last").reset_index(drop=True)
        df["rank_r_p50"] = np.arange(1, len(df) + 1)
    elif "p_premium" in df.columns and pd.to_numeric(df["p_premium"], errors="coerce").notna().any():
        df = df.sort_values(by=["p_premium"], ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = _rebuild_rank_front(df)

    # 9) top30 + full
    topn = int(cfg.top_n)
    df_top = df.head(topn).copy()
    df_full = df.copy()

    v2_cols = []
    for q in qs:
        v2_cols.append(f"r_p{int(round(float(q) * 100)):02d}")
    for q in qs:
        v2_cols.append(f"close_T2_p{int(round(float(q) * 100)):02d}")

    out_cols = [
        "rank", "trade_date", "target_date", "ts_code", "name",
        "close_T",
        *v2_cols,
        "rank_r_p50",
        "p_premium", "e_premium", "score_ev", "risk_flags", "confidence", "data_quality",
        "dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason",
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

    # 10) verify（只有当 T+2 真值真的可拿到时才做；否则保持 PENDING）
    verify_pending = True
    verify_reason = "pending"

    verify_cols = [
        "rank", "trade_date", "target_date", "ts_code", "name",
        "close_T", "close_T2_actual",
        "r_actual",
        mid_key,
        "in_p10", "in_p50",
        "err_r_p50", "err_close_p50",
        "e_premium", "actual_ret", "hit_up",
    ]
    df_verify = out_top[["rank", "trade_date", "target_date", "ts_code", "name"]].copy()
    for c in verify_cols:
        if c not in df_verify.columns:
            df_verify[c] = pd.NA

    # 仅当 target_date 的真值日数据存在时才验证；否则一直 pending（避免“同日假命中”）
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

        tmp["err_r_p50"] = pd.to_numeric(tmp["r_actual"], errors="coerce") - pd.to_numeric(tmp.get(mid_key), errors="coerce")
        mid_price_key = f"close_T2_p{int(round(float(q_mid) * 100)):02d}"
        tmp["err_close_p50"] = pd.to_numeric(tmp["close_T2_actual"], errors="coerce") - pd.to_numeric(tmp.get(mid_price_key), errors="coerce")

        tmp["actual_ret"] = tmp["close_T2_actual"] / tmp["close_T"] - 1
        tmp["hit_up"] = tmp["actual_ret"].apply(lambda x: "是" if pd.notna(x) and float(x) > 0 else ("否" if pd.notna(x) else ""))

        keep = [c for c in verify_cols if c in tmp.columns]
        df_verify = tmp[keep].copy()
        verify_pending = False
        verify_reason = "ok"
    else:
        verify_pending = True
        verify_reason = f"truth_not_ready: T_ok={not pending_truth_T} T2_ok={r_t2.ok} ({pending_reason})"

    p_verify = _write_csv(cfg.out_verify_csv(trade_date), df_verify)

    # 11) 渲染报告 md（附加 Factor Packs 审计块）
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

    # 12) last_run（把审计写进去）
    audit_kv = make_audit_kv(
        extra_prefix="factor",
        packs_used=pack_status.packs_used,
        packs_missing=pack_status.packs_missing,
        degrade_mode=pack_status.degrade_mode,
        missing_fields=pack_status.missing_fields,
        notes=pack_status.notes,
    )

    _write_last_run(cfg, trade_date, {
        "ok": True,
        "target_date": target_date,
        "pending": bool(pending or verify_pending),
        "pending_reason": pending_reason,
        "truth_T_ok": (not pending_truth_T),
        "truth_T_reason": pending_truth_reason_T,
        "verify_pending": bool(verify_pending),
        "verify_reason": verify_reason,
        "out_top30": str(p_top),
        "out_full": str(p_full),
        "out_verify": str(p_verify),
        "report_md": str(p_md),
        **audit_kv,
    })

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
