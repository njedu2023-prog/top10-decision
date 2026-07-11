#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild Premium HTML report archive.

Purpose:
  Historical HTML files are static. When the report UI template improves,
  old files will not inherit the new navigation/tabs/history panels unless
  they are regenerated. This script rebuilds every premium_YYYYMMDD.html
  from saved Premium CSV artifacts using the current renderer.

Inputs:
  outputs/premium/premium_full_YYYYMMDD.csv
  outputs/premium/premium_top30_YYYYMMDD.csv
  outputs/premium/premium_verify_YYYYMMDD.csv
  outputs/premium/learning/limitup_probability_training_samples.csv

Outputs:
  docs/reports/premium_YYYYMMDD.html
  docs/reports/premium_latest.html
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_symbol(module_name: str, file_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


PremiumConfig = _load_symbol(
    "premium_rebuild_config",
    SRC / "top10decision" / "premium" / "config.py",
    "PremiumConfig",
)
render_premium_report_html = _load_symbol(
    "premium_rebuild_views",
    SRC / "top10decision" / "premium" / "premium_views.py",
    "render_premium_report_html",
)
attach_limitup_validation = _load_symbol(
    "premium_rebuild_views_attach",
    SRC / "top10decision" / "premium" / "premium_views.py",
    "attach_limitup_validation",
)

from top10decision.premium.market_truth import ensure_daily_cached, load_daily  # noqa: E402

DATE_RE = re.compile(r"(20\d{6})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _date_from_path(path: Path) -> Optional[str]:
    m = DATE_RE.search(path.name)
    return m.group(1) if m else None


def _first_existing_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return str(hit)
    return None


def _to_yyyymmdd(x: object) -> str:
    s = str(x or "").strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "")
    m = DATE_RE.search(s)
    return m.group(1) if m else s[:8]


def _norm_ts_code(x: object) -> str:
    s = str(x or "").strip()
    if not s or s.lower() == "nan":
        return ""
    if "." in s:
        left, right = s.split(".", 1)
        digits = "".join(ch for ch in left if ch.isdigit()).zfill(6)
        return f"{digits}.{right.upper()}"
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


def _first_value(df: pd.DataFrame, names: Sequence[str], default: str = "") -> str:
    col = _first_existing_col(df, names)
    if not col:
        return default
    for value in df[col].dropna().astype(str).tolist():
        s = value.strip()
        if s and s.lower() not in {"nan", "none", "<na>", "nat"}:
            return s
    return default


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_csv_smart(path)
    except Exception:
        return pd.DataFrame()


def _artifact_dates(cfg: PremiumConfig) -> List[str]:
    dates = set()
    for pattern in ("premium_full_*.csv", "premium_top30_*.csv", "premium_verify_*.csv"):
        for path in cfg.out_root().glob(pattern):
            d = _date_from_path(path)
            if d:
                dates.add(d)
    return sorted(dates)


def _load_training_labels(cfg: PremiumConfig) -> pd.DataFrame:
    path = cfg.out_learning_dir() / "limitup_probability_training_samples.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_optional_csv(path)
    if df.empty:
        return df

    date_col = _first_existing_col(df, ["d_trade_date", "trade_date", "base_date", "d_analysis_trade_date"])
    code_col = _first_existing_col(df, ["ts_code", "code", "symbol"])
    if not date_col or not code_col:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["_hist_date"] = df[date_col].map(_to_yyyymmdd)
    out["_hist_code"] = df[code_col].map(_norm_ts_code)
    out["t_limitup_actual_hist"] = pd.to_numeric(
        df[_first_existing_col(df, ["t_limitup_hit", "t_limitup_actual"])],
        errors="coerce",
    ) if _first_existing_col(df, ["t_limitup_hit", "t_limitup_actual"]) else np.nan
    out["t_touch_limitup_actual_hist"] = pd.to_numeric(
        df[_first_existing_col(df, ["t_touch_limitup", "t_touch_limitup_actual"])],
        errors="coerce",
    ) if _first_existing_col(df, ["t_touch_limitup", "t_touch_limitup_actual"]) else np.nan
    out["t_limitup_verify_ready_hist"] = pd.to_numeric(
        df[_first_existing_col(df, ["label_matured", "t_limitup_verify_ready"])],
        errors="coerce",
    ) if _first_existing_col(df, ["label_matured", "t_limitup_verify_ready"]) else 1
    out["t1_close_ret_hist"] = pd.to_numeric(df.get("t1_close_ret"), errors="coerce") if "t1_close_ret" in df.columns else np.nan
    out["t1_high_ret_hist"] = pd.to_numeric(df.get("t1_high_ret"), errors="coerce") if "t1_high_ret" in df.columns else np.nan
    out = out.dropna(subset=["_hist_date", "_hist_code"])
    return out.drop_duplicates(subset=["_hist_date", "_hist_code"], keep="last").reset_index(drop=True)


def _patch_verify_from_training_labels(df_verify: pd.DataFrame, labels: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if df_verify.empty or labels.empty:
        return df_verify
    code_col = _first_existing_col(df_verify, ["ts_code", "code", "symbol"])
    if not code_col:
        return df_verify

    out = df_verify.copy()
    out["_hist_date"] = str(trade_date)
    out["_hist_code"] = out[code_col].map(_norm_ts_code)
    labels_one_day = labels[labels["_hist_date"].astype(str) == str(trade_date)].copy()
    if labels_one_day.empty:
        return out.drop(columns=["_hist_date", "_hist_code"], errors="ignore")

    out = out.merge(labels_one_day, on=["_hist_date", "_hist_code"], how="left")
    fill_map = {
        "t_limitup_actual": "t_limitup_actual_hist",
        "t_touch_limitup_actual": "t_touch_limitup_actual_hist",
        "t_limitup_verify_ready": "t_limitup_verify_ready_hist",
        "t1_close_ret": "t1_close_ret_hist",
        "t1_high_ret": "t1_high_ret_hist",
    }
    for dst, src in fill_map.items():
        if src not in out.columns:
            continue
        if dst not in out.columns:
            out[dst] = out[src]
        else:
            dst_num = pd.to_numeric(out[dst], errors="coerce")
            out[dst] = dst_num.where(dst_num.notna(), out[src])
    if "t_limitup_verify_reason" not in out.columns:
        out["t_limitup_verify_reason"] = ""
    ready = pd.to_numeric(out.get("t_limitup_verify_ready"), errors="coerce").fillna(0).eq(1)
    reason = out["t_limitup_verify_reason"].astype(str)
    out["t_limitup_verify_reason"] = np.where(ready & reason.isin(["", "nan", "None", "<NA>"]), "ok_from_training_archive", reason)
    return out.drop(columns=[c for c in out.columns if c.endswith("_hist")] + ["_hist_date", "_hist_code"], errors="ignore")


def _verify_ready_rows(df_verify: pd.DataFrame) -> int:
    if df_verify is None or df_verify.empty:
        return 0
    ready = pd.to_numeric(df_verify.get("t_limitup_verify_ready", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return int(ready.eq(1).sum())


def _refresh_verify_with_t_truth(
    cfg: PremiumConfig,
    df_verify: pd.DataFrame,
    trade_date: str,
    buy_date: str,
) -> tuple[pd.DataFrame, str]:
    if df_verify is None or df_verify.empty:
        return df_verify, "verify_empty"
    if _verify_ready_rows(df_verify) > 0:
        return df_verify, "already_ready"
    buy_date = _to_yyyymmdd(buy_date)
    if not re.fullmatch(r"20\d{6}", str(buy_date)):
        return df_verify, "missing_buy_date"

    r_buy = ensure_daily_cached(cfg, buy_date)
    if not r_buy.ok:
        return df_verify, f"t_daily_not_ready:{r_buy.reason}"
    try:
        daily_buy = load_daily(cfg, buy_date)
        refreshed, stats = attach_limitup_validation(
            df_verify=df_verify,
            daily_t=daily_buy,
            trade_date=trade_date,
            buy_date=buy_date,
        )
        if _verify_ready_rows(refreshed) > 0:
            return refreshed, f"refreshed_from_t_daily:{buy_date}:{stats.top10_hits}/{stats.top10_total}"
        return refreshed, f"refresh_no_ready_rows:{stats.reason}"
    except Exception as e:
        return df_verify, f"refresh_error:{type(e).__name__}"


def _num_series(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    seen = set()
    for name in names:
        col = _first_existing_col(df, [name])
        if not col or col in seen:
            continue
        seen.add(col)
        x = pd.to_numeric(df[col], errors="coerce")
        missing = out.isna() & x.notna()
        out.loc[missing] = x.loc[missing]
    return out.fillna(default)


def _bool_like_series(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    seen = set()
    for name in names:
        col = _first_existing_col(df, [name])
        if not col or col in seen:
            continue
        seen.add(col)
        num = pd.to_numeric(df[col], errors="coerce")
        parsed = num.clip(lower=0, upper=1)
        raw = df[col].astype(str).str.strip().str.lower()
        parsed.loc[num.isna() & raw.isin({"1", "true", "yes", "y", "是", "命中", "hit", "up"})] = 1.0
        parsed.loc[num.isna() & raw.isin({"0", "false", "no", "n", "否", "未命中", "miss", "down"})] = 0.0
        missing = out.isna() & parsed.notna()
        out.loc[missing] = parsed.loc[missing]
    return out.fillna(default)


def _t_up_actual_series(df: pd.DataFrame) -> pd.Series:
    direct = _bool_like_series(df, ["t_up_hit", "t_up_actual", "t_close_up_actual", "T日上涨实际"], default=np.nan)
    t_close = _num_series(df, ["t_close", "close_T_actual", "close_t_actual", "T日收盘价"], default=np.nan)
    d_close = _num_series(df, ["d_close", "close_T", "base_close", "D日收盘价", "收盘价"], default=np.nan)
    missing = direct.isna() & t_close.notna() & d_close.notna() & (d_close > 0)
    direct.loc[missing] = (t_close.loc[missing] > d_close.loc[missing]).astype(float)
    return direct


def _prob_series(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    seen = set()
    for name in names:
        col = _first_existing_col(df, [name])
        if not col or col in seen:
            continue
        seen.add(col)
        x = pd.to_numeric(df[col], errors="coerce")
        x = x.where(~(x > 1.0), x / 100.0).clip(lower=0.0, upper=1.0)
        missing = out.isna() & x.notna()
        out.loc[missing] = x.loc[missing]
    return out.fillna(default)


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    try:
        if method == "spearman":
            return float(pair["x"].rank(method="average").corr(pair["y"].rank(method="average"), method="pearson"))
        if method == "kendall":
            x_vals = pair["x"].to_numpy(dtype=float)
            y_vals = pair["y"].to_numpy(dtype=float)
            concordant = 0
            discordant = 0
            ties_x = 0
            ties_y = 0
            n = len(pair)
            for i in range(n - 1):
                dx = x_vals[i] - x_vals[i + 1:]
                dy = y_vals[i] - y_vals[i + 1:]
                sx = np.sign(dx)
                sy = np.sign(dy)
                ties_x += int((sx == 0).sum())
                ties_y += int((sy == 0).sum())
                prod = sx * sy
                concordant += int((prod > 0).sum())
                discordant += int((prod < 0).sum())
            denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
            return float((concordant - discordant) / denom) if denom > 0 else float("nan")
        return float(pair["x"].corr(pair["y"], method="pearson"))
    except Exception:
        return float("nan")


def _daily_rank_ic_stats(
    valid: pd.Series,
    score: pd.Series,
    actual: pd.Series,
    date_s: pd.Series,
    prefix: str,
) -> Dict[str, object]:
    tmp = pd.DataFrame({
        "date": date_s.astype(str),
        "score": pd.to_numeric(score, errors="coerce"),
        "actual": pd.to_numeric(actual, errors="coerce"),
        "valid": valid.fillna(False).astype(bool),
    })
    tmp = tmp[tmp["valid"] & tmp["date"].str.match(r"^20\d{6}$", na=False)].dropna(subset=["score", "actual"])
    if tmp.empty:
        return {
            f"{prefix}_ic_days": 0,
            f"{prefix}_ic_rows": 0,
            f"{prefix}_spearman_ic_mean": float("nan"),
            f"{prefix}_spearman_ic_median": float("nan"),
            f"{prefix}_spearman_ic_positive_rate": float("nan"),
            f"{prefix}_kendall_tau_mean": float("nan"),
            f"{prefix}_spearman_ic_all": float("nan"),
        }

    rows: List[Dict[str, object]] = []
    for d, g in tmp.groupby("date", sort=True):
        if len(g) < 3 or g["score"].nunique() < 2 or g["actual"].nunique() < 2:
            continue
        sp = _safe_corr(g["score"], g["actual"], "spearman")
        kt = _safe_corr(g["score"], g["actual"], "kendall")
        if np.isfinite(sp) or np.isfinite(kt):
            rows.append({"date": str(d), "spearman": sp, "kendall": kt, "n": int(len(g))})

    ic = pd.DataFrame(rows)
    out: Dict[str, object] = {
        f"{prefix}_ic_days": int(len(ic)),
        f"{prefix}_ic_rows": int(len(tmp)),
        f"{prefix}_spearman_ic_all": _safe_corr(tmp["score"], tmp["actual"], "spearman"),
    }
    if ic.empty:
        out.update({
            f"{prefix}_spearman_ic_mean": float("nan"),
            f"{prefix}_spearman_ic_median": float("nan"),
            f"{prefix}_spearman_ic_positive_rate": float("nan"),
            f"{prefix}_kendall_tau_mean": float("nan"),
        })
    else:
        sp = pd.to_numeric(ic["spearman"], errors="coerce")
        kt = pd.to_numeric(ic["kendall"], errors="coerce")
        out.update({
            f"{prefix}_spearman_ic_mean": float(sp.mean()) if sp.notna().any() else float("nan"),
            f"{prefix}_spearman_ic_median": float(sp.median()) if sp.notna().any() else float("nan"),
            f"{prefix}_spearman_ic_positive_rate": float((sp > 0).mean()) if sp.notna().any() else float("nan"),
            f"{prefix}_kendall_tau_mean": float(kt.mean()) if kt.notna().any() else float("nan"),
        })
        for win in (5, 20, 60):
            sub = ic.tail(win)
            spw = pd.to_numeric(sub["spearman"], errors="coerce")
            out[f"{prefix}_spearman_ic_{win}d"] = float(spw.mean()) if spw.notna().any() else float("nan")
            out[f"{prefix}_spearman_ic_positive_rate_{win}d"] = float((spw > 0).mean()) if spw.notna().any() else float("nan")
            out[f"{prefix}_ic_days_{win}d"] = int(spw.notna().sum())
    return out


def _rank_tier_stats(rank: pd.Series, valid: pd.Series, actual: pd.Series, t1_ret: pd.Series) -> Dict[str, object]:
    tiers = [
        ("top10", "TOP1-10", rank <= 10),
        ("top20_tail", "TOP11-20", (rank > 10) & (rank <= 20)),
        ("top30_tail", "TOP21-30", (rank > 20) & (rank <= 30)),
        ("after30", "TOP31+", rank > 30),
    ]
    parts: List[str] = []
    out: Dict[str, object] = {}
    for key, label, mask in tiers:
        m = valid & mask & actual.notna()
        total = int(m.sum())
        hits = int(actual[m].eq(1).sum()) if total else 0
        hit_rate = float(hits) / float(total) if total else float("nan")
        ret_m = valid & mask & t1_ret.notna()
        avg_ret = float(pd.to_numeric(t1_ret[ret_m], errors="coerce").mean()) if ret_m.any() else float("nan")
        out[f"tier_{key}_total"] = total
        out[f"tier_{key}_hits"] = hits
        out[f"tier_{key}_hit_rate"] = hit_rate
        out[f"tier_{key}_t1_ret_mean"] = avg_ret
        if total:
            ret_text = "-" if not np.isfinite(avg_ret) else f"{avg_ret:.4f}"
            parts.append(f"{label}:{hits}/{total}/{hit_rate:.3f}/ret={ret_text}")
    out["tier_summary"] = " | ".join(parts)
    if np.isfinite(out.get("tier_top10_hit_rate", float("nan"))) and np.isfinite(out.get("tier_top20_tail_hit_rate", float("nan"))):
        out["tier_top10_vs_11_20_hit_spread"] = float(out["tier_top10_hit_rate"]) - float(out["tier_top20_tail_hit_rate"])
    else:
        out["tier_top10_vs_11_20_hit_spread"] = float("nan")
    return out


def _historical_limitup_stats_from_df(df: pd.DataFrame, source: str) -> Dict[str, object]:
    if df is None or df.empty:
        return {"ready": False, "reason": "history_empty", "source": source}

    rank = _num_series(df, ["rank", "dec_rank"], default=np.nan)
    actual = _bool_like_series(df, ["t_limitup_hit", "t_limitup_actual"], default=np.nan)
    ready = _bool_like_series(df, ["label_matured", "t_limitup_verify_ready"], default=np.nan)
    prob = _prob_series(df, ["t_limitup_prob", "t_limitup_prob_model", "t_limitup_prob_rule", "T日涨停概率"], default=np.nan)
    t1_score = _prob_series(
        df,
        [
            "t1_continue_up_rate",
            "t1_continue_up_rate_rule",
            "t1_up_prob_model",
            "t1_up_rate",
            "T+1延续上涨率",
            "T+1继续上涨概率",
            "T+1上涨率",
        ],
        default=np.nan,
    )
    t1_ret = _num_series(df, ["t1_close_ret", "t1_ret", "t1_return", "real_premium_ret"], default=np.nan)
    t_up_actual = _t_up_actual_series(df)
    valid = (ready.fillna(0).eq(1) if ready.notna().any() else pd.Series(True, index=df.index)) & actual.notna() & rank.notna()
    if not valid.any():
        return {"ready": False, "reason": "no_ready_history_rows", "source": source}

    def calc(n: int) -> tuple[int, int, float]:
        m = valid & (rank <= n)
        total = int(m.sum())
        hits = int(actual[m].eq(1).sum())
        rate = float(hits) / float(total) if total else float("nan")
        return total, hits, rate

    up_valid = (ready.fillna(0).eq(1) if ready.notna().any() else pd.Series(True, index=df.index)) & rank.notna() & t_up_actual.notna()

    def calc_up(n: int) -> tuple[int, int, float]:
        m = up_valid & (rank <= n)
        total = int(m.sum())
        hits = int(t_up_actual[m].eq(1).sum())
        rate = float(hits) / float(total) if total else float("nan")
        return total, hits, rate

    top1_total, top1_hits, top1_rate = calc(1)
    top3_total, top3_hits, top3_rate = calc(3)
    top5_total, top5_hits, top5_rate = calc(5)
    top10_total, top10_hits, top10_rate = calc(10)
    top20_total, top20_hits, top20_rate = calc(20)
    top1_up_total, top1_up_hits, top1_up_rate = calc_up(1)
    top3_up_total, top3_up_hits, top3_up_rate = calc_up(3)
    top5_up_total, top5_up_hits, top5_up_rate = calc_up(5)

    date_col = _first_existing_col(df, ["_history_date", "d_trade_date", "trade_date", "base_date", "d_analysis_trade_date"])
    if date_col:
        date_s = df[date_col].map(_to_yyyymmdd)
    else:
        date_s = pd.Series(["00000000"] * len(df), index=df.index)
    n_days = 0
    if date_col:
        dates = date_s[valid]
        n_days = int(dates[dates.astype(str).str.match(r"^20\d{6}$", na=False)].nunique())

    rolling: Dict[str, object] = {}
    if date_col:
        all_dates = sorted(date_s[valid & date_s.astype(str).str.match(r"^20\d{6}$", na=False)].unique().tolist())
        for win in (5, 20, 60):
            keep_dates = set(all_dates[-win:])
            m = valid & date_s.isin(keep_dates) & (rank <= 10)
            total = int(m.sum())
            hits = int(actual[m].eq(1).sum())
            rolling[f"top10_hit_rate_{win}d"] = float(hits) / float(total) if total else float("nan")
            rolling[f"top10_hits_{win}d"] = hits
            rolling[f"top10_total_{win}d"] = total

    cal_valid = valid & prob.notna()
    brier = float(np.nanmean((prob[cal_valid] - actual[cal_valid]) ** 2)) if cal_valid.any() else float("nan")
    ece = float("nan")
    bucket_rows: List[str] = []
    if cal_valid.any():
        total_cal = int(cal_valid.sum())
        err_sum = 0.0
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]:
            m = cal_valid & (prob >= lo) & (prob < hi)
            n = int(m.sum())
            if n <= 0:
                continue
            avg_pred = float(prob[m].mean())
            hit_rate = float(actual[m].mean())
            err_sum += (n / total_cal) * abs(avg_pred - hit_rate)
            bucket_rows.append(f"{lo:.1f}-{min(hi, 1.0):.1f}:{hit_rate:.3f}/{avg_pred:.3f}/{n}")
        ece = float(err_sum)

    rank_quality: Dict[str, object] = {}
    rank_quality.update(_daily_rank_ic_stats(valid & prob.notna(), prob, actual, date_s, "limitup"))
    rank_quality.update(_daily_rank_ic_stats(valid & t1_score.notna() & t1_ret.notna(), t1_score, t1_ret, date_s, "t1_ret"))
    rank_quality.update(_rank_tier_stats(rank, valid, actual, t1_ret))

    return {
        "ready": bool(top10_total > 0 or top20_total > 0),
        "reason": "ok",
        "source": source,
        "n_days": n_days,
        "top1_total": top1_total,
        "top1_hits": top1_hits,
        "top1_hit_rate": top1_rate,
        "top3_total": top3_total,
        "top3_hits": top3_hits,
        "top3_hit_rate": top3_rate,
        "top5_total": top5_total,
        "top5_hits": top5_hits,
        "top5_hit_rate": top5_rate,
        "top1_up_total": top1_up_total,
        "top1_up_hits": top1_up_hits,
        "top1_up_rate": top1_up_rate,
        "top3_up_total": top3_up_total,
        "top3_up_hits": top3_up_hits,
        "top3_up_rate": top3_up_rate,
        "top5_up_total": top5_up_total,
        "top5_up_hits": top5_up_hits,
        "top5_up_rate": top5_up_rate,
        "top10_total": top10_total,
        "top10_hits": top10_hits,
        "top10_hit_rate": top10_rate,
        "top20_total": top20_total,
        "top20_hits": top20_hits,
        "top20_hit_rate": top20_rate,
        "calibration_rows": int(cal_valid.sum()),
        "calibration_brier": brier,
        "calibration_ece": ece,
        "calibration_bins": " | ".join(bucket_rows),
        **rolling,
        **rank_quality,
    }


def _collect_historical_limitup_stats(cfg: PremiumConfig) -> Dict[str, object]:
    trainset_path = cfg.out_learning_dir() / "limitup_probability_training_samples.csv"
    rows: List[pd.DataFrame] = []
    if trainset_path.exists():
        df = _read_optional_csv(trainset_path)
        if not df.empty:
            rows.append(df)

    for path in sorted(cfg.out_root().glob("premium_verify_*.csv")):
        df = _read_optional_csv(path)
        if df.empty:
            continue
        if "d_trade_date" not in df.columns:
            df["d_trade_date"] = _date_from_path(path) or ""
        rows.append(df)
    if not rows:
        return {"ready": False, "reason": "no_history_verify_files", "source": "premium_verify_*.csv"}
    history = pd.concat(rows, ignore_index=True, sort=False)
    code_col = _first_existing_col(history, ["ts_code", "code", "symbol"])
    if code_col:
        history["_history_date"] = ""
        for date_col in ("d_trade_date", "trade_date", "base_date", "d_analysis_trade_date"):
            if date_col not in history.columns:
                continue
            candidate_dates = history[date_col].map(_to_yyyymmdd)
            missing = ~history["_history_date"].astype(str).str.match(r"^20\d{6}$", na=False)
            history.loc[missing, "_history_date"] = candidate_dates.loc[missing]
        history["_history_code"] = history[code_col].map(_norm_ts_code)
        history = history.drop_duplicates(["_history_date", "_history_code"], keep="last")
    return _historical_limitup_stats_from_df(
        history,
        "limitup_probability_training_samples.csv+premium_verify_*.csv",
    )


def _top_frame_for_date(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    for path in (
        cfg.out_full_csv(trade_date),
        cfg.out_top30_csv(trade_date),
        cfg.out_top20_csv(trade_date),
        cfg.out_top10_csv(trade_date),
    ):
        df = _read_optional_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def _render_one(
    cfg: PremiumConfig,
    trade_date: str,
    report_dates: Sequence[str],
    historical_stats: Dict[str, object],
    training_labels: pd.DataFrame,
) -> Optional[Path]:
    df_top = _top_frame_for_date(cfg, trade_date)
    if df_top.empty:
        return None

    verify_path = cfg.out_verify_csv(trade_date)
    df_verify = _read_optional_csv(verify_path)
    if df_verify.empty:
        df_verify = df_top.copy()
    df_verify = _patch_verify_from_training_labels(df_verify, training_labels, trade_date)

    buy_date = _first_value(df_top, ["buy_date", "t_trade_date"], "")
    target_date = _first_value(df_top, ["target_date", "t1_trade_date", "next_trade_date"], "")
    if not buy_date:
        buy_date = _first_value(df_verify, ["buy_date", "t_trade_date"], "")
    if not target_date:
        target_date = _first_value(df_verify, ["target_date", "t1_trade_date", "next_trade_date"], "")

    refresh_reason = "not_attempted"
    if _verify_ready_rows(df_verify) <= 0:
        df_verify, refresh_reason = _refresh_verify_with_t_truth(cfg, df_verify, trade_date, buy_date)
        if _verify_ready_rows(df_verify) > 0:
            _write_csv(verify_path, df_verify)

    verify_pending = bool(_verify_ready_rows(df_verify) <= 0)
    verify_reason = "ok" if not verify_pending else refresh_reason

    html = render_premium_report_html(
        trade_date=trade_date,
        buy_date=buy_date or "-",
        target_date=target_date or "-",
        df_top=df_top,
        df_verify=df_verify,
        verify_pending=verify_pending,
        verify_reason=verify_reason,
        gen_ts=_utc_now_iso(),
        model_version=str(getattr(cfg, "model_version", "-")),
        audit_notes=[
            "archive_rebuilt=True",
            f"history_source={historical_stats.get('source', '-')}",
            f"truth_refresh={refresh_reason}",
        ],
        report_dates=report_dates,
        historical_limitup_stats=historical_stats,
    )
    out_path = cfg.report_html_path(trade_date)
    _write_text(out_path, html)
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild all Premium HTML reports using the current UI template.")
    p.add_argument("--latest-date", default="", help="Optional YYYYMMDD to use for premium_latest.html; default=max artifact date.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = PremiumConfig.load()
    dates = _artifact_dates(cfg)
    if not dates:
        print("[premium-report-rebuild] no premium artifacts found; skipped")
        return 0

    if args.latest_date:
        latest_date = _to_yyyymmdd(args.latest_date)
        if latest_date not in dates:
            print(f"[premium-report-rebuild] latest_date={latest_date} not in artifacts; using {dates[-1]}")
            latest_date = dates[-1]
    else:
        latest_date = dates[-1]

    historical_stats = _collect_historical_limitup_stats(cfg)
    training_labels = _load_training_labels(cfg)

    built: List[Path] = []
    skipped: List[str] = []
    for trade_date in dates:
        out_path = _render_one(cfg, trade_date, dates, historical_stats, training_labels)
        if out_path is None:
            skipped.append(trade_date)
            continue
        built.append(out_path)
        if args.verbose:
            print(f"[premium-report-rebuild] built {out_path}")

    latest_path = cfg.report_html_path(latest_date)
    if latest_path.exists():
        _write_text(cfg.report_latest_html_path(), latest_path.read_text(encoding="utf-8"))

    print(
        "[premium-report-rebuild] "
        f"built={len(built)} skipped={len(skipped)} latest={latest_date} "
        f"history_top10={historical_stats.get('top10_hits', 0)}/{historical_stats.get('top10_total', 0)}"
    )
    if skipped and args.verbose:
        print("[premium-report-rebuild] skipped_dates=" + ",".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
