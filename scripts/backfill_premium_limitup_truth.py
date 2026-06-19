#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill Premium T-day limit-up truth for matured verify files.

This script is intentionally standalone. It repairs historical
premium_verify_*.csv files before HTML archive rebuilds, including files that
already contain stale empty truth columns.
"""

from __future__ import annotations

import argparse
import math
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


RECOMPUTE_COLS = [
    "open_T_actual",
    "high_T_actual",
    "close_T_actual",
    "t_limit_price_est",
    "t_open_ret",
    "t_intraday_ret",
    "t_close_ret",
    "t_up_actual",
    "t_high_profit_hit",
    "t_limitup_actual",
    "t_touch_limitup_actual",
    "t_limitup_verify_ready",
    "t_limitup_verify_reason",
    "t_limitup_verify_trade_date",
    "d_analysis_trade_date",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_date(x: object) -> str:
    s = str(x or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "".join(ch for ch in s if ch.isdigit())[:8]


def _canonical_ts_code(x: object) -> str:
    s = str(x or "").strip().upper()
    if not s or s in {"NAN", "NONE", "<NA>"}:
        return ""
    if "." in s:
        raw, suffix = s.split(".", 1)
        raw = "".join(ch for ch in raw if ch.isdigit())[-6:]
        suffix = suffix[:2]
        return f"{raw}.{suffix}" if raw and suffix else s
    digits = "".join(ch for ch in s if ch.isdigit())
    raw = digits[-6:]
    if not raw:
        return ""
    if s.startswith(("SH", "SZ", "BJ")):
        suffix = s[:2]
    elif s.endswith(("SH", "SZ", "BJ")):
        suffix = s[-2:]
    elif raw.startswith(("43", "83", "87", "88", "92")):
        suffix = "BJ"
    elif raw.startswith(("5", "6", "9")):
        suffix = "SH"
    else:
        suffix = "SZ"
    return f"{raw}.{suffix}"


def _limit_rate_for_code(x: object) -> float:
    code = _canonical_ts_code(x)
    raw = code.split(".", 1)[0] if "." in code else "".join(ch for ch in code if ch.isdigit())[-6:]
    suffix = code.split(".", 1)[1] if "." in code else ""
    if suffix == "BJ" or raw.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if raw.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ts_code": str}, encoding="utf-8-sig")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _remote_daily_url(trade_date: str) -> str:
    year = trade_date[:4]
    return f"https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main/data/raw/{year}/{trade_date}/daily.csv"


def _daily_candidates(root: Path, trade_date: str) -> Iterable[Path]:
    year = trade_date[:4]
    yield root / "data" / "market" / f"daily_{trade_date}.csv"
    yield root / "data" / "raw" / year / trade_date / "daily.csv"


def _load_daily(root: Path, trade_date: str) -> tuple[pd.DataFrame | None, str]:
    for path in _daily_candidates(root, trade_date):
        if path.exists():
            try:
                return _read_csv(path), f"local:{path}"
            except Exception as exc:
                last_err = f"local_invalid:{path}:{type(exc).__name__}"
                break
    else:
        last_err = "local_missing"

    url = _remote_daily_url(trade_date)
    try:
        with urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8-sig")
        df = pd.read_csv(StringIO(text), dtype={"ts_code": str})
        cache = root / "data" / "market" / f"daily_{trade_date}.csv"
        _write_csv(cache, df)
        return df, f"remote:{url}"
    except HTTPError as exc:
        return None, f"remote_http_{exc.code}:{url}"
    except URLError as exc:
        return None, f"{last_err};remote_error:{type(exc.reason).__name__}"
    except Exception as exc:
        return None, f"{last_err};remote_error:{type(exc).__name__}"


def _first_value(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns or df.empty:
        return ""
    values = df[col].dropna().astype(str)
    return values.iloc[0] if len(values) else ""


def _ready_rows(df: pd.DataFrame) -> int:
    if "t_limitup_verify_ready" not in df.columns:
        return 0
    ready = pd.to_numeric(df["t_limitup_verify_ready"], errors="coerce").fillna(0)
    return int(ready.eq(1).sum())


def _attach_truth(df_verify: pd.DataFrame, daily_t: pd.DataFrame, trade_date: str, buy_date: str) -> pd.DataFrame:
    if df_verify.empty:
        return df_verify
    if "ts_code" not in df_verify.columns:
        out = df_verify.copy()
        out["t_limitup_verify_ready"] = 0
        out["t_limitup_verify_reason"] = "verify_missing_ts_code"
        return out
    required = {"ts_code", "open", "high", "close"}
    if not required.issubset(set(daily_t.columns)):
        out = df_verify.copy()
        out["t_limitup_verify_ready"] = 0
        out["t_limitup_verify_reason"] = "daily_missing_price_cols"
        return out

    original_cols = list(df_verify.columns)
    daily = daily_t[["ts_code", "open", "high", "close"]].copy()
    daily["_premium_join_ts_code"] = daily["ts_code"].map(_canonical_ts_code)
    daily = daily[daily["_premium_join_ts_code"].astype(str).str.len() > 0]
    daily = daily.drop_duplicates("_premium_join_ts_code", keep="last")
    daily = daily.rename(
        columns={
            "open": "open_T_actual",
            "high": "high_T_actual",
            "close": "close_T_actual",
        }
    )
    daily = daily.drop(columns=["ts_code"], errors="ignore")

    out = df_verify.copy().drop(columns=[c for c in RECOMPUTE_COLS if c in df_verify.columns], errors="ignore")
    out["_premium_join_ts_code"] = out["ts_code"].map(_canonical_ts_code)
    out = out.merge(daily, on="_premium_join_ts_code", how="left")
    out = out.drop(columns=["_premium_join_ts_code"], errors="ignore")

    d_close = pd.to_numeric(out.get("close_T"), errors="coerce")
    t_open = pd.to_numeric(out.get("open_T_actual"), errors="coerce")
    t_high = pd.to_numeric(out.get("high_T_actual"), errors="coerce")
    t_close = pd.to_numeric(out.get("close_T_actual"), errors="coerce")
    rates = out["ts_code"].map(_limit_rate_for_code).astype(float)
    limit_px = (d_close * (1.0 + rates)).round(2)
    ready = d_close.notna() & t_high.notna() & t_close.notna() & d_close.gt(0)

    out["t_limit_price_est"] = limit_px
    out["t_open_ret"] = np.where(ready & t_open.notna(), t_open / d_close - 1.0, pd.NA)
    out["t_intraday_ret"] = np.where(ready, t_high / d_close - 1.0, pd.NA)
    out["t_close_ret"] = np.where(ready, t_close / d_close - 1.0, pd.NA)
    out["t_up_actual"] = np.where(ready, t_close.gt(d_close).astype(int), pd.NA)
    out["t_high_profit_hit"] = np.where(ready, (pd.to_numeric(out["t_intraday_ret"], errors="coerce") >= 0.02).astype(int), pd.NA)
    out["t_limitup_actual"] = np.where(ready, t_close.ge(limit_px * 0.9985).astype(int), pd.NA)
    out["t_touch_limitup_actual"] = np.where(ready, t_high.ge(limit_px * 0.9985).astype(int), pd.NA)
    out["t_limitup_verify_ready"] = ready.astype(int)
    out["t_limitup_verify_reason"] = np.where(ready, "ok", "missing_D_or_T_price")
    out["t_limitup_verify_trade_date"] = str(buy_date)
    out["d_analysis_trade_date"] = str(trade_date)

    for col in original_cols:
        if col not in out.columns:
            out[col] = pd.NA
    tail_cols = [c for c in out.columns if c not in original_cols]
    return out[original_cols + tail_cols]


def _top10_summary(df: pd.DataFrame) -> str:
    ready = pd.to_numeric(df.get("t_limitup_verify_ready"), errors="coerce").fillna(0).eq(1)
    rank = pd.to_numeric(df.get("rank"), errors="coerce")
    actual = pd.to_numeric(df.get("t_limitup_actual"), errors="coerce")
    mask = ready & rank.le(10) & actual.notna()
    total = int(mask.sum())
    hits = int(actual[mask].eq(1).sum()) if total else 0
    rate = hits / total if total else math.nan
    return f"{hits}/{total}" if not math.isfinite(rate) else f"{hits}/{total} ({rate:.2%})"


def backfill(root: Path, verbose: bool = False) -> int:
    out_dir = root / "outputs" / "premium"
    files = sorted(out_dir.glob("premium_verify_*.csv"))
    changed = 0
    for path in files:
        df = _read_csv(path)
        trade_date = _clean_date(_first_value(df, "trade_date")) or path.stem.rsplit("_", 1)[-1]
        buy_date = _clean_date(_first_value(df, "buy_date"))
        if not buy_date:
            if verbose:
                print(f"[skip] {path.name}: missing buy_date")
            continue

        daily, source = _load_daily(root, buy_date)
        if daily is None or daily.empty:
            if verbose:
                print(f"[skip] {path.name}: T daily not ready {buy_date}: {source}")
            continue

        before_ready = _ready_rows(df)
        fixed = _attach_truth(df, daily, trade_date, buy_date)
        after_ready = _ready_rows(fixed)
        if after_ready <= 0:
            if verbose:
                print(f"[skip] {path.name}: no matched rows after attach, source={source}")
            continue

        if not fixed.equals(df):
            _write_csv(path, fixed)
            changed += 1
            if verbose:
                print(f"[ok] {path.name}: ready {before_ready}->{after_ready}, top10={_top10_summary(fixed)}, source={source}")
        elif verbose:
            print(f"[ok] {path.name}: already current, top10={_top10_summary(fixed)}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    root = _repo_root()
    changed = backfill(root, verbose=args.verbose)
    print(f"premium_limitup_truth_backfill_changed={changed}")


if __name__ == "__main__":
    main()
