#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auditable Premium TOP10 execution truth for T auction to T+1 11:00."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SCHEMA_VERSION = "premium_execution_truth_v1"
DEFAULT_START_DATE = "20260610"
DEFAULT_SELL_TIME = "11:00:00"
DEFAULT_COST_BPS = 35.0
DEFAULT_FETCH_BUDGET = 120
DATE_RE = re.compile(r"(20\d{6})")
FINAL_STATUSES = {
    "READY",
    "NO_AUCTION_MATCH",
    "NO_FILL_PRICE_CAP",
    "NO_SELL_AT_1100",
    "BAD_TRADING_DATES",
}


@dataclass(frozen=True)
class ExecutionTruthResult:
    ledger: pd.DataFrame
    summary: Dict[str, object]
    backtest: Dict[str, object]


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_code(value: object) -> str:
    text = str(value or "").strip().upper().replace("_", ".")
    if "." in text:
        left, right = text.split(".", 1)
        digits = "".join(ch for ch in left if ch.isdigit())[-6:].zfill(6)
        return f"{digits}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())[-6:].zfill(6)
    if not digits.strip("0"):
        return ""
    suffix = "BJ" if digits.startswith(("4", "8", "92")) else "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{suffix}"


def _date_from_path(path: Path) -> str:
    match = DATE_RE.search(path.name)
    return match.group(1) if match else ""


def _date_value(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        if name not in row.index:
            continue
        digits = "".join(ch for ch in str(row.get(name, "")) if ch.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return ""


def _numeric_value(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(row.get(name), errors="coerce")
        if np.isfinite(value):
            return float(value)
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(row.get(name, "")))
        if match:
            return float(match.group(1))
    return float("nan")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes()) if path.exists() else ""


def _sha256_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    ordered = frame.copy()
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    if "ts_code" in ordered.columns:
        ordered = ordered.sort_values("ts_code", kind="stable")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def _row_sha256(row: Mapping[str, object]) -> str:
    payload = json.dumps(
        {str(key): _json_safe(value) for key, value in sorted(row.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _matured(trade_date: str, clock: str) -> bool:
    try:
        target = datetime.strptime(
            f"{trade_date} {clock}", "%Y%m%d %H:%M:%S"
        ).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    except ValueError:
        return False
    return datetime.now(ZoneInfo("Asia/Shanghai")) >= target


def _load_open_dates(market_root: Path) -> List[str]:
    path = market_root / "trade_cal_sse.csv"
    frame = _read_csv(path)
    if frame.empty or not {"cal_date", "is_open"}.issubset(frame.columns):
        return []
    dates = (
        frame.loc[pd.to_numeric(frame["is_open"], errors="coerce").eq(1), "cal_date"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str[:8]
    )
    return sorted(set(dates[dates.str.fullmatch(r"\d{8}", na=False)]))


def _path_is_valid(d_date: str, t_date: str, t1_date: str, open_dates: Sequence[str]) -> bool:
    if not open_dates:
        return False
    positions = {date: idx for idx, date in enumerate(open_dates)}
    index = positions.get(d_date)
    return bool(
        index is not None
        and index + 2 < len(open_dates)
        and open_dates[index + 1] == t_date
        and open_dates[index + 2] == t1_date
    )


def _empty_record() -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "d_trade_date": "",
        "t_trade_date": "",
        "t1_trade_date": "",
        "rank": np.nan,
        "ts_code": "",
        "name": "",
        "candidate_source": "",
        "candidate_source_sha256": "",
        "candidate_row_sha256": "",
        "buy_rule": "T opening call auction; published max-buy-price cap",
        "max_buy_price": np.nan,
        "auction_price": np.nan,
        "auction_volume": np.nan,
        "auction_amount": np.nan,
        "t_up_limit": np.nan,
        "buy_cap_pass": np.nan,
        "auction_queue_risk": np.nan,
        "fill_observed": np.nan,
        "fill_confidence": "",
        "sell_rule": "T+1 11:00 minute open",
        "sell_time": "",
        "sell_price": np.nan,
        "sell_volume": np.nan,
        "gross_return": np.nan,
        "cost_bps": np.nan,
        "net_return": np.nan,
        "pnl_per_10000": np.nan,
        "is_win": np.nan,
        "model_eligible": 0,
        "status": "UNPROCESSED",
        "auction_source": "",
        "auction_source_sha256": "",
        "sell_source": "",
        "sell_source_sha256": "",
        "truth_updated_at": "",
    }


def _candidate_records(
    out_root: Path,
    project_root: Path,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_root.glob("premium_top10_*.csv")):
        d_date = _date_from_path(path)
        if not d_date or d_date < start_date or (end_date and d_date > end_date):
            continue
        frame = _read_csv(path)
        if frame.empty or "rank" not in frame.columns:
            continue
        frame = frame.copy()
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.loc[frame["rank"].between(1, 10)].sort_values("rank", kind="stable")
        source_hash = _sha256_file(path)
        source_name = _relative(path, project_root)
        for _, row in frame.iterrows():
            record = _empty_record()
            record.update(
                {
                    "d_trade_date": d_date,
                    "t_trade_date": _date_value(row, ("buy_date", "t_trade_date")),
                    "t1_trade_date": _date_value(
                        row, ("target_date", "t1_trade_date", "next_trade_date")
                    ),
                    "rank": int(row["rank"]),
                    "ts_code": _normalize_code(row.get("ts_code")),
                    "name": str(row.get("name", "")).strip(),
                    "candidate_source": source_name,
                    "candidate_source_sha256": source_hash,
                    "max_buy_price": _numeric_value(
                        row, ("t_max_buy_price", "T日可接受买入价", "max_buy_price")
                    ),
                }
            )
            record["candidate_row_sha256"] = _row_sha256(
                {
                    "d_trade_date": record["d_trade_date"],
                    "t_trade_date": record["t_trade_date"],
                    "t1_trade_date": record["t1_trade_date"],
                    "rank": record["rank"],
                    "ts_code": record["ts_code"],
                    "max_buy_price": record["max_buy_price"],
                }
            )
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=list(_empty_record().keys()))
    out = pd.DataFrame(rows)
    duplicate = out.duplicated(["d_trade_date", "rank"], keep=False)
    if duplicate.any():
        bad = out.loc[duplicate, ["d_trade_date", "rank", "ts_code"]].to_dict("records")
        raise RuntimeError(f"duplicate frozen Premium rank slots: {bad[:5]}")
    return out.sort_values(["d_trade_date", "rank"], kind="stable").reset_index(drop=True)


def _validate_frozen_history(current: pd.DataFrame, previous: pd.DataFrame) -> None:
    if current.empty or previous.empty:
        return
    fresh = current.copy()
    old = previous.copy()
    if not {"d_trade_date", "rank", "ts_code", "candidate_row_sha256", "status"}.issubset(old.columns):
        return
    for frame in (fresh, old):
        frame["d_trade_date"] = (
            frame["d_trade_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
        )
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["ts_code"] = frame["ts_code"].map(_normalize_code)
    merged = fresh.merge(
        old[["d_trade_date", "rank", "ts_code", "candidate_row_sha256", "status"]],
        on=["d_trade_date", "rank"],
        how="inner",
        suffixes=("", "_previous"),
    )
    terminal = merged["status_previous"].astype(str).isin(FINAL_STATUSES)
    drift = terminal & (
        merged["ts_code"].astype(str).ne(merged["ts_code_previous"].astype(str))
        | merged["candidate_row_sha256"].astype(str).ne(
            merged["candidate_row_sha256_previous"].astype(str)
        )
    )
    if drift.any():
        bad = merged.loc[drift, ["d_trade_date", "rank", "ts_code", "ts_code_previous"]]
        raise RuntimeError(f"frozen Premium candidate drift: {bad.to_dict('records')[:5]}")


def _load_auction_frame(market_root: Path, trade_date: str) -> Tuple[pd.DataFrame, Path]:
    root = market_root / "raw" / trade_date[:4] / trade_date
    for name in ("stk_auction.csv", "stk_auction_o.csv"):
        path = root / name
        frame = _read_csv(path)
        if frame.empty or "ts_code" not in frame.columns:
            continue
        out = frame.copy()
        out["ts_code"] = out["ts_code"].map(_normalize_code)
        return out.drop_duplicates("ts_code", keep="last"), path
    return pd.DataFrame(), root / "stk_auction.csv"


def _fetch_auction_frame(trade_date: str) -> pd.DataFrame:
    if not str(os.getenv("TUSHARE_TOKEN", "") or "").strip():
        return pd.DataFrame()
    try:
        from top10decision.data.tushare_minute import TushareClient

        return TushareClient.from_env(timeout_seconds=30).opening_auction(trade_date)
    except Exception:
        return pd.DataFrame()


def _auction_values(row: pd.Series) -> Tuple[float, float, float]:
    price = _numeric_value(row, ("price", "open", "vwap"))
    volume = _numeric_value(row, ("vol", "volume"))
    amount = _numeric_value(row, ("amount",))
    return price, volume, amount


def _load_t_up_limits(market_root: Path, trade_date: str) -> Dict[str, float]:
    candidates = (
        market_root / f"features_limit_{trade_date}.csv",
        market_root / "raw" / trade_date[:4] / trade_date / "stk_limit.csv",
    )
    for path in candidates:
        frame = _read_csv(path)
        if frame.empty or not {"ts_code", "up_limit"}.issubset(frame.columns):
            continue
        codes = frame["ts_code"].map(_normalize_code)
        values = pd.to_numeric(frame["up_limit"], errors="coerce")
        return {
            code: float(value)
            for code, value in zip(codes, values)
            if code and np.isfinite(value) and float(value) > 0
        }
    return {}


def _minute_file(market_root: Path, trade_date: str, ts_code: str) -> Path:
    return (
        market_root
        / "minute_1m"
        / trade_date[:4]
        / trade_date
        / f"{ts_code.replace('.', '_')}.csv"
    )


def _extract_sell_bar(
    frame: pd.DataFrame,
    trade_date: str,
    sell_time: str,
) -> Optional[Dict[str, object]]:
    if frame.empty:
        return None
    time_col = "time" if "time" in frame.columns else "trade_time" if "trade_time" in frame.columns else ""
    if not time_col:
        return None
    timestamps = pd.to_datetime(frame[time_col], errors="coerce")
    target = pd.to_datetime(
        f"{trade_date} {sell_time}", format="%Y%m%d %H:%M:%S", errors="coerce"
    )
    prices = pd.to_numeric(
        frame["open"] if "open" in frame.columns else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    )
    volume = pd.to_numeric(
        frame["vol"] if "vol" in frame.columns else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    )
    eligible = timestamps.eq(target) & prices.gt(0)
    if not eligible.any():
        return None
    idx = eligible[eligible].index[0]
    payload = {
        "ts_code": _normalize_code(frame.loc[idx].get("ts_code", "")),
        "time": timestamps.loc[idx].strftime("%Y-%m-%d %H:%M:%S"),
        "open": float(prices.loc[idx]),
        "vol": float(volume.loc[idx]) if np.isfinite(volume.loc[idx]) else np.nan,
        "amount": _numeric_value(frame.loc[idx], ("amount",)),
    }
    payload["sha256"] = _row_sha256(payload)
    return payload


def _local_sell_truth(
    market_root: Path, trade_date: str, ts_code: str, sell_time: str
) -> Optional[Dict[str, object]]:
    path = _minute_file(market_root, trade_date, ts_code)
    result = _extract_sell_bar(_read_csv(path), trade_date, sell_time)
    if result is not None:
        result["source"] = str(path)
    return result


def _fetch_grouped_sell_truth(
    requests: Mapping[str, Sequence[str]],
    *,
    sell_time: str,
    fetch_budget: int,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, object]],
    int,
    List[str],
    set[Tuple[str, str]],
]:
    if fetch_budget <= 0 or not requests or not str(os.getenv("TUSHARE_TOKEN", "") or "").strip():
        return {}, 0, [], set()
    try:
        from top10decision.data.tushare_minute import (
            HISTORICAL_MINUTE_FIELDS,
            TushareClient,
        )
    except Exception as exc:
        return {}, 0, [f"client_import:{type(exc).__name__}"], set()

    client = TushareClient.from_env(timeout_seconds=35)
    answers: Dict[Tuple[str, str], Dict[str, object]] = {}
    errors: List[str] = []
    queried: set[Tuple[str, str]] = set()
    calls = 0
    jobs: List[Tuple[str, str, List[str]]] = []
    for ts_code, raw_dates in requests.items():
        chunk: List[str] = []
        for trade_date in sorted(set(raw_dates)):
            if chunk:
                span = (
                    pd.Timestamp(trade_date) - pd.Timestamp(chunk[0])
                ).days
                if span > 20:
                    jobs.append((chunk[0], ts_code, chunk))
                    chunk = []
            chunk.append(trade_date)
        if chunk:
            jobs.append((chunk[0], ts_code, chunk))
    for _, ts_code, dates in sorted(jobs):
        if calls >= fetch_budget:
            break
        calls += 1
        try:
            frame = client.call(
                "stk_mins",
                {
                    "ts_code": ts_code,
                    "start_date": f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]} 09:15:00",
                    "end_date": f"{dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]} {sell_time}",
                    "freq": "1min",
                },
                HISTORICAL_MINUTE_FIELDS,
            )
            queried.update((ts_code, trade_date) for trade_date in dates)
            for trade_date in dates:
                hit = _extract_sell_bar(frame, trade_date, sell_time)
                if hit is not None:
                    hit["source"] = "tushare:stk_mins"
                    answers[(ts_code, trade_date)] = hit
        except Exception as exc:
            errors.append(f"{ts_code}:{type(exc).__name__}:{str(exc)[:120]}")
    return answers, calls, errors, queried


def _restore_terminal(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    if current.empty or previous.empty or "status" not in previous.columns:
        return current
    old = previous.copy()
    old["rank"] = pd.to_numeric(old.get("rank"), errors="coerce")
    old = old.loc[old["status"].astype(str).isin(FINAL_STATUSES)]
    if old.empty:
        return current
    keys = ["d_trade_date", "rank", "ts_code"]
    old = old.drop_duplicates(keys, keep="last").set_index(keys)
    out = current.copy().set_index(keys)
    shared = out.index.intersection(old.index)
    for column in out.columns:
        if column in old.columns and column not in {
            "candidate_source",
            "candidate_source_sha256",
            "candidate_row_sha256",
        }:
            out.loc[shared, column] = old.loc[shared, column]
    return out.reset_index()


def build_execution_truth_ledger(
    out_root: Path,
    market_root: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = "",
    cost_bps: Optional[float] = None,
    sell_time: str = DEFAULT_SELL_TIME,
    fetch_budget: int = DEFAULT_FETCH_BUDGET,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out_root = Path(out_root)
    market_root = Path(market_root)
    project_root = out_root.parents[1]
    verify_root = out_root / "verification"
    ledger_path = verify_root / "premium_execution_truth_ledger.csv"
    previous = _read_csv(ledger_path)
    current = _candidate_records(
        out_root, project_root, start_date=str(start_date), end_date=str(end_date)
    )
    _validate_frozen_history(current, previous)
    current = _restore_terminal(current, previous)
    if current.empty:
        return current, {"api_requests": 0, "fetch_errors": []}

    if cost_bps is None:
        cost_bps = float(os.getenv("PREMIUM_EXECUTION_COST_BPS", str(DEFAULT_COST_BPS)))
    open_dates = _load_open_dates(market_root)
    now_iso = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    auction_cache: Dict[str, Tuple[pd.DataFrame, Path, str]] = {}
    limit_cache: Dict[str, Dict[str, float]] = {}

    for index, row in current.iterrows():
        if str(row.get("status", "")) in FINAL_STATUSES:
            continue
        d_date = str(row["d_trade_date"])
        t_date = str(row["t_trade_date"])
        t1_date = str(row["t1_trade_date"])
        code = str(row["ts_code"])
        current.at[index, "cost_bps"] = float(cost_bps)
        current.at[index, "truth_updated_at"] = now_iso
        if not code or not _path_is_valid(d_date, t_date, t1_date, open_dates):
            current.at[index, "status"] = "BAD_TRADING_DATES"
            continue
        if not _matured(t_date, "09:26:00"):
            current.at[index, "status"] = "PENDING_T_AUCTION"
            continue

        if t_date not in auction_cache:
            auction, source_path = _load_auction_frame(market_root, t_date)
            source_label = _relative(source_path, project_root)
            if auction.empty:
                auction = _fetch_auction_frame(t_date)
                source_label = "tushare:stk_auction_o" if not auction.empty else source_label
            digest = (
                _sha256_file(source_path)
                if source_path.exists()
                else _sha256_frame(auction)
            )
            auction_cache[t_date] = (auction, source_path, digest)
        auction, auction_path, auction_hash = auction_cache[t_date]
        hit = auction.loc[auction["ts_code"].map(_normalize_code).eq(code)] if not auction.empty else pd.DataFrame()
        current.at[index, "auction_source"] = (
            _relative(auction_path, project_root) if auction_path.exists() else "tushare:stk_auction_o"
        )
        current.at[index, "auction_source_sha256"] = auction_hash
        if auction.empty:
            current.at[index, "status"] = "MISSING_T_AUCTION"
            continue
        if hit.empty:
            current.at[index, "fill_observed"] = 0
            current.at[index, "fill_confidence"] = "NO_MATCHED_AUCTION_RECORD"
            current.at[index, "status"] = "NO_AUCTION_MATCH"
            continue

        auction_price, auction_volume, auction_amount = _auction_values(hit.iloc[0])
        current.at[index, "auction_price"] = auction_price
        current.at[index, "auction_volume"] = auction_volume
        current.at[index, "auction_amount"] = auction_amount
        if not np.isfinite(auction_price) or auction_price <= 0 or not np.isfinite(auction_volume) or auction_volume <= 0:
            current.at[index, "fill_observed"] = 0
            current.at[index, "fill_confidence"] = "NO_POSITIVE_MATCHED_VOLUME"
            current.at[index, "status"] = "NO_AUCTION_MATCH"
            continue

        max_buy_price = pd.to_numeric(row.get("max_buy_price"), errors="coerce")
        cap_pass = not np.isfinite(max_buy_price) or auction_price <= float(max_buy_price) + 1e-9
        current.at[index, "buy_cap_pass"] = int(cap_pass)
        current.at[index, "fill_observed"] = int(cap_pass)
        if not cap_pass:
            current.at[index, "fill_confidence"] = "RULE_PRICE_CAP_REJECTED"
            current.at[index, "status"] = "NO_FILL_PRICE_CAP"
            continue

        if t_date not in limit_cache:
            limit_cache[t_date] = _load_t_up_limits(market_root, t_date)
        t_up_limit = limit_cache[t_date].get(code, float("nan"))
        queue_risk = bool(np.isfinite(t_up_limit) and auction_price >= t_up_limit * 0.9985)
        current.at[index, "t_up_limit"] = t_up_limit
        current.at[index, "auction_queue_risk"] = int(queue_risk)
        current.at[index, "fill_confidence"] = (
            "LOW_LIMIT_QUEUE_SHADOW_MATCH" if queue_risk else "MEDIUM_MATCHED_AUCTION_VOLUME"
        )
        current.at[index, "status"] = (
            "PENDING_T1_1100" if not _matured(t1_date, "11:01:00") else "MISSING_T1_1100"
        )

    requests: Dict[str, List[str]] = {}
    local_truth: Dict[Tuple[str, str], Dict[str, object]] = {}
    needs_sell = current["status"].astype(str).eq("MISSING_T1_1100")
    for index, row in current.loc[needs_sell].iterrows():
        code = str(row["ts_code"])
        trade_date = str(row["t1_trade_date"])
        hit = _local_sell_truth(market_root, trade_date, code, sell_time)
        if hit is not None:
            local_truth[(code, trade_date)] = hit
        else:
            requests.setdefault(code, []).append(trade_date)

    fetched_truth, api_requests, fetch_errors, queried_sell_keys = _fetch_grouped_sell_truth(
        requests, sell_time=sell_time, fetch_budget=max(0, int(fetch_budget))
    )
    sell_truth = {**local_truth, **fetched_truth}
    for index, row in current.loc[needs_sell].iterrows():
        code = str(row["ts_code"])
        trade_date = str(row["t1_trade_date"])
        hit = sell_truth.get((code, trade_date))
        if hit is None:
            if (code, trade_date) in queried_sell_keys:
                current.at[index, "status"] = "NO_SELL_AT_1100"
            continue
        buy_price = float(row["auction_price"])
        sell_price = float(hit["open"])
        gross_return = sell_price / buy_price - 1.0
        net_return = gross_return - float(cost_bps) / 10000.0
        queue_value = pd.to_numeric(row.get("auction_queue_risk"), errors="coerce")
        queue_risk = bool(np.isfinite(queue_value) and int(queue_value) == 1)
        source = str(hit.get("source", ""))
        source_path = Path(source) if source and not source.startswith("tushare:") else None
        current.at[index, "sell_time"] = str(hit["time"])
        current.at[index, "sell_price"] = sell_price
        current.at[index, "sell_volume"] = hit.get("vol", np.nan)
        current.at[index, "gross_return"] = gross_return
        current.at[index, "net_return"] = net_return
        current.at[index, "pnl_per_10000"] = net_return * 10000.0
        current.at[index, "is_win"] = int(net_return > 0)
        current.at[index, "model_eligible"] = int(not queue_risk)
        current.at[index, "status"] = "READY"
        current.at[index, "sell_source"] = (
            _relative(source_path, project_root) if source_path is not None else source
        )
        current.at[index, "sell_source_sha256"] = str(hit.get("sha256", ""))
        current.at[index, "truth_updated_at"] = now_iso

    current["rank"] = pd.to_numeric(current["rank"], errors="coerce").astype("Int64")
    current = current.sort_values(["d_trade_date", "rank"], kind="stable").reset_index(drop=True)
    runtime = {"api_requests": api_requests, "fetch_errors": fetch_errors}
    return current, runtime


def summarize_execution_truth(
    ledger: pd.DataFrame, runtime: Optional[Mapping[str, object]] = None
) -> Dict[str, object]:
    runtime = runtime or {}
    status = ledger.get("status", pd.Series("", index=ledger.index)).astype(str)
    ready = ledger.loc[status.eq("READY")].copy()
    eligible = ready.loc[pd.to_numeric(ready.get("model_eligible"), errors="coerce").eq(1)]
    settled = ledger.loc[status.isin(FINAL_STATUSES)].copy()
    expected = int(ledger["d_trade_date"].nunique() * 10) if not ledger.empty else 0
    per_day = ledger.groupby("d_trade_date")["rank"].nunique() if not ledger.empty else pd.Series(dtype=float)
    settled_per_day = settled.groupby("d_trade_date")["rank"].nunique() if not settled.empty else pd.Series(dtype=float)
    ready_per_day = ready.groupby("d_trade_date")["rank"].nunique() if not ready.empty else pd.Series(dtype=float)
    return {
        "schema_version": SCHEMA_VERSION,
        "records": int(len(ledger)),
        "d_days": int(ledger["d_trade_date"].nunique()) if not ledger.empty else 0,
        "expected_top10_records": expected,
        "candidate_complete_days": int((per_day >= 10).sum()),
        "truth_complete_days": int((settled_per_day >= 10).sum()),
        "all_filled_days": int((ready_per_day >= 10).sum()),
        "ready_records": int(len(ready)),
        "model_eligible_records": int(len(eligible)),
        "pending_records": int(status.str.startswith("PENDING").sum()),
        "missing_records": int(status.str.startswith("MISSING").sum()),
        "no_fill_records": int(status.isin({"NO_AUCTION_MATCH", "NO_FILL_PRICE_CAP"}).sum()),
        "no_sell_records": int(status.eq("NO_SELL_AT_1100").sum()),
        "queue_risk_records": int(
            pd.to_numeric(ledger.get("auction_queue_risk"), errors="coerce").eq(1).sum()
        ),
        "ready_coverage": float(len(ready) / expected) if expected else float("nan"),
        "api_requests": int(runtime.get("api_requests", 0) or 0),
        "fetch_error_count": len(runtime.get("fetch_errors", []) or []),
        "fetch_errors": list(runtime.get("fetch_errors", []) or [])[:20],
        "first_d_date": str(ledger["d_trade_date"].min()) if not ledger.empty else "",
        "last_d_date": str(ledger["d_trade_date"].max()) if not ledger.empty else "",
        "entry": "T official opening-auction matched price with published price cap",
        "exit": "T+1 exact 11:00 one-minute bar open",
        "warning": "Auction matched volume is market truth, not broker order-level fill confirmation.",
    }


def build_actual_top1_backtest(ledger: pd.DataFrame) -> Dict[str, object]:
    if ledger.empty:
        rank1 = ledger.copy()
    else:
        rank1 = ledger.loc[pd.to_numeric(ledger["rank"], errors="coerce").eq(1)].copy()
    mature = rank1.loc[~rank1.get("status", pd.Series("", index=rank1.index)).astype(str).str.startswith("PENDING")]
    filled = rank1.loc[rank1.get("status", pd.Series("", index=rank1.index)).astype(str).eq("READY")].copy()
    returns = pd.to_numeric(filled.get("net_return"), errors="coerce").dropna()
    if returns.empty:
        compound = drawdown = average = median = win_rate = float("nan")
        wins = 0
    else:
        position_returns = returns * 0.10
        equity = (1.0 + position_returns).cumprod()
        compound = float(equity.iloc[-1] - 1.0)
        drawdown = float((equity / equity.cummax() - 1.0).min())
        average = float(returns.mean())
        median = float(returns.median())
        wins = int((returns > 0).sum())
        win_rate = float(wins / len(returns))
    return {
        "schema_version": "premium_execution_actual_top1_backtest_v1",
        "model_version": "premium_execution_truth_v1",
        "window_start": str(rank1["d_trade_date"].min()) if not rank1.empty else "",
        "window_end": str(rank1["d_trade_date"].max()) if not rank1.empty else "",
        "a_share_trading_days": int(rank1["d_trade_date"].nunique()) if not rank1.empty else 0,
        "warmup_trade_days": 0,
        "mature_evaluation_days": int(mature["d_trade_date"].nunique()) if not mature.empty else 0,
        "model_ready_days": 0,
        "signals": int(len(mature)),
        "filled_signals": int(len(returns)),
        "winning_filled_signals": wins,
        "filled_win_rate": win_rate,
        "filled_average_net_return": average,
        "filled_median_net_return": median,
        "full_position_compound_return": float((1.0 + returns).prod() - 1.0) if len(returns) else float("nan"),
        "full_position_max_drawdown": (
            float(((1.0 + returns).cumprod() / (1.0 + returns).cumprod().cummax() - 1.0).min())
            if len(returns)
            else float("nan")
        ),
        "ten_percent_position_compound_return": compound,
        "ten_percent_position_max_drawdown": drawdown,
        "recent_holdout_signals": 0,
        "recent_holdout_filled_signals": 0,
        "recent_holdout_compound_return": float("nan"),
        "cost_bps": float(pd.to_numeric(filled.get("cost_bps"), errors="coerce").dropna().iloc[-1]) if not filled.empty else DEFAULT_COST_BPS,
        "entry": "T official opening-auction matched price with published price cap",
        "exit": "T+1 exact 11:00 one-minute bar open",
        "validation": "frozen original Rank 1 shadow ledger; no reranking and no proxy prices",
        "warning": "Descriptive shadow result only; auction matched volume is not broker order-level fill confirmation and does not guarantee future profit.",
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def write_execution_truth_artifacts(
    out_root: Path, result: ExecutionTruthResult
) -> Dict[str, Path]:
    verify_root = Path(out_root) / "verification"
    models_root = Path(out_root) / "models"
    verify_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    ledger_path = verify_root / "premium_execution_truth_ledger.csv"
    summary_path = verify_root / "premium_execution_truth_summary.json"
    backtest_path = models_root / "execution_actual_top1_backtest_meta.json"
    result.ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(_json_safe(result.summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    backtest_path.write_text(
        json.dumps(_json_safe(result.backtest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {"ledger": ledger_path, "summary": summary_path, "backtest": backtest_path}


def build_and_write_execution_truth(
    out_root: Path,
    market_root: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = "",
    cost_bps: Optional[float] = None,
    sell_time: str = DEFAULT_SELL_TIME,
    fetch_budget: int = DEFAULT_FETCH_BUDGET,
) -> Tuple[ExecutionTruthResult, Dict[str, Path]]:
    ledger, runtime = build_execution_truth_ledger(
        out_root,
        market_root,
        start_date=start_date,
        end_date=end_date,
        cost_bps=cost_bps,
        sell_time=sell_time,
        fetch_budget=fetch_budget,
    )
    result = ExecutionTruthResult(
        ledger=ledger,
        summary=summarize_execution_truth(ledger, runtime),
        backtest=build_actual_top1_backtest(ledger),
    )
    return result, write_execution_truth_artifacts(out_root, result)


__all__ = [
    "DEFAULT_COST_BPS",
    "DEFAULT_FETCH_BUDGET",
    "DEFAULT_SELL_TIME",
    "DEFAULT_START_DATE",
    "ExecutionTruthResult",
    "SCHEMA_VERSION",
    "build_actual_top1_backtest",
    "build_and_write_execution_truth",
    "build_execution_truth_ledger",
    "summarize_execution_truth",
    "write_execution_truth_artifacts",
]
