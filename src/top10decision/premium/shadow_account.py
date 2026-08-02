#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Premium TOP1 shadow ledger using executable market truth only."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"(20\d{6})")
DEFAULT_START_DATE = "20260701"
DEFAULT_SELL_TIME = "11:00:00"
DEFAULT_COST_BPS = 35.0


@dataclass(frozen=True)
class ShadowBuildResult:
    ledger: pd.DataFrame
    summary: Dict[str, object]


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return pd.read_csv(path)


def _date_from_path(path: Path) -> str:
    match = DATE_RE.search(path.name)
    return match.group(1) if match else ""


def _date_value(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        if name not in row.index:
            continue
        value = str(row.get(name, "")).strip()
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return ""


def _normalize_code(value: object) -> str:
    text = str(value or "").strip().upper().replace("_", ".")
    if "." in text:
        left, right = text.split(".", 1)
        digits = "".join(ch for ch in left if ch.isdigit())[-6:].zfill(6)
        return f"{digits}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())[-6:].zfill(6)
    if digits.startswith(("4", "8", "92")):
        suffix = "BJ"
    elif digits.startswith(("5", "6", "9")):
        suffix = "SH"
    else:
        suffix = "SZ"
    return f"{digits}.{suffix}"


def _rank1(path: Path) -> Optional[pd.Series]:
    frame = _read_csv(path)
    if frame.empty or "rank" not in frame.columns:
        return None
    rank = pd.to_numeric(frame["rank"], errors="coerce")
    hit = frame.loc[rank.eq(1)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _truth_matured(trade_date: str, clock: str) -> bool:
    try:
        target = datetime.strptime(
            f"{trade_date} {clock}", "%Y%m%d %H:%M:%S"
        ).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    except ValueError:
        return False
    return datetime.now(ZoneInfo("Asia/Shanghai")) >= target


def _auction_truth(market_root: Path, trade_date: str, ts_code: str) -> Tuple[str, float, float, str]:
    root = market_root / "raw" / trade_date[:4] / trade_date
    candidates = (
        (root / "stk_auction.csv", ("price", "open")),
        (root / "stk_auction_o.csv", ("open", "price")),
    )
    for path, price_columns in candidates:
        if not path.exists():
            continue
        frame = _read_csv(path)
        if frame.empty or "ts_code" not in frame.columns:
            continue
        codes = frame["ts_code"].map(_normalize_code)
        hit = frame.loc[codes.eq(ts_code)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        price = float("nan")
        for column in price_columns:
            candidate = pd.to_numeric(row.get(column), errors="coerce")
            if np.isfinite(candidate) and float(candidate) > 0:
                price = float(candidate)
                break
        volume = pd.to_numeric(row.get("vol"), errors="coerce")
        if np.isfinite(price) and np.isfinite(volume) and float(volume) > 0:
            return "READY", price, float(volume), str(path)

    missing_path = candidates[0][0]
    status = "MISSING_T_AUCTION" if _truth_matured(trade_date, "09:26:00") else "PENDING_T_AUCTION"
    return status, float("nan"), float("nan"), str(missing_path)


def _fetch_historical_minute(
    market_root: Path,
    trade_date: str,
    ts_code: str,
    sell_time: str,
) -> Optional[Path]:
    if not str(os.getenv("TUSHARE_TOKEN", "") or "").strip():
        return None
    try:
        from top10decision.data.tushare_minute import TushareClient, write_minute_snapshot

        repo_root = market_root.parent.parent
        client = TushareClient.from_env(timeout_seconds=30)
        frame = client.historical_minute(
            ts_code,
            trade_date,
            latest_time=sell_time[:5],
        )
        if frame.empty:
            return None
        path, _ = write_minute_snapshot(
            frame,
            repo_root,
            trade_date,
            ts_code,
            source="tushare:stk_mins",
        )
        return path
    except Exception:
        return None


def _sell_truth(
    market_root: Path,
    trade_date: str,
    ts_code: str,
    sell_time: str,
) -> Tuple[str, float, str, str]:
    file_name = ts_code.replace(".", "_") + ".csv"
    path = market_root / "minute_1m" / trade_date[:4] / trade_date / file_name
    if not path.exists():
        if not _truth_matured(trade_date, "11:01:00"):
            return "PENDING_T1_1100", float("nan"), "", str(path)
        fetched = _fetch_historical_minute(market_root, trade_date, ts_code, sell_time)
        if fetched is None:
            return "MISSING_T1_1100", float("nan"), "", str(path)
        path = fetched
    frame = _read_csv(path)
    if frame.empty or "time" not in frame.columns:
        return "MISSING_T1_1100", float("nan"), "", str(path)

    timestamps = pd.to_datetime(frame["time"], errors="coerce")
    session_date = pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
    if pd.isna(session_date):
        return "BAD_T1_DATE", float("nan"), "", str(path)
    start = session_date + pd.Timedelta(hours=int(sell_time[:2]), minutes=int(sell_time[3:5]))
    end = start
    volume = (
        pd.to_numeric(frame["vol"], errors="coerce")
        if "vol" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    price = (
        pd.to_numeric(frame["open"], errors="coerce")
        if "open" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    eligible = timestamps.ge(start) & timestamps.le(end) & price.gt(0) & volume.gt(0)
    hit = frame.loc[eligible].copy()
    if hit.empty:
        return "MISSING_T1_1100", float("nan"), "", str(path)
    hit["_timestamp"] = timestamps.loc[eligible]
    hit["_price"] = price.loc[eligible]
    row = hit.sort_values("_timestamp", kind="stable").iloc[0]
    return "READY", float(row["_price"]), row["_timestamp"].strftime("%Y-%m-%d %H:%M:%S"), str(path)


def _empty_record(d_trade_date: str) -> Dict[str, object]:
    return {
        "d_trade_date": d_trade_date,
        "t_trade_date": "",
        "t1_trade_date": "",
        "rank": 1,
        "ts_code": "",
        "name": "",
        "buy_time": "09:25:00",
        "buy_price": np.nan,
        "auction_volume": np.nan,
        "sell_rule": "T+1 11:00 minute open",
        "sell_time": "",
        "sell_price": np.nan,
        "gross_return": np.nan,
        "cost_bps": np.nan,
        "net_return": np.nan,
        "pnl_per_10000": np.nan,
        "is_win": np.nan,
        "status": "MISSING_RANK1",
        "fill_assumption": "small shadow order; positive matched auction volume",
        "buy_source": "",
        "sell_source": "",
    }


def build_top1_shadow_ledger(
    out_root: Path,
    market_root: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = "",
    cost_bps: Optional[float] = None,
    sell_time: str = DEFAULT_SELL_TIME,
) -> pd.DataFrame:
    """Build one immutable TOP1 shadow record per Premium D-day artifact."""
    if cost_bps is None:
        cost_bps = float(os.getenv("PREMIUM_SHADOW_COST_BPS", str(DEFAULT_COST_BPS)))
    rows = []
    for path in sorted(out_root.glob("premium_top10_*.csv")):
        d_trade_date = _date_from_path(path)
        if not d_trade_date or d_trade_date < str(start_date):
            continue
        if end_date and d_trade_date > str(end_date):
            continue
        record = _empty_record(d_trade_date)
        try:
            row = _rank1(path)
        except Exception:
            row = None
        if row is None:
            rows.append(record)
            continue

        ts_code = _normalize_code(row.get("ts_code"))
        t_trade_date = _date_value(row, ("buy_date", "t_trade_date"))
        t1_trade_date = _date_value(row, ("target_date", "t1_trade_date", "next_trade_date"))
        record.update(
            {
                "t_trade_date": t_trade_date,
                "t1_trade_date": t1_trade_date,
                "ts_code": ts_code,
                "name": str(row.get("name", "")).strip(),
                "cost_bps": float(cost_bps),
            }
        )
        if not t_trade_date or not t1_trade_date:
            record["status"] = "MISSING_TRADING_DATES"
            rows.append(record)
            continue

        buy_status, buy_price, auction_volume, buy_source = _auction_truth(
            market_root, t_trade_date, ts_code
        )
        record.update(
            {
                "buy_price": buy_price,
                "auction_volume": auction_volume,
                "buy_source": buy_source,
            }
        )
        if buy_status != "READY":
            record["status"] = buy_status
            rows.append(record)
            continue

        sell_status, sell_price, actual_sell_time, sell_source = _sell_truth(
            market_root, t1_trade_date, ts_code, sell_time
        )
        record.update(
            {
                "sell_price": sell_price,
                "sell_time": actual_sell_time,
                "sell_source": sell_source,
            }
        )
        if sell_status != "READY":
            record["status"] = sell_status
            rows.append(record)
            continue

        gross_return = sell_price / buy_price - 1.0
        net_return = gross_return - float(cost_bps) / 10000.0
        record.update(
            {
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl_per_10000": net_return * 10000.0,
                "is_win": int(net_return > 0),
                "status": "READY",
            }
        )
        rows.append(record)

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return pd.DataFrame(columns=list(_empty_record("").keys()))
    return ledger.sort_values("d_trade_date", kind="stable").reset_index(drop=True)


def summarize_top1_shadow(ledger: pd.DataFrame) -> Dict[str, object]:
    statuses = (
        ledger["status"].astype(str)
        if "status" in ledger.columns
        else pd.Series("", index=ledger.index, dtype=str)
    )
    ready = ledger.loc[statuses.eq("READY")].copy()
    returns = (
        pd.to_numeric(ready["net_return"], errors="coerce").dropna()
        if "net_return" in ready.columns
        else pd.Series(dtype=float)
    )
    if returns.empty:
        total = compound = average = median = max_drawdown = float("nan")
        wins = 0
    else:
        equity = (1.0 + returns).cumprod()
        peak = equity.cummax()
        total = float(returns.sum())
        compound = float(equity.iloc[-1] - 1.0)
        average = float(returns.mean())
        median = float(returns.median())
        max_drawdown = float((equity / peak - 1.0).min())
        wins = int((returns > 0).sum())
    pending = int(statuses.str.startswith("PENDING").sum())
    missing = int(statuses.str.startswith("MISSING").sum())
    cost_values = (
        pd.to_numeric(ledger["cost_bps"], errors="coerce").dropna()
        if "cost_bps" in ledger.columns
        else pd.Series(dtype=float)
    )
    return {
        "generated_from": "premium_top10_*.csv",
        "buy_rule": "T opening call auction actual matched price",
        "sell_rule": "T+1 11:00 minute open",
        "fill_assumption": "small shadow order and positive auction matched volume",
        "cost_bps": float(cost_values.iloc[-1]) if not cost_values.empty else DEFAULT_COST_BPS,
        "records": int(len(ledger)),
        "completed": int(len(returns)),
        "pending": pending,
        "missing": missing,
        "wins": wins,
        "losses": int(len(returns) - wins),
        "win_rate": float(wins / len(returns)) if len(returns) else float("nan"),
        "average_net_return": average,
        "median_net_return": median,
        "total_net_return": total,
        "unit_compound_return": compound,
        "unit_max_drawdown": max_drawdown,
        "first_d_date": (
            str(ledger["d_trade_date"].min())
            if len(ledger) and "d_trade_date" in ledger.columns
            else ""
        ),
        "last_d_date": (
            str(ledger["d_trade_date"].max())
            if len(ledger) and "d_trade_date" in ledger.columns
            else ""
        ),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_top1_shadow_artifacts(
    out_root: Path,
    result: ShadowBuildResult,
) -> Dict[str, Path]:
    verify_root = out_root / "verification"
    verify_root.mkdir(parents=True, exist_ok=True)
    ledger_path = verify_root / "premium_top1_shadow_ledger.csv"
    summary_path = verify_root / "premium_top1_shadow_summary.json"
    result.ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(_json_safe(result.summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths = {"ledger": ledger_path, "summary": summary_path}
    if not result.ledger.empty:
        months = result.ledger["d_trade_date"].astype(str).str[:6]
        for month in sorted(months.dropna().unique()):
            monthly_path = verify_root / f"premium_top1_shadow_{month}.csv"
            result.ledger.loc[months.eq(month)].to_csv(
                monthly_path, index=False, encoding="utf-8-sig"
            )
            paths[f"month_{month}"] = monthly_path
    return paths


def build_and_write_top1_shadow(
    out_root: Path,
    market_root: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = "",
    cost_bps: Optional[float] = None,
    sell_time: str = DEFAULT_SELL_TIME,
) -> Tuple[ShadowBuildResult, Dict[str, Path]]:
    ledger = build_top1_shadow_ledger(
        out_root,
        market_root,
        start_date=start_date,
        end_date=end_date,
        cost_bps=cost_bps,
        sell_time=sell_time,
    )
    result = ShadowBuildResult(ledger=ledger, summary=summarize_top1_shadow(ledger))
    return result, write_top1_shadow_artifacts(out_root, result)


__all__ = [
    "DEFAULT_COST_BPS",
    "DEFAULT_SELL_TIME",
    "DEFAULT_START_DATE",
    "ShadowBuildResult",
    "build_and_write_top1_shadow",
    "build_top1_shadow_ledger",
    "summarize_top1_shadow",
    "write_top1_shadow_artifacts",
]
