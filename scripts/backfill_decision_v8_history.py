#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.auction_v3 import AuctionV3Config, AuctionV3Engine  # noqa: E402
from top10decision.data.tushare_minute import (  # noqa: E402
    AUCTION_OPEN_FIELDS,
    TushareClient,
    write_calendar,
)


DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
)
LIMIT_FIELDS = (
    "ts_code",
    "trade_date",
    "pre_close",
    "up_limit",
    "down_limit",
)
DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "turnover_rate",
    "volume_ratio",
    "total_mv",
    "circ_mv",
)
LIMIT_LIST_FIELDS = (
    "trade_date",
    "ts_code",
    "industry",
    "name",
    "close",
    "pct_chg",
    "amount",
    "float_mv",
    "total_mv",
    "turnover_ratio",
    "fd_amount",
    "first_time",
    "last_time",
    "open_times",
    "limit_times",
    "limit",
)


def _date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _call_paged(
    client: TushareClient,
    api_name: str,
    params: dict[str, Any],
    fields: Iterable[str],
    *,
    page_size: int = 6_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for page in range(3):
        page_params = {
            **params,
            "limit": page_size,
            "offset": page * page_size,
        }
        frame = client.call(api_name, page_params, fields)
        if frame.empty:
            break
        parts.append(frame)
        if len(frame) < page_size:
            break
    if not parts:
        return pd.DataFrame(columns=list(fields))
    out = pd.concat(parts, ignore_index=True)
    if "ts_code" in out.columns:
        out = out.drop_duplicates("ts_code", keep="last")
    return out.reset_index(drop=True)


def _confirmed_limit_up_list(
    daily: pd.DataFrame,
    limits: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    if not detail.empty:
        out = detail.copy()
        if "limit" in out.columns:
            limit_type = out["limit"].astype(str).str.upper().str.strip()
            out = out[limit_type.isin({"U", "UP", "涨停", ""})]
            out = out.rename(columns={"limit": "limit_type"})
        if "limit_type" not in out.columns:
            out["limit_type"] = "U"
    else:
        out = daily.merge(
            limits[["ts_code", "up_limit"]],
            on="ts_code",
            how="inner",
        )
        close = pd.to_numeric(out["close"], errors="coerce")
        up_limit = pd.to_numeric(out["up_limit"], errors="coerce")
        out = out[(close - up_limit).abs().le(0.011)].copy()
        out["limit_type"] = "U"
        out["name"] = ""
        out["industry"] = ""
        out["open_times"] = pd.NA
        out["fd_amount"] = pd.NA
    if out.empty:
        return out
    sort_columns = [
        column
        for column in ("limit_times", "fd_amount", "amount", "ts_code")
        if column in out.columns
    ]
    ascending = [
        column == "ts_code"
        for column in sort_columns
    ]
    if sort_columns:
        out = out.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
            kind="stable",
        )
    out["rank"] = range(1, len(out) + 1)
    return out.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def _covered_dates(history_root: Path) -> set[str]:
    covered: set[str] = set()
    for path in history_root.glob("training_*.csv"):
        frame = _read_csv(path)
        if "signal_date" in frame.columns:
            covered.update(
                value
                for value in frame["signal_date"].map(_date)
                if value
            )
    return covered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact, strict-calendar Decision V8 history from "
            "Tushare daily and opening-auction truth"
        )
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument(
        "--end-date",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"),
    )
    parser.add_argument("--max-missing-dates", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--request-sleep", type=float, default=0.08)
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Commit successful dates and exit zero when an endpoint is unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    start_date = _date(args.start_date)
    end_date = _date(args.end_date)
    if not start_date or not end_date or start_date > end_date:
        raise ValueError("start/end date must be valid YYYYMMDD")
    if not str(os.environ.get("TUSHARE_TOKEN", "") or "").strip():
        if args.optional:
            print("[decision-v8-backfill] TUSHARE_TOKEN unavailable; skipped")
            return 0
        raise RuntimeError("TUSHARE_TOKEN is not configured")

    client = TushareClient.from_env(
        timeout_seconds=max(1, int(args.timeout_seconds))
    )
    calendar = client.trade_calendar(start_date, end_date)
    write_calendar(calendar, root)
    open_dates = calendar.loc[
        calendar["is_open"].eq(1), "cal_date"
    ].astype(str).tolist()
    history_root = root / "data" / "auction_v3" / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    covered = _covered_dates(history_root)
    eligible_targets = open_dates[:-8] if len(open_dates) > 8 else []
    missing = [date for date in eligible_targets if date not in covered]
    target_dates = missing[: max(0, int(args.max_missing_dates))]
    if not target_dates:
        print(
            json.dumps(
                {
                    "status": "up_to_date",
                    "calendar_open_dates": len(open_dates),
                    "covered_signal_dates": len(covered),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    first_index = max(0, open_dates.index(target_dates[0]) - 5)
    last_index = min(
        len(open_dates),
        open_dates.index(target_dates[-1]) + 9,
    )
    fetch_dates = open_dates[first_index:last_index]
    failures: list[dict[str, str]] = []
    endpoint_rows: dict[str, int] = {
        "daily": 0,
        "stk_limit": 0,
        "daily_basic": 0,
        "limit_list_d": 0,
        "stk_auction_o": 0,
    }

    with tempfile.TemporaryDirectory(
        prefix="decision-v8-backfill-"
    ) as temp_name:
        temp_root = Path(temp_name)
        for index, trade_date in enumerate(fetch_dates, start=1):
            market_root = (
                temp_root
                / "data"
                / "market"
                / "raw"
                / trade_date[:4]
                / trade_date
            )
            market_root.mkdir(parents=True, exist_ok=True)
            try:
                daily = _call_paged(
                    client,
                    "daily",
                    {"trade_date": trade_date},
                    DAILY_FIELDS,
                )
                time.sleep(max(0.0, float(args.request_sleep)))
                limits = _call_paged(
                    client,
                    "stk_limit",
                    {"trade_date": trade_date},
                    LIMIT_FIELDS,
                )
                time.sleep(max(0.0, float(args.request_sleep)))
                daily_basic = _call_paged(
                    client,
                    "daily_basic",
                    {"trade_date": trade_date},
                    DAILY_BASIC_FIELDS,
                )
                if "circ_mv" in daily_basic.columns:
                    daily_basic = daily_basic.rename(
                        columns={"circ_mv": "float_mv"}
                    )
                time.sleep(max(0.0, float(args.request_sleep)))
                try:
                    detail = _call_paged(
                        client,
                        "limit_list_d",
                        {"trade_date": trade_date},
                        LIMIT_LIST_FIELDS,
                    )
                except Exception as exc:
                    detail = pd.DataFrame()
                    failures.append(
                        {
                            "trade_date": trade_date,
                            "endpoint": "limit_list_d",
                            "reason": type(exc).__name__,
                        }
                    )
                detail = _confirmed_limit_up_list(daily, limits, detail)
                time.sleep(max(0.0, float(args.request_sleep)))
                auction = client.call(
                    "stk_auction_o",
                    {"trade_date": trade_date},
                    AUCTION_OPEN_FIELDS,
                )
                if daily.empty or limits.empty:
                    raise RuntimeError("daily or stk_limit returned no rows")
                _write_csv(daily, market_root / "daily.csv")
                _write_csv(limits, market_root / "stk_limit.csv")
                _write_csv(daily_basic, market_root / "daily_basic.csv")
                _write_csv(detail, market_root / "limit_list_d.csv")
                _write_csv(auction, market_root / "stk_auction_o.csv")
                endpoint_rows["daily"] += len(daily)
                endpoint_rows["stk_limit"] += len(limits)
                endpoint_rows["daily_basic"] += len(daily_basic)
                endpoint_rows["limit_list_d"] += len(detail)
                endpoint_rows["stk_auction_o"] += len(auction)
                if index == 1 or index % 10 == 0 or index == len(fetch_dates):
                    print(
                        "[decision-v8-backfill] "
                        f"fetched {index}/{len(fetch_dates)} dependency dates; "
                        f"current={trade_date}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "trade_date": trade_date,
                        "endpoint": "core",
                        "reason": type(exc).__name__,
                    }
                )
                if not args.optional:
                    raise

        temp_calendar = temp_root / "data" / "market" / "trade_cal_sse.csv"
        temp_calendar.parent.mkdir(parents=True, exist_ok=True)
        calendar.to_csv(temp_calendar, index=False, encoding="utf-8-sig")
        history = AuctionV3Engine(
            AuctionV3Config(
                root=temp_root,
                min_train_dates=2,
                min_train_rows=10,
            )
        ).build_history()
        if not history.empty:
            history = history[
                history["signal_date"].astype(str).isin(target_dates)
            ].copy()

    if history.empty:
        summary = {
            "status": "no_training_rows",
            "target_dates": target_dates,
            "fetch_dates": fetch_dates,
            "failures": failures,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if args.optional else 1

    history["history_source"] = "tushare_compact_backfill"
    history["history_contract_version"] = (
        "decision_v8_strict_calendar_no_future"
    )
    history["backfill_generated_at_utc"] = (
        datetime.now(ZoneInfo("UTC"))
        .replace(microsecond=0)
        .isoformat()
    )
    history = history.sort_values(
        ["signal_date", "source_rank", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)
    output_path = history_root / (
        f"training_{target_dates[0]}_{target_dates[-1]}.csv"
    )
    if output_path.exists():
        existing = _read_csv(output_path)
        if _sha256_frame(existing) != _sha256_frame(history):
            raise RuntimeError(
                f"immutable compact history conflict: {output_path}"
            )
    else:
        _write_csv(history, output_path)

    all_covered = _covered_dates(history_root)
    official_auction_rows = int(
        history.get(
            "auction_truth_source",
            pd.Series(dtype=str),
        ).eq("tushare_stk_auction_o").sum()
    )
    manifest = {
        "schema_version": "decision_v8_history_manifest_v1",
        "generated_at_utc": datetime.now(ZoneInfo("UTC"))
        .replace(microsecond=0)
        .isoformat(),
        "calendar_source": "tushare:trade_cal:SSE",
        "strict_calendar": True,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "target_signal_dates": target_dates,
        "target_signal_date_count": len(target_dates),
        "produced_signal_dates": int(
            history["signal_date"].astype(str).nunique()
        ),
        "produced_rows": int(len(history)),
        "official_auction_truth_rows": official_auction_rows,
        "auction_truth_coverage": (
            official_auction_rows / len(history)
            if len(history)
            else 0.0
        ),
        "total_compact_signal_dates": len(all_covered),
        "target_independent_dates": 500,
        "output_file": str(output_path.relative_to(root)),
        "output_sha256": _sha256_frame(history),
        "endpoint_rows": endpoint_rows,
        "failures": failures,
        "credential_persisted": False,
    }
    (history_root / "manifest_latest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
