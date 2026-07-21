from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


API_URL = "https://api.tushare.pro"
MINUTE_FIELDS = ("time", "open", "close", "high", "low", "vol", "amount")


def normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        return f"{left.zfill(6)}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 6:
        return ""
    symbol = digits[-6:]
    suffix = "BJ" if symbol.startswith(("4", "8", "920")) else "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
    return f"{symbol}.{suffix}"


def minute_output_path(root: Path, trade_date: str, ts_code: str) -> Path:
    safe_code = normalize_code(ts_code).replace(".", "_")
    return root / "data" / "market" / "minute_1m" / trade_date[:4] / trade_date / f"{safe_code}.csv"


def read_minute_snapshot(root: Path, trade_date: str, ts_code: str) -> pd.DataFrame:
    """Read a persisted minute snapshot without requiring a credential."""
    path = minute_output_path(root, trade_date, ts_code)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if frame.empty or "time" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["time"] = out["time"].astype(str).str.strip()
    for column in ("open", "close", "high", "low", "vol", "amount"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("time").reset_index(drop=True)


def opening_auction_price_from_snapshot(frame: pd.DataFrame) -> float:
    """Return the first regular-session bar open, a proxy for the auction print."""
    if frame.empty or "time" not in frame.columns:
        return float("nan")
    out = frame.copy()
    clock = out["time"].astype(str).str.extract(r"(\d{2}:\d{2}:\d{2})", expand=False)
    regular = out[clock.ge("09:30:00")]
    if regular.empty:
        return float("nan")
    first = regular.iloc[0]
    for column in ("open", "close"):
        try:
            value = float(first.get(column))
        except Exception:
            continue
        if pd.notna(value) and value > 0:
            return value
    return float("nan")


def opening_auction_price(root: Path, trade_date: str, ts_code: str) -> float:
    return opening_auction_price_from_snapshot(read_minute_snapshot(root, trade_date, ts_code))


@dataclass(frozen=True)
class TushareClient:
    token: str
    timeout_seconds: int = 30

    @classmethod
    def from_env(
        cls,
        env_name: str = "TUSHARE_TOKEN",
        *,
        timeout_seconds: int = 8,
    ) -> "TushareClient":
        token = str(os.environ.get(env_name, "") or "").strip()
        if not token:
            raise RuntimeError(f"{env_name} is not configured")
        return cls(token=token, timeout_seconds=max(1, int(timeout_seconds)))

    def call(self, api_name: str, params: dict[str, Any], fields: Iterable[str]) -> pd.DataFrame:
        response = requests.post(
            API_URL,
            json={
                "api_name": api_name,
                "token": self.token,
                "params": params,
                "fields": ",".join(fields),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        code = int(payload.get("code", -1) or 0)
        if code != 0:
            message = str(payload.get("msg") or "Tushare request failed")
            raise RuntimeError(f"Tushare {api_name} error code={code}: {message}")
        data = payload.get("data") or {}
        columns = list(data.get("fields") or [])
        rows = list(data.get("items") or [])
        return pd.DataFrame(rows, columns=columns)

    def trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        frame = self.call(
            "trade_cal",
            {"exchange": "SSE", "start_date": start_date, "end_date": end_date},
            ("exchange", "cal_date", "is_open", "pretrade_date"),
        )
        if frame.empty:
            raise RuntimeError("Tushare trade_cal returned no rows")
        frame = frame.copy()
        frame["cal_date"] = frame["cal_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
        frame["is_open"] = pd.to_numeric(frame["is_open"], errors="coerce").fillna(0).astype(int)
        return frame.sort_values("cal_date").drop_duplicates("cal_date", keep="last")

    def current_minute(self, ts_code: str, freq: str = "1MIN") -> pd.DataFrame:
        code = normalize_code(ts_code)
        if not code:
            raise ValueError(f"Invalid A-share code: {ts_code}")
        frame = self.call("rt_min_daily", {"ts_code": code, "freq": freq}, MINUTE_FIELDS)
        if frame.empty:
            return frame
        frame = frame.copy()
        frame.insert(0, "ts_code", code)
        for column in ("open", "close", "high", "low", "vol", "amount"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["time"] = frame["time"].astype(str).str.strip()
        frame = frame.dropna(subset=["open", "close"]).drop_duplicates("time", keep="last")
        return frame.sort_values("time").reset_index(drop=True)


def write_calendar(frame: pd.DataFrame, root: Path) -> Path:
    path = root / "data" / "market" / "trade_cal_sse.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_minute_snapshot(frame: pd.DataFrame, root: Path, trade_date: str, ts_code: str) -> tuple[Path, Path]:
    path = minute_output_path(root, trade_date, ts_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    meta_path = path.with_suffix(".meta.json")
    meta = {
        "source": "tushare:rt_min_daily",
        "trade_date": trade_date,
        "ts_code": normalize_code(ts_code),
        "frequency": "1MIN",
        "rows": int(len(frame)),
        "fields": list(frame.columns),
        "credential_persisted": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, meta_path


__all__ = [
    "API_URL",
    "MINUTE_FIELDS",
    "TushareClient",
    "minute_output_path",
    "normalize_code",
    "opening_auction_price",
    "opening_auction_price_from_snapshot",
    "read_minute_snapshot",
    "write_calendar",
    "write_minute_snapshot",
]
