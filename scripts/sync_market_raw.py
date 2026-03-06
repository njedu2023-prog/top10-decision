#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sync_market_raw.py

目标：
- 从 a-share-top3-data 仓库同步 market 原始多源文件
- 按日期分目录落到本仓库：
    data/market/raw/{YYYY}/{YYYYMMDD}/{filename}
- 同时维护 latest 镜像目录：
    data/market/raw/latest/{filename}
- 记录同步审计：
    data/market/raw/{YYYY}/{YYYYMMDD}/_sync_meta.json
    data/market/raw/latest/_sync_meta.json

已确认的上游主路径结构：
- data/raw/{YYYY}/{YYYYMMDD}/{filename}

例如：
- data/raw/2026/20260306/daily.csv

职责边界：
- 本脚本只负责“同步 raw 原料”
- 不负责 FS 构建
- 不负责 Decision / Premium 计算
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


RAW_DIR = Path("data/market/raw")
LATEST_DIR_NAME = "latest"
TIMEOUT = 20


@dataclass(frozen=True)
class SourceSpec:
    local_stem: str
    upstream_name: str
    required: bool = False


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec("daily", "daily.csv", required=True),
    SourceSpec("daily_basic", "daily_basic.csv", required=True),
    SourceSpec("hot_boards", "hot_boards.csv", required=False),
    SourceSpec("limit_break_d", "limit_break_d.csv", required=False),
    SourceSpec("limit_list_d", "limit_list_d.csv", required=False),
    SourceSpec("limit_up_tags", "limit_up_tags.csv", required=False),
    SourceSpec("moneyflow_hsgt", "moneyflow_hsgt.csv", required=False),
    SourceSpec("namechange", "namechange.csv", required=False),
    SourceSpec("stk_limit", "stk_limit.csv", required=True),
    SourceSpec("stock_basic", "stock_basic.csv", required=True),
    SourceSpec("top_list", "top_list.csv", required=False),
]

UPSTREAM_META_NAME = "_meta.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _norm_trade_date(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    if re.fullmatch(r"\d{8}", s):
        return s
    return None


def _trade_year(trade_date: str) -> str:
    return trade_date[:4]


def _base_raw_url(owner: str, repo: str, branch: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"


def _build_fallback_relpaths(filename: str, trade_date: str | None) -> list[str]:
    """
    主路径已确认，但仍保留少量兜底，便于上游结构微调时不至于完全失效。
    """
    paths: list[str] = []

    if trade_date:
        year = _trade_year(trade_date)
        paths.extend([
            f"data/raw/{year}/{trade_date}/{filename}",
            f"data/raw/{trade_date}/{filename}",
            f"data/{trade_date}/{filename}",
            f"{trade_date}/{filename}",
        ])

    paths.extend([
        f"data/raw/latest/{filename}",
        f"data/latest/{filename}",
        f"data/{filename}",
        filename,
    ])

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _build_candidate_urls(owner: str, repo: str, branch: str, filename: str, trade_date: str | None) -> list[str]:
    base = _base_raw_url(owner, repo, branch)
    rels = _build_fallback_relpaths(filename, trade_date)
    return [f"{base}/{rel}" for rel in rels]


def _http_get_text(url: str, token: str | None = None) -> tuple[bool, str, int]:
    headers = {
        "User-Agent": "top10-decision-sync-market-raw/1.1",
        "Accept": "text/plain,application/json;q=0.9,*/*;q=0.8",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            resp.encoding = resp.encoding or "utf-8"
            return True, resp.text, resp.status_code
        return False, "", resp.status_code
    except Exception:
        return False, "", 0


def _fetch_first_available(urls: list[str], token: str | None = None) -> tuple[str | None, str | None, int | None]:
    last_code: int | None = None
    for url in urls:
        ok, text, code = _http_get_text(url, token=token)
        last_code = code
        if ok and text:
            return url, text, code
    return None, None, last_code


def _infer_trade_date_from_csv_text(text: str) -> str | None:
    if not text:
        return None

    try:
        reader = csv.DictReader(io.StringIO(text))
        first_row = next(reader, None)
        if not first_row:
            return None

        for key in ("trade_date", "date"):
            if key in first_row:
                td = _norm_trade_date(first_row.get(key))
                if td:
                    return td
    except Exception:
        return None

    return None


def _infer_trade_date_from_meta(meta: dict[str, Any]) -> str | None:
    if not meta:
        return None

    for key in ("trade_date", "asof_date", "snapshot_date", "date"):
        td = _norm_trade_date(meta.get(key))
        if td:
            return td

    nested = meta.get("meta")
    if isinstance(nested, dict):
        for key in ("trade_date", "asof_date", "snapshot_date", "date"):
            td = _norm_trade_date(nested.get(key))
            if td:
                return td

    return None


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_upstream_meta(owner: str, repo: str, branch: str, trade_date: str | None, token: str | None) -> tuple[dict[str, Any], str | None]:
    urls = _build_candidate_urls(owner, repo, branch, UPSTREAM_META_NAME, trade_date)
    hit_url, text, _ = _fetch_first_available(urls, token=token)
    if not hit_url or not text:
        return {}, None

    try:
        return json.loads(text), hit_url
    except Exception:
        return {}, hit_url


def _dated_dir(trade_date: str) -> Path:
    return RAW_DIR / _trade_year(trade_date) / trade_date


def _latest_dir() -> Path:
    return RAW_DIR / LATEST_DIR_NAME


def _build_dated_path(upstream_name: str, trade_date: str) -> Path:
    return _dated_dir(trade_date) / upstream_name


def _build_latest_path(upstream_name: str) -> Path:
    return _latest_dir() / upstream_name


def _build_meta_dated_path(trade_date: str) -> Path:
    return _dated_dir(trade_date) / "_sync_meta.json"


def _build_meta_latest_path() -> Path:
    return _latest_dir() / "_sync_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步 a-share-top3-data 的 market raw 多源文件")
    parser.add_argument("--trade-date", dest="trade_date", default=None, help="交易日 YYYYMMDD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    trade_date = _norm_trade_date(args.trade_date or os.getenv("TRADE_DATE"))
    owner = os.getenv("MARKET_RAW_OWNER", "njedu2023-prog")
    repo = os.getenv("MARKET_RAW_REPO", "a-share-top3-data")
    branch = os.getenv("MARKET_RAW_BRANCH", "main")
    github_token = os.getenv("GITHUB_TOKEN", "").strip() or None

    _ensure_dir(RAW_DIR)

    upstream_meta, upstream_meta_url = _load_upstream_meta(
        owner=owner,
        repo=repo,
        branch=branch,
        trade_date=trade_date,
        token=github_token,
    )

    resolved_trade_date = trade_date or _infer_trade_date_from_meta(upstream_meta)

    results: list[dict[str, Any]] = []
    required_failures: list[str] = []

    daily_cache_text: str | None = None
    daily_cache_url: str | None = None

    for spec in SOURCE_SPECS:
        urls = _build_candidate_urls(
            owner=owner,
            repo=repo,
            branch=branch,
            filename=spec.upstream_name,
            trade_date=trade_date,
        )

        hit_url: str | None = None
        text: str | None = None
        last_code: int | None = None

        if spec.local_stem == "daily" and daily_cache_text is not None:
            hit_url = daily_cache_url
            text = daily_cache_text
            last_code = 200
        else:
            hit_url, text, last_code = _fetch_first_available(urls, token=github_token)
            if spec.local_stem == "daily" and hit_url and text:
                daily_cache_url = hit_url
                daily_cache_text = text

        if hit_url and text:
            if resolved_trade_date is None and spec.local_stem == "daily":
                resolved_trade_date = _infer_trade_date_from_csv_text(text)

            results.append({
                "name": spec.local_stem,
                "upstream_name": spec.upstream_name,
                "required": spec.required,
                "success": True,
                "source_url": hit_url,
                "status_code": last_code,
                "error": "",
            })
        else:
            results.append({
                "name": spec.local_stem,
                "upstream_name": spec.upstream_name,
                "required": spec.required,
                "success": False,
                "source_url": "",
                "status_code": last_code,
                "error": "not_found_in_candidate_urls",
            })
            if spec.required:
                required_failures.append(spec.local_stem)

    if resolved_trade_date is None:
        print("[sync_market_raw] ERROR: 无法解析 trade_date（既未显式传入，也无法从上游 meta/daily 推断）")
        return 2

    write_failures: list[str] = []
    enriched_results: list[dict[str, Any]] = []

    target_dated_dir = _dated_dir(resolved_trade_date)
    target_latest_dir = _latest_dir()
    _ensure_dir(target_dated_dir)
    _ensure_dir(target_latest_dir)

    for item in results:
        spec = next(s for s in SOURCE_SPECS if s.local_stem == item["name"])

        if not item["success"]:
            enriched_results.append(item)
            continue

        hit_url = item["source_url"]
        ok, text, code = _http_get_text(hit_url, token=github_token)
        if not ok or not text:
            item["success"] = False
            item["status_code"] = code
            item["error"] = "download_failed_on_write_phase"
            if spec.required:
                required_failures.append(spec.local_stem)
            enriched_results.append(item)
            continue

        dated_path = _build_dated_path(spec.upstream_name, resolved_trade_date)
        latest_path = _build_latest_path(spec.upstream_name)

        try:
            _write_text(dated_path, text)
            _write_text(latest_path, text)
            item["dated_path"] = str(dated_path)
            item["latest_path"] = str(latest_path)
            item["bytes"] = len(text.encode("utf-8"))
            enriched_results.append(item)
        except Exception as e:
            item["success"] = False
            item["error"] = f"write_failed:{e}"
            if spec.required:
                required_failures.append(spec.local_stem)
            write_failures.append(spec.local_stem)
            enriched_results.append(item)

    sync_meta = {
        "trade_date": resolved_trade_date,
        "created_at_utc": _now_utc(),
        "source_repo": {
            "owner": owner,
            "repo": repo,
            "branch": branch,
        },
        "requested_trade_date": trade_date,
        "resolved_trade_date": resolved_trade_date,
        "raw_storage_pattern": "data/market/raw/{YYYY}/{YYYYMMDD}/{filename}",
        "raw_latest_pattern": "data/market/raw/latest/{filename}",
        "upstream_primary_pattern": "data/raw/{YYYY}/{YYYYMMDD}/{filename}",
        "upstream_meta_url": upstream_meta_url or "",
        "upstream_meta": upstream_meta,
        "files": enriched_results,
        "required_failures": sorted(set(required_failures)),
        "write_failures": sorted(set(write_failures)),
        "summary": {
            "success_count": sum(1 for x in enriched_results if x.get("success")),
            "failure_count": sum(1 for x in enriched_results if not x.get("success")),
            "required_failure_count": len(set(required_failures)),
        },
    }

    meta_dated_path = _build_meta_dated_path(resolved_trade_date)
    meta_latest_path = _build_meta_latest_path()
    _write_json(meta_dated_path, sync_meta)
    _write_json(meta_latest_path, sync_meta)

    print(f"[sync_market_raw] resolved_trade_date={resolved_trade_date}")
    print(f"[sync_market_raw] source_repo={owner}/{repo}@{branch}")
    print(f"[sync_market_raw] dated_dir={target_dated_dir}")
    print(f"[sync_market_raw] latest_dir={target_latest_dir}")

    for item in enriched_results:
        status = "OK" if item.get("success") else "FAIL"
        print(
            f"[sync_market_raw] {status} {item['name']} "
            f"url={item.get('source_url', '')} "
            f"dated={item.get('dated_path', '')} "
            f"latest={item.get('latest_path', '')}"
        )

    print(f"[sync_market_raw] meta_dated={meta_dated_path}")
    print(f"[sync_market_raw] meta_latest={meta_latest_path}")

    if required_failures:
        print(f"[sync_market_raw] ERROR: required files missing -> {sorted(set(required_failures))}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
