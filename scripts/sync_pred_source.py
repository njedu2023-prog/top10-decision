#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sync_pred_source.py

职责（硬规则）：
- 跨仓库拉取必须独立：sync 不能混在 runner
- 将外部/本地预测源写入固定快照：data/pred/pred_source_latest.csv
- 在可识别 trade_date 时，同时落历史归档：
  data/pred/archive/pred_source_{trade_date}.csv
- 不做任何业务计算/字段适配（适配在 adapters）

环境变量：
- TOP10_PRED_URL   : 远端 CSV（GitHub Raw 等）
- TOP10_PRED_PATH  : 本地 CSV 路径（调试用）
- TRADE_DATE       : 指定交易日 YYYYMMDD（优先级最高，可选）

输出（IO 契约输入快照，latest 绝对不改动）：
- data/pred/pred_source_latest.csv
- data/pred/archive/pred_source_{trade_date}.csv  （若能识别出 trade_date）
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import urllib.request
import csv
import io
from pathlib import Path


SNAPSHOT_PATH = Path("data/pred/pred_source_latest.csv")
ARCHIVE_DIR = Path("data/pred/archive")


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "top10-decision-sync"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _read_local_bytes(src: Path) -> bytes:
    return src.read_bytes()


def _write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _extract_trade_date(text: str) -> str:
    """
    从字符串中提取 8 位日期 YYYYMMDD。
    仅做弱推断，不校验是否为真实交易日。
    """
    if not text:
        return ""

    m = re.search(r"(?<!\d)(20\d{6})(?!\d)", text)
    return m.group(1) if m else ""


def _extract_trade_date_from_csv_bytes(data: bytes) -> str:
    """
    从 CSV 内容中解析实际 trade_date。
    latest URL 通常不带日期，必须以内文日期归档，否则历史学习样本会缺少
    pred_source_{trade_date}.csv，甚至误用 latest。
    """
    text = ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return ""

    try:
        reader = csv.DictReader(io.StringIO(text))
    except Exception:
        return ""

    date_cols = (
        "trade_date",
        "signal_date",
        "date",
        "verify_date",
        "target_trade_date",
    )
    counts: dict[str, int] = {}
    for i, row in enumerate(reader):
        if i >= 500:
            break
        for col in date_cols:
            val = row.get(col)
            td = _extract_trade_date(str(val or ""))
            if td:
                counts[td] = counts.get(td, 0) + 1
                break

    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _resolve_trade_date(url: str, path: str, data: bytes | None = None) -> str:
    """
    trade_date 解析优先级：
    1) 环境变量 TRADE_DATE
    2) 从 TOP10_PRED_URL 中提取
    3) 从 TOP10_PRED_PATH 中提取
    4) 从 CSV 内容 trade_date/signal_date 等字段中提取
    """
    env_trade_date = (os.getenv("TRADE_DATE") or "").strip()
    if env_trade_date:
        if re.fullmatch(r"20\d{6}", env_trade_date):
            return env_trade_date
        print(
            f"[SYNC][WARN] TRADE_DATE 格式非法，期望 YYYYMMDD，实际={env_trade_date}；将继续尝试自动提取。",
            file=sys.stderr,
        )

    from_url = _extract_trade_date(url)
    if from_url:
        return from_url

    from_path = _extract_trade_date(path)
    if from_path:
        return from_path

    if data:
        from_csv = _extract_trade_date_from_csv_bytes(data)
        if from_csv:
            return from_csv

    return ""


def _write_archive_if_possible(data: bytes, trade_date: str) -> None:
    if not trade_date:
        print("[SYNC][WARN] 未识别到 trade_date；本次仅更新 latest，不落 archive。")
        return

    archive_path = ARCHIVE_DIR / f"pred_source_{trade_date}.csv"
    _write_bytes(archive_path, data)
    print(f"[SYNC] wrote archive  -> {archive_path}")


def main() -> int:
    url = (os.getenv("TOP10_PRED_URL") or "").strip()
    path = (os.getenv("TOP10_PRED_PATH") or "").strip()

    if not url and not path:
        print("[SYNC][ERR] 未提供 TOP10_PRED_URL / TOP10_PRED_PATH，无法同步预测源。", file=sys.stderr)
        return 2

    if url:
        print(f"[SYNC] use TOP10_PRED_URL={url}")
        data = _download_bytes(url)
        trade_date = _resolve_trade_date(url=url, path=path, data=data)
        if trade_date:
            print(f"[SYNC] resolved trade_date={trade_date}")
        else:
            print("[SYNC][WARN] trade_date unresolved")
        _write_bytes(SNAPSHOT_PATH, data)
        print(f"[SYNC] wrote snapshot -> {SNAPSHOT_PATH}")
        _write_archive_if_possible(data, trade_date)
        return 0

    p = Path(path)
    if not p.exists():
        print(f"[SYNC][ERR] TOP10_PRED_PATH 不存在：{p}", file=sys.stderr)
        return 2

    print(f"[SYNC] use TOP10_PRED_PATH={p}")
    data = _read_local_bytes(p)
    trade_date = _resolve_trade_date(url=url, path=path, data=data)
    if trade_date:
        print(f"[SYNC] resolved trade_date={trade_date}")
    else:
        print("[SYNC][WARN] trade_date unresolved")
    _write_bytes(SNAPSHOT_PATH, data)
    print(f"[SYNC] wrote snapshot -> {SNAPSHOT_PATH}")
    _write_archive_if_possible(data, trade_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
