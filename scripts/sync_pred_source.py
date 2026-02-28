#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sync_pred_source.py

职责（硬规则）：
- 跨仓库拉取必须独立：sync 不能混在 runner
- 将外部/本地预测源写入固定快照：data/pred/pred_source_latest.csv
- 不做任何业务计算/字段适配（适配在 adapters）

环境变量：
- TOP10_PRED_URL  : 远端 CSV（GitHub Raw 等）
- TOP10_PRED_PATH : 本地 CSV 路径（调试用）

输出（IO 契约输入快照，绝对不改动）：
- data/pred/pred_source_latest.csv
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path


SNAPSHOT_PATH = Path("data/pred/pred_source_latest.csv")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "top10-decision-sync"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dst.write_bytes(data)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> int:
    url = (os.getenv("TOP10_PRED_URL") or "").strip()
    path = (os.getenv("TOP10_PRED_PATH") or "").strip()

    if not url and not path:
        print("[SYNC][ERR] 未提供 TOP10_PRED_URL / TOP10_PRED_PATH，无法同步预测源。", file=sys.stderr)
        return 2

    if url:
        print(f"[SYNC] use TOP10_PRED_URL={url}")
        _download(url, SNAPSHOT_PATH)
        print(f"[SYNC] wrote snapshot -> {SNAPSHOT_PATH}")
        return 0

    p = Path(path)
    if not p.exists():
        print(f"[SYNC][ERR] TOP10_PRED_PATH 不存在：{p}", file=sys.stderr)
        return 2

    print(f"[SYNC] use TOP10_PRED_PATH={p}")
    _copy(p, SNAPSHOT_PATH)
    print(f"[SYNC] wrote snapshot -> {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
