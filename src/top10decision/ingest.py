#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest.py

硬规则：
- 数据入口只允许一个：本模块
- runner 不直接读 URL/不直接读其它旧文件
- 本模块只读固定输入快照：data/pred/pred_source_latest.csv

注意：
- 字段兼容/映射只在 adapters 做
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from top10decision.adapters.decisio_adapter import normalize_pred_fields


SNAPSHOT_PATH = Path("data/pred/pred_source_latest.csv")


def _read_csv_any(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def load_pred_snapshot() -> pd.DataFrame:
    """
    唯一入口：读取 data/pred/pred_source_latest.csv
    - 不读取 env url/path
    - 不 fallback 到旧 load_latest_pred
    """
    df = _read_csv_any(SNAPSHOT_PATH)
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_pred_fields(df)
    return df
