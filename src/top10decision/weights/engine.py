#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from top10decision.writers.io_contract import TOPN_DEFAULT


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


def _pick_theme(row: pd.Series) -> str:
    for k in ("theme", "Theme", "board", "industry", "sector"):
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


@dataclass
class WeightCaps:
    w_max: float
    theme_cap: float
    gross_cap: float


def build_weights_with_backups(
    candidates: pd.DataFrame,
    topn: int,
    caps: WeightCaps,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
    - targets：TopN 目标（weight>0, target_rank）
    - backups：候补池（weight=0, backup_rank）
    """
    _ensure_cols(candidates, ["ts_code", "name", "ev_pred"])

    df = candidates.sort_values("ev_pred", ascending=False).reset_index(drop=True).copy()
    df["theme"] = df.apply(_pick_theme, axis=1)

    picked_idx = []
    theme_used: Dict[str, float] = {}
    gross_used = 0.0

    if topn <= 0:
        topn = TOPN_DEFAULT
    base_w = min(caps.gross_cap, 1.0) / float(topn)

    for i in range(len(df)):
        if len(picked_idx) >= topn:
            break

        th = df.loc[i, "theme"] or ""
        w = min(base_w, caps.w_max)

        if th:
            used = theme_used.get(th, 0.0)
            if used + w > caps.theme_cap:
                continue

        if gross_used + w > caps.gross_cap + 1e-9:
            break

        picked_idx.append(i)
        gross_used += w
        if th:
            theme_used[th] = theme_used.get(th, 0.0) + w

    targets = df.loc[picked_idx].copy()
    if targets.empty:
        backups = df.copy()
        return targets, backups

    targets["weight"] = min(base_w, caps.w_max)
    targets["target_rank"] = list(range(1, len(targets) + 1))
    targets["backup_rank"] = ""

    rest = df.drop(index=picked_idx).reset_index(drop=True).copy()
    rest["weight"] = 0.0
    rest["target_rank"] = ""
    rest["backup_rank"] = list(range(1, len(rest) + 1))

    return targets.reset_index(drop=True), rest
