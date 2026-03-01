#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — IO 层（读写/落盘/追溯）

本文件职责：
- 统一处理 Premium 的输入读取（decision 预测表、close 真值表）
- 统一处理 Premium 的输出落盘（rank csv/md、eval_history、_last_run.txt）
- 自动创建输出目录（避免目录不存在导致报错）
- 统一追溯字段：run_id / commit_sha / created_at_utc

注意：
- 本模块只处理文件层，不做业务计算。

P1.1 工程收口：
- load_close_table(cfg) 已被 Market Truth Layer 替代（data/market/daily_YYYYMMDD.csv）。
  但为兼容旧代码，暂不删除，仅标记 deprecated。
"""

from __future__ import annotations

import glob
import os
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .config import PremiumConfig
from .schemas import PremiumEvalHistorySchema, PremiumRankOutputSchema


# =========================
# 1) 追溯信息
# =========================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_commit_sha(repo_root: Path) -> str:
    """
    尝试从 git 获取 commit sha；失败则返回 'unknown'。
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12] if out else "unknown"
    except Exception:
        return "unknown"


def get_run_id() -> str:
    """
    GitHub Actions 下优先用 GITHUB_RUN_ID / GITHUB_RUN_NUMBER；
    否则用时间戳生成一个可读 run_id。
    """
    rid = os.getenv("GITHUB_RUN_ID", "").strip()
    rno = os.getenv("GITHUB_RUN_NUMBER", "").strip()
    if rid:
        return f"gh_{rid}"
    if rno:
        return f"ghno_{rno}"
    return datetime.now(timezone.utc).strftime("local_%Y%m%d%H%M%S")


# =========================
# 2) 输出目录准备
# =========================

def ensure_output_dirs(cfg: PremiumConfig) -> None:
    cfg.out_root().mkdir(parents=True, exist_ok=True)
    cfg.out_rank_dir().mkdir(parents=True, exist_ok=True)
    cfg.out_models_dir().mkdir(parents=True, exist_ok=True)
    cfg.out_learning_dir().mkdir(parents=True, exist_ok=True)


# =========================
# 3) 输入读取
# =========================

@dataclass(frozen=True)
class DecisionInputFile:
    path: Path
    df: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    # 常见：utf-8 / utf-8-sig / gbk
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 最后兜底
    return pd.read_csv(path)


def _extract_trade_date_from_df(df: pd.DataFrame) -> Optional[str]:
    """
    尝试从 df['trade_date'] 取出唯一值。
    不强制依赖 schemas 的 alias（因为此处只是帮助排序选择文件）。
    """
    for col in df.columns:
        if str(col).strip().lower() in ("trade_date", "date", "dt", "交易日期", "日期"):
            s = df[col].dropna().astype(str).unique()
            if len(s) == 1:
                return s[0].strip()
            # 多值也可能存在（异常），返回最大值作为兜底
            if len(s) > 1:
                return sorted([x.strip() for x in s])[-1]
    return None


def load_decision_inputs(cfg: PremiumConfig) -> List[DecisionInputFile]:
    """
    读取 decision 输入表（第2日预测表）。
    返回按 trade_date 升序排序的列表（如果取不到 trade_date，则按文件名排序）。
    """
    repo_root = cfg.repo_root()
    pattern = str((repo_root / cfg.decision_input_glob).resolve())
    paths = [Path(p).resolve() for p in glob.glob(pattern)]
    files: List[DecisionInputFile] = []

    for p in sorted(paths):
        try:
            df = _read_csv(p)
            files.append(DecisionInputFile(path=p, df=df))
        except Exception:
            # 读失败就跳过，避免一个坏文件拖死整个流程
            continue

    # 尝试按 df 内的 trade_date 排序
    def sort_key(item: DecisionInputFile) -> Tuple[int, str]:
        td = _extract_trade_date_from_df(item.df)
        if td and td.isdigit() and len(td) == 8:
            return (0, td)
        return (1, item.path.name)

    return sorted(files, key=sort_key)


def load_close_table(cfg: PremiumConfig) -> pd.DataFrame:
    """
    DEPRECATED（P1.1）：
    - 旧链路：读取 data/close/*.csv（或 cfg.close_input_glob 指定的真值表）
    - 新链路：请使用 Market Truth Layer（src/top10decision/premium/market_truth.py）
      统一从 data/market/daily_YYYYMMDD.csv 获取真值（缺则自动拉取并缓存）。

    说明：
    - 本函数暂时保留仅为兼容旧代码/旧脚本；
    - 后续将统一移除。
    """
    warnings.warn(
        "load_close_table(cfg) is deprecated. Use Market Truth Layer (market_truth.py) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    repo_root = cfg.repo_root()
    pattern = str((repo_root / cfg.close_input_glob).resolve())
    paths = [Path(p).resolve() for p in glob.glob(pattern)]
    if not paths:
        return pd.DataFrame()

    dfs = []
    for p in sorted(paths):
        try:
            df = _read_csv(p)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    # 去重（若存在）
    for c in out.columns:
        if str(c).strip().lower() in ("trade_date", "date", "dt"):
            out[c] = out[c].astype(str).str.strip()
    for c in out.columns:
        if str(c).strip().lower() in ("ts_code", "code", "symbol", "ticker"):
            out[c] = out[c].astype(str).str.strip()
    return out


# =========================
# 4) 输出落盘（rank / md）
# =========================

def _ensure_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
    """
    保证 df 至少包含 columns 中的所有列，不存在则补 NaN，并按 columns 重排。
    """
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df.loc[:, list(columns)]


def write_rank_csv(cfg: PremiumConfig, trade_date: str, df_rank: pd.DataFrame) -> Path:
    ensure_output_dirs(cfg)

    p = cfg.rank_csv_path(trade_date)
    df_out = _ensure_columns(df_rank.copy(), PremiumRankOutputSchema.COLUMNS)

    # 统一写入
    df_out.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_rank_md(cfg: PremiumConfig, trade_date: str, md_text: str) -> Path:
    ensure_output_dirs(cfg)

    p = cfg.rank_md_path(trade_date)
    p.write_text(md_text, encoding="utf-8")
    return p


# =========================
# 5) learning 目录落库（eval_history + last_run）
# =========================

def append_eval_history(cfg: PremiumConfig, row: dict) -> Path:
    """
    向 learning/premium_eval_history.csv 追加一行。
    若文件不存在，则创建并写 header。
    """
    ensure_output_dirs(cfg)

    p = cfg.eval_history_path()
    cols = list(PremiumEvalHistorySchema.COLUMNS)

    # 补齐列
    for c in cols:
        row.setdefault(c, pd.NA)

    df_row = pd.DataFrame([row])[cols]
    if p.exists():
        try:
            df_old = _read_csv(p)
            df_new = pd.concat([df_old, df_row], ignore_index=True)
        except Exception:
            df_new = df_row
    else:
        df_new = df_row

    df_new.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_last_run(cfg: PremiumConfig, trade_date: str, extra: Optional[dict] = None) -> Path:
    """
    覆盖写 outputs/premium/_last_run.txt
    内容：trade_date / run_id / commit_sha / created_at_utc + 可选 extra
    """
    ensure_output_dirs(cfg)

    repo_root = cfg.repo_root()
    run_id = get_run_id()
    sha = get_commit_sha(repo_root)
    ts = utc_now_iso()

    lines = [
        f"trade_date: {trade_date}",
        f"run_id: {run_id}",
        f"commit_sha: {sha}",
        f"model_version: {cfg.model_version}",
        f"created_at_utc: {ts}",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

    p = cfg.out_last_run_path()
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "utc_now_iso",
    "get_commit_sha",
    "get_run_id",
    "ensure_output_dirs",
    "load_decision_inputs",
    "load_close_table",
    "write_rank_csv",
    "write_rank_md",
    "append_eval_history",
    "write_last_run",
]
