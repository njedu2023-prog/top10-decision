#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — IO 层（读写/落盘/追溯）

本文件职责：
- 统一处理 Premium 的输入读取（pred_source_latest / decision 预测表 / close 真值表(旧)）
- 统一处理 Premium 的输出落盘（V1: top30/full/verify/md；旧: rank csv/md、eval_history、_last_run.txt）
- 自动创建输出目录（避免目录不存在导致报错）
- 统一追溯字段：run_id / commit_sha / created_at_utc

注意：
- 本模块只处理文件层，不做业务计算。
- 学习模块能力不损失：旧接口全部保留（train/rank/learning 相关）。
"""

from __future__ import annotations

import glob
import os
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """
    同时兼容：
    - 旧口径：outputs/premium/{rank,models,learning}
    - 新口径：outputs/premium/ + docs/reports
    """
    cfg.out_root().mkdir(parents=True, exist_ok=True)

    # 旧口径目录（保留）
    if hasattr(cfg, "out_rank_dir"):
        cfg.out_rank_dir().mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "out_models_dir"):
        cfg.out_models_dir().mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "out_learning_dir"):
        cfg.out_learning_dir().mkdir(parents=True, exist_ok=True)

    # 新口径报告目录
    if hasattr(cfg, "reports_root"):
        cfg.reports_root().mkdir(parents=True, exist_ok=True)


# =========================
# 3) 通用 CSV 读取
# =========================

def _read_csv(path: Path) -> pd.DataFrame:
    # 常见：utf-8 / utf-8-sig / gbk
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _to_yyyymmdd(x: object) -> str:
    s = str(x).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "")
    return s[:8]


# =========================
# 4) 输入读取（旧：decision / close）
# =========================

@dataclass(frozen=True)
class DecisionInputFile:
    path: Path
    df: pd.DataFrame


def _extract_trade_date_from_df(df: pd.DataFrame) -> Optional[str]:
    """
    尝试从 df['trade_date'] 取出唯一值。
    不强制依赖 schemas alias（这里只用于排序选择文件）。
    """
    for col in df.columns:
        if str(col).strip().lower() in ("trade_date", "date", "dt", "交易日期", "日期"):
            s = df[col].dropna().astype(str).unique()
            if len(s) == 1:
                return _to_yyyymmdd(s[0].strip())
            if len(s) > 1:
                return sorted([_to_yyyymmdd(x.strip()) for x in s])[-1]
    return None


def load_decision_inputs(cfg: PremiumConfig) -> List[DecisionInputFile]:
    """
    ♻️ 旧口径：读取 decision 输入表（第2日预测表）。
    返回按 trade_date 升序排序的列表（取不到 trade_date 则按文件名排序）。
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
            continue

    def sort_key(item: DecisionInputFile) -> Tuple[int, str]:
        td = _extract_trade_date_from_df(item.df)
        if td and td.isdigit() and len(td) == 8:
            return (0, td)
        return (1, item.path.name)

    return sorted(files, key=sort_key)


def load_close_table(cfg: PremiumConfig) -> pd.DataFrame:
    """
    DEPRECATED（保留兼容）：
    - 旧链路：读取 cfg.close_input_glob（通常 data/market/daily_*.csv）
    - 新链路：请使用 Market Truth Layer（market_truth.py）
      统一从 data/market/daily_YYYYMMDD.csv 获取真值（缺则自动拉取并缓存）。
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

    # 基础清洗（若存在）
    for c in out.columns:
        if str(c).strip().lower() in ("trade_date", "date", "dt"):
            out[c] = out[c].astype(str).str.strip()
    for c in out.columns:
        if str(c).strip().lower() in ("ts_code", "code", "symbol", "ticker"):
            out[c] = out[c].astype(str).str.strip()

    return out


# =========================
# 5) 输入读取（新：pred_source_latest + decision merge）
# =========================

def load_pred_source_latest(cfg: PremiumConfig) -> pd.DataFrame:
    """
    ✅ 新口径：读取主输入 pred_source_latest（全量候选）。
    """
    ensure_output_dirs(cfg)
    p = cfg.pred_source_latest_path()
    if not p.exists():
        raise FileNotFoundError(f"pred_source_latest not found: {p}")
    df = _read_csv(p)
    if df.empty:
        raise ValueError(f"pred_source_latest is empty: {p}")
    return df


def load_decision_merge(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    """
    ✅ 新口径：读取 decision 产物用于 merge 标签（不得过滤）。
    输出列固定：trade_date/ts_code/name + dec_*（缺失则为空）
    """
    ensure_output_dirs(cfg)
    repo_root = cfg.repo_root()

    pattern = str((repo_root / cfg.decision_glob).resolve())
    paths = [Path(p).resolve() for p in glob.glob(pattern)]
    if not paths:
        return pd.DataFrame()

    trade_date = _to_yyyymmdd(trade_date)

    hit = []
    for p in sorted(paths):
        try:
            df = _read_csv(p)
            if df is None or df.empty:
                continue
            # 尝试统一字段
            cols_map = {str(c).strip().lower(): c for c in df.columns}

            def pick(*names: str) -> Optional[str]:
                for n in names:
                    if n.lower() in cols_map:
                        return cols_map[n.lower()]
                return None

            c_date = pick("trade_date", "date", "dt", "交易日期", "日期")
            c_code = pick("ts_code", "code", "symbol", "ticker", "股票代码", "代码")
            c_name = pick("name", "stock_name", "股票名称", "名称")

            if not c_date or not c_code:
                continue

            d = df.copy()
            d["trade_date"] = d[c_date].astype(str).map(_to_yyyymmdd)
            d["ts_code"] = d[c_code].astype(str).str.strip()
            if c_name:
                d["name"] = d[c_name].astype(str).str.strip()

            if (d["trade_date"].astype(str) == trade_date).any():
                hit.append(d)
        except Exception:
            continue

    if not hit:
        return pd.DataFrame()

    dec = pd.concat(hit, ignore_index=True)
    dec = dec[dec["trade_date"].astype(str) == trade_date].copy()

    cols = {str(c).strip().lower(): c for c in dec.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    out = pd.DataFrame({
        "trade_date": dec["trade_date"].astype(str),
        "ts_code": dec["ts_code"].astype(str).str.strip(),
    })

    if "name" in dec.columns:
        out["name"] = dec["name"].astype(str).str.strip()

    m_rank = pick("dec_rank", "decision_rank", "rank", "决策排名")
    m_w = pick("dec_weight", "weight", "target_weight", "决策权重")
    m_can = pick("dec_can_buy", "can_buy", "可买提示")
    m_pf = pick("dec_p_fill", "p_fill", "P_fill")
    m_reason = pick("dec_reason", "reason", "label", "决策原因", "决策标签")

    out["dec_rank"] = dec[m_rank] if m_rank else pd.NA
    out["dec_weight"] = dec[m_w] if m_w else pd.NA
    out["dec_can_buy"] = dec[m_can] if m_can else pd.NA
    out["dec_p_fill"] = dec[m_pf] if m_pf else pd.NA
    out["dec_reason"] = dec[m_reason] if m_reason else pd.NA

    out = out.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)
    return out


# =========================
# 6) 输出落盘（新：top30/full/verify + 报告）
# =========================

def write_premium_top30(cfg: PremiumConfig, trade_date: str, df_top30: pd.DataFrame) -> Path:
    ensure_output_dirs(cfg)
    p = cfg.out_top30_csv(trade_date)
    df_top30.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_premium_full(cfg: PremiumConfig, trade_date: str, df_full: pd.DataFrame) -> Path:
    ensure_output_dirs(cfg)
    p = cfg.out_full_csv(trade_date)
    df_full.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_premium_verify(cfg: PremiumConfig, trade_date: str, df_verify: pd.DataFrame) -> Path:
    ensure_output_dirs(cfg)
    p = cfg.out_verify_csv(trade_date)
    df_verify.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_report_md(cfg: PremiumConfig, trade_date: str, md_text: str) -> Tuple[Path, Path]:
    """
    写 docs/reports/premium_{trade_date}.md + premium_latest.md
    """
    ensure_output_dirs(cfg)
    p = cfg.report_md_path(trade_date)
    p_latest = cfg.report_latest_md_path()
    p.write_text(md_text, encoding="utf-8")
    p_latest.write_text(md_text, encoding="utf-8")
    return p, p_latest


# =========================
# 7) 输出落盘（旧：rank / md）
# =========================

def _ensure_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df.loc[:, list(columns)]


def write_rank_csv(cfg: PremiumConfig, trade_date: str, df_rank: pd.DataFrame) -> Path:
    """
    ♻️ 旧口径：写 outputs/premium/rank/premium_rank_{trade_date}.csv
    """
    ensure_output_dirs(cfg)
    p = cfg.rank_csv_path(trade_date)
    df_out = _ensure_columns(df_rank.copy(), PremiumRankOutputSchema.COLUMNS)
    df_out.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_rank_md(cfg: PremiumConfig, trade_date: str, md_text: str) -> Path:
    """
    ♻️ 旧口径：写 outputs/premium/rank/premium_rank_{trade_date}.md
    """
    ensure_output_dirs(cfg)
    p = cfg.rank_md_path(trade_date)
    p.write_text(md_text, encoding="utf-8")
    return p


# =========================
# 8) learning 落库（旧：eval_history + last_run）
# =========================

def append_eval_history(cfg: PremiumConfig, row: dict) -> Path:
    """
    ♻️ 旧口径：向 learning/premium_eval_history.csv 追加一行。
    """
    ensure_output_dirs(cfg)
    p = cfg.eval_history_path()
    cols = list(PremiumEvalHistorySchema.COLUMNS)

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
        f"model_version: {getattr(cfg, 'model_version', '-')}",
        f"created_at_utc: {ts}",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

    p = cfg.out_last_run_path()
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    # trace
    "utc_now_iso",
    "get_commit_sha",
    "get_run_id",
    # dirs
    "ensure_output_dirs",
    # inputs (old)
    "load_decision_inputs",
    "load_close_table",
    # inputs (new)
    "load_pred_source_latest",
    "load_decision_merge",
    # outputs (new)
    "write_premium_top30",
    "write_premium_full",
    "write_premium_verify",
    "write_report_md",
    # outputs (old)
    "write_rank_csv",
    "write_rank_md",
    # learning (old)
    "append_eval_history",
    "write_last_run",
]
