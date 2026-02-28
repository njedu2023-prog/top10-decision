# -*- coding: utf-8 -*-

"""
Ingest - 预测输入层（Top10-Decision）【收敛版 / 新链路唯一入口】

职责（单一）：
- 只读取本仓库预测快照：data/pred/pred_source_latest.csv
- 允许用 TOP10_PRED_PATH / pred_path 显式覆盖（用于回放/应急/测试）
- 做最小字段契约校验 + 标准化，向下游提供稳定 DataFrame

注意：
- 本模块不再兼容旧链路文件名（pred_top10_latest.csv 等），避免技术债膨胀。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_PRED_PATH = Path("data/pred/pred_source_latest.csv")


def _log(msg: str) -> None:
    print(f"[ingest] {msg}")


def _warn(msg: str) -> None:
    print(f"[ingest][WARN] {msg}")


def _normalize_yyyymmdd(x) -> str:
    """
    允许输入：20260227 / '20260227' / '2026-02-27' / Timestamp 等
    输出：'YYYYMMDD'；无法识别则返回空字符串
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.isdigit() and len(s) == 8:
        return s
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y%m%d")
    except Exception:
        return ""


def _resolve_pred_path(pred_path: Optional[str] = None) -> Path:
    """
    路径优先级：
    1) 显式参数 pred_path
    2) 环境变量 TOP10_PRED_PATH
    3) 默认 DEFAULT_PRED_PATH（data/pred/pred_source_latest.csv）
    """
    if pred_path and str(pred_path).strip():
        return Path(str(pred_path).strip())

    env_path = os.getenv("TOP10_PRED_PATH", "").strip()
    if env_path:
        return Path(env_path)

    return DEFAULT_PRED_PATH


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化输出列：尽量“可运行”，但不做旧链路兼容。
    最低要求：ts_code
    强烈建议：trade_date, name
    核心数值列：prob / StrengthScore / ThemeBoost 若缺则填 0.0
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 必需：ts_code
    if "ts_code" not in df.columns:
        raise ValueError("预测源文件必须包含列：ts_code")

    # 建议：trade_date / name
    if "trade_date" not in df.columns:
        _warn("预测源文件缺少 trade_date，将填空字符串（建议上游补齐 YYYYMMDD）。")
        df["trade_date"] = ""
    if "name" not in df.columns:
        _warn("预测源文件缺少 name，将填空字符串（建议上游补齐）。")
        df["name"] = ""

    # 可选：verify_date
    if "verify_date" not in df.columns:
        df["verify_date"] = ""

    # 核心分数列：缺失填 0
    if "prob" not in df.columns:
        _warn("预测源文件缺少 prob，将填 0.0（建议上游补齐）。")
        df["prob"] = 0.0
    if "StrengthScore" not in df.columns:
        _warn("预测源文件缺少 StrengthScore，将填 0.0（建议上游补齐）。")
        df["StrengthScore"] = 0.0
    if "ThemeBoost" not in df.columns:
        _warn("预测源文件缺少 ThemeBoost，将填 0.0（建议上游补齐）。")
        df["ThemeBoost"] = 0.0

    # 日期规范化
    df["trade_date"] = df["trade_date"].apply(_normalize_yyyymmdd)
    df["verify_date"] = df["verify_date"].apply(_normalize_yyyymmdd)

    # 数值列规范化
    for col in ["prob", "StrengthScore", "ThemeBoost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ts_code 规范化
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    return df


def load_latest_pred(pred_path: Optional[str] = None) -> pd.DataFrame:
    """
    读取预测快照（默认：data/pred/pred_source_latest.csv）

    参数：
    - pred_path: 可显式指定文件路径（优先级最高）
    - 环境变量 TOP10_PRED_PATH 也可覆盖

    返回：
    - 标准化后的 DataFrame
    """
    path = _resolve_pred_path(pred_path=pred_path)

    if not path.exists():
        raise FileNotFoundError(
            "缺少预测源文件：\n"
            f"- tried: {path}\n\n"
            "请确认：\n"
            f"1) 默认文件是否存在：{DEFAULT_PRED_PATH}\n"
            "2) 或设置环境变量 TOP10_PRED_PATH 指向有效 CSV\n"
            "3) 或调用 load_latest_pred(pred_path=...) 显式传入路径\n"
        )

    df = pd.read_csv(path)
    df = _ensure_columns(df)

    _log(f"loaded: {path} rows={len(df)} cols={len(df.columns)}")
    return df
