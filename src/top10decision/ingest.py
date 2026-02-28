# -*- coding: utf-8 -*-

"""
Ingest - 预测输入层（Top10-Decision）

职责：
- 统一读取“上游预测源文件”（优先新链路 pred_source_latest.csv）
- 提供最小字段契约校验与标准化（避免下游到处写 if/else）
- 兼容旧链路文件名，保证历史可跑，但会给出警告

默认策略（优先级从高到低）：
1) 显式参数 pred_path
2) 环境变量 TOP10_PRED_PATH
3) data/pred/pred_source_latest.csv   （新链路）
4) data/pred/pred_top10_latest.csv    （旧链路兼容）
5) data/pred/pred_top10_latest.csv    （你之前仓库里曾出现过的路径变体）

输出：
- 返回 pandas.DataFrame，至少包含：
  ts_code, trade_date, name, prob, StrengthScore, ThemeBoost
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _warn(msg: str) -> None:
    # 统一 warning 输出（Actions / 本地都能看到）
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
    # 纯 8 位数字
    if s.isdigit() and len(s) == 8:
        return s
    # 尝试用 pandas 解析日期
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y%m%d")
    except Exception:
        return ""


def _pick_pred_path(pred_dir: str = "data/pred", pred_path: Optional[str] = None) -> Tuple[Path, str]:
    """
    返回： (path, source_tag)
    source_tag 用于标记来源：explicit/env/new/old
    """
    # 1) 显式参数
    if pred_path:
        p = Path(pred_path)
        return p, "explicit"

    # 2) 环境变量
    env_path = os.getenv("TOP10_PRED_PATH", "").strip()
    if env_path:
        return Path(env_path), "env"

    # 3) 默认目录下候选文件（按优先级）
    d = Path(pred_dir)
    candidates = [
        (d / "pred_source_latest.csv", "new"),
        (d / "pred_top10_latest.csv", "old"),
        (d / "pred_top10_latest.csv", "old_alt"),
    ]
    for p, tag in candidates:
        if p.exists():
            return p, tag

    # 如果都不存在，仍返回“新链路默认路径”，便于报错信息更明确
    return d / "pred_source_latest.csv", "missing"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化输出列，尽量不报错，但会给出警告。
    下游最关心：ts_code + (prob / StrengthScore / ThemeBoost) + trade_date/name
    """
    # 统一列名大小写/空格
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 必需：ts_code
    if "ts_code" not in df.columns:
        raise ValueError("预测文件必须包含列：ts_code")

    # 可选但强烈建议：trade_date / name
    if "trade_date" not in df.columns:
        _warn("预测文件缺少 trade_date，将填空字符串；建议上游补齐 trade_date（YYYYMMDD）。")
        df["trade_date"] = ""
    if "name" not in df.columns:
        _warn("预测文件缺少 name，将填空字符串；建议上游补齐 name。")
        df["name"] = ""

    # verify_date（可用于回测/对照）
    if "verify_date" not in df.columns:
        df["verify_date"] = ""

    # 核心分数列：缺失则补默认值
    if "prob" not in df.columns:
        _warn("预测文件缺少 prob，将填 0.0；建议上游输出 prob。")
        df["prob"] = 0.0
    if "StrengthScore" not in df.columns:
        _warn("预测文件缺少 StrengthScore，将填 0.0；建议上游输出 StrengthScore。")
        df["StrengthScore"] = 0.0
    if "ThemeBoost" not in df.columns:
        _warn("预测文件缺少 ThemeBoost，将填 0.0；建议上游输出 ThemeBoost。")
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


def load_latest_pred(pred_dir: str = "data/pred", pred_path: Optional[str] = None) -> pd.DataFrame:
    """
    读取最新预测源文件（建议：pred_source_latest.csv）。

    参数：
    - pred_dir: 默认 data/pred
    - pred_path: 可显式指定文件路径（优先级最高）
    - 环境变量 TOP10_PRED_PATH 也可覆盖

    返回：
    - 标准化后的 DataFrame
    """
    path, tag = _pick_pred_path(pred_dir=pred_dir, pred_path=pred_path)

    if not path.exists():
        raise FileNotFoundError(
            f"缺少预测源文件：{path}\n"
            f"请确认：\n"
            f"1) 新链路文件 data/pred/pred_source_latest.csv 是否存在；或\n"
            f"2) 设置环境变量 TOP10_PRED_PATH 指向有效 CSV；或\n"
            f"3) 使用 pred_path 显式传入路径。"
        )

    df = pd.read_csv(path)
    df = _ensure_columns(df)

    # 读到旧链路文件时给出明显提示
    if tag.startswith("old"):
        _warn(f"当前读取的是旧链路文件：{path.name}（tag={tag}）。建议迁移到 pred_source_latest.csv。")
    else:
        print(f"[ingest] loaded: {path} (tag={tag}), rows={len(df)}")

    return df
