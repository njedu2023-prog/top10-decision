#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd

from top10decision.utils import to_jq_code
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal

from top10decision.writers.io_contract import (
    norm_ymd,
    ensure_cols,
    SIGNAL_LATEST,
    SIGNAL_DATED_FMT,
    WEIGHTS_LATEST,
    WEIGHTS_DATED_FMT,
    CANDIDATES_FMT,
)


# =========================
# 内部小工具：元数据
# =========================

def _commit_sha() -> str:
    # GitHub Actions: GITHUB_SHA；本地可为空
    return str(os.getenv("GITHUB_SHA", "")).strip()


def _generated_at_bjt() -> str:
    # 北京时间 ISO（不依赖 pytz）
    try:
        ts = pd.Timestamp.now(tz="Asia/Shanghai")
    except Exception:
        ts = pd.Timestamp.now()
    return ts.strftime("%Y-%m-%d %H:%M:%S%z")


def _ensure_explain_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    candidates_snapshot 强契约：解释字段缺失必须补齐。
    这里不做任何“计算”，只补默认值，防止 silent 退化。
    """
    out = df.copy()

    defaults = {
        # 日期链路（如缺就空）
        "signal_date": "",
        "exec_date": "",
        "exit_date": "",

        # 预测与解释
        "p_fill_pred": 0.0,
        "e_ret_pred": 0.0,
        "cost_est": 0.0,
        "risk_penalty": 0.0,
        "ev_pred": 0.0,

        # 闭环需要但允许空
        "weight_exec": "",
        "regime": "",
        "risk_budget": "",
        "version": 1,
        "generated_at_bjt": _generated_at_bjt(),
        "commit_sha": _commit_sha(),
    }

    for c, v in defaults.items():
        if c not in out.columns:
            out[c] = v

    # name/ts_code 如果缺，应该在上游就报错；这里仍保护一下
    if "ts_code" not in out.columns:
        raise RuntimeError("write_candidates_snapshot: 缺少必要字段 ts_code（上游输出不合格）")
    if "name" not in out.columns:
        out["name"] = ""

    return out


# =========================
# signals
# =========================

def write_signals(latest_df: pd.DataFrame, trade_date: str) -> None:
    """
    signals IO 契约：
    - docs/signals/top10_latest.csv
    - docs/signals/top10_YYYYMMDD.csv
    """
    if latest_df is None or latest_df.empty:
        raise RuntimeError("write_signals: latest_df 为空，拒绝写 signals（避免用空信号覆盖 latest）。")

    # 最小字段保护（保持 joinquant 契约稳定）
    ensure_cols(latest_df, ["jq_code", "target_weight", "trade_date", "target_trade_date"])

    # 写 latest
    write_latest_signal(latest_df, out_path=str(SIGNAL_LATEST))

    # 写 dated（按参数 trade_date 决定文件名）
    td = norm_ymd(trade_date)
    if td:
        write_latest_signal(latest_df, out_path=SIGNAL_DATED_FMT.format(yyyymmdd=td))


# =========================
# weights
# =========================

def write_weights(weights_df: pd.DataFrame, exec_date: str) -> Tuple[str, str]:
    """
    weights IO 契约：
    - docs/weights/weights_latest.csv
    - docs/weights/weights_YYYYMMDD.csv
    """
    Path("docs/weights").mkdir(parents=True, exist_ok=True)

    latest_path = str(WEIGHTS_LATEST)
    dated = norm_ymd(exec_date)
    dated_path = WEIGHTS_DATED_FMT.format(yyyymmdd=dated) if dated else "docs/weights/weights_unknown.csv"

    weights_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(dated_path, index=False, encoding="utf-8-sig")
    return latest_path, dated_path


# =========================
# candidates_snapshot
# =========================

def write_candidates_snapshot(cand_df: pd.DataFrame, signal_date: str) -> str:
    """
    candidates IO 契约：
    - data/decision/decision_candidates_YYYYMMDD.csv

    强烈建议（P1）：
    - 保证解释字段齐全（p_fill_pred/e_ret_pred/cost_est/risk_penalty/ev_pred）
    - 写入审计元数据（commit_sha/generated_at/version）
    - 未来 merge 聚宽反馈时，至少能定位“哪一次运行写的这份候选快照”
    """
    if cand_df is None or cand_df.empty:
        raise RuntimeError("write_candidates_snapshot: cand_df 为空，拒绝写 candidates_snapshot。")

    Path("data/decision").mkdir(parents=True, exist_ok=True)
    sd = norm_ymd(signal_date) or "unknown"
    path = CANDIDATES_FMT.format(yyyymmdd=sd)

    out = _ensure_explain_cols(cand_df)

    # 若上游没写 signal_date，则用函数参数兜底
    if out["signal_date"].astype(str).eq("").all():
        out["signal_date"] = norm_ymd(signal_date)

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# =========================
# joinquant signal builder（保持旧契约）
# =========================

def build_signal_df_for_joinquant(
    weights_df: pd.DataFrame,
    risk_budget: float,
    regime_name: str,
    trade_date: str,
    target_trade_date: str,
) -> pd.DataFrame:
    """
    兼容旧 joinquant 信号格式：
    - 只输出 weight>0 的目标行（候补不进入 signals）
    输出字段保持原契约：
    ["trade_date","target_trade_date","jq_code","target_weight","risk_budget","regime","reason"]
    """
    ensure_cols(weights_df, ["ts_code", "weight"])
    df = weights_df.copy()
    df = df[df["weight"].astype(float) > 0].copy()

    df["jq_code"] = df["ts_code"].apply(to_jq_code)
    df["trade_date"] = norm_ymd(trade_date)
    df["target_trade_date"] = norm_ymd(target_trade_date)
    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)
    df["reason"] = "P0_EV_weight"
    df["target_weight"] = df["weight"].astype(float)

    return df[["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]].copy()
