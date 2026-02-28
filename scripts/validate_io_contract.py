#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_io_contract.py

目标：锁死 top10-decision 的 IO 契约（路径/命名/字段不允许悄悄改变）

✅ 修复点（2026-03-01）：
- signals 的 dated 文件命名使用 trade_date：
  docs/signals/top10_{trade_date}.csv
- candidates_snapshot 命名使用 signal_date（本系统 signal_date == trade_date）：
  data/decision/decision_candidates_{trade_date}.csv
- weights/report/eval/execution 使用 exec_date：
  docs/weights/weights_{exec_date}.csv
  outputs/decision/decision_report_{exec_date}.md
  outputs/decision/eval_{exec_date}.json
  data/decision/decision_execution_{exec_date}.csv

检查内容（P0 必须）：
- 产物文件存在（latest + dated）
- 关键 CSV 必要列存在（允许额外列）
- eval/report 与 exec_date 对齐（最小一致性）
- ✅ P1：learning_table 结构锁死（字段升级后仍要强校验）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


# =========================
# helpers
# =========================

def _fail(msg: str) -> None:
    print(f"[CONTRACT][FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _warn(msg: str) -> None:
    print(f"[CONTRACT][WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[CONTRACT][OK] {msg}")


def _ensure_exists(p: Path, label: str) -> None:
    if not p.exists():
        _fail(f"缺少产物：{label} -> {p.as_posix()}")
    _ok(f"存在：{label} -> {p.as_posix()}")


def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    _fail(f"无法读取 CSV：{path.as_posix()}")
    return pd.DataFrame()


def _ensure_cols(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"{label} 缺少必要列：{missing}；现有列：{list(df.columns)}")
    _ok(f"{label} 列验收通过（至少包含：{list(required)}）")


def _norm_ymd(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 8 and s.isdigit():
        return s
    try:
        i = int(float(s))
        s2 = str(i)
        return s2 if (len(s2) == 8 and s2.isdigit()) else s2
    except Exception:
        return s


def _first_ymd_from_col(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    return _norm_ymd(s.iloc[0])


def _get_learning_required_cols() -> List[str]:
    """
    ✅ P1：learning_table 的字段不允许漂移。
    优先从 writers.filesystem 导入 LEARNING_COLUMNS（最稳），导入失败再用兜底。
    """
    try:
        from top10decision.writers.filesystem import LEARNING_COLUMNS  # type: ignore
        # 这里不要求“全列都必须出现”，但至少要包含其中的关键列
        # 为避免未来扩展导致 validate 过于严格，我们取一个“必须集合”
        must = [
            "signal_date",
            "exec_date",
            "exit_date",
            "ts_code",
            "jq_code",
            "name",
            "weight_exec",
            "filled_flag",
            "fill_rate_real",
            "buy_price",
            "sell_price",
            "ret_exec",
            "p_fill_pred",
            "e_ret_pred",
            "cost_est",
            "risk_penalty",
            "ev_pred",
            "e_ret_real",
            "ev_real",
            "regime",
            "risk_budget",
            "version",
            "generated_at_bjt",
            "commit_sha",
        ]
        # 如果 LEARNING_COLUMNS 里缺了 must 里的任何一个，说明 filesystem 与 validate 不一致，应立即失败
        miss_in_schema = [c for c in must if c not in LEARNING_COLUMNS]
        if miss_in_schema:
            _fail(f"writers.filesystem.LEARNING_COLUMNS 缺少关键字段：{miss_in_schema}（请先修复 schema 定义）")
        return must
    except Exception:
        # 兜底：按我们当前 P1 约定的 must 集合
        return [
            "signal_date",
            "exec_date",
            "exit_date",
            "ts_code",
            "jq_code",
            "name",
            "weight_exec",
            "filled_flag",
            "fill_rate_real",
            "buy_price",
            "sell_price",
            "ret_exec",
            "p_fill_pred",
            "e_ret_pred",
            "cost_est",
            "risk_penalty",
            "ev_pred",
            "e_ret_real",
            "ev_real",
            "regime",
            "risk_budget",
            "version",
            "generated_at_bjt",
            "commit_sha",
        ]


# =========================
# contract check
# =========================

def main() -> int:
    # ---- 固定 latest
    signal_latest = Path("docs/signals/top10_latest.csv")
    weights_latest = Path("docs/weights/weights_latest.csv")
    learning_table = Path("data/decision/decision_learning.csv")

    _ensure_exists(signal_latest, "signals_latest")
    _ensure_exists(weights_latest, "weights_latest")
    _ensure_exists(learning_table, "decision_learning")

    # ---- latest CSV 字段
    sig_df = _read_csv_any(signal_latest)
    _ensure_cols(
        sig_df,
        ["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"],
        "signals_latest.csv",
    )

    w_df = _read_csv_any(weights_latest)
    _ensure_cols(
        w_df,
        ["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"],
        "weights_latest.csv",
    )

    learn_df = _read_csv_any(learning_table)
    _ensure_cols(
        learn_df,
        _get_learning_required_cols(),
        "decision_learning.csv",
    )

    # ---- 关键日期：trade_date 来自 signals_latest；exec_date 来自 weights_latest
    trade_date = _first_ymd_from_col(sig_df, "trade_date")
    exec_date = _first_ymd_from_col(w_df, "exec_date")

    if not trade_date or len(trade_date) != 8:
        _fail(f"无法从 signals_latest.csv 推导 trade_date（得到：{trade_date}）")
    if not exec_date or len(exec_date) != 8:
        _fail(f"无法从 weights_latest.csv 推导 exec_date（得到：{exec_date}）")

    _ok(f"推导日期：trade_date={trade_date} exec_date={exec_date}")

    # ---- dated：signals 用 trade_date
    signal_dated = Path(f"docs/signals/top10_{trade_date}.csv")
    _ensure_exists(signal_dated, "signals_dated(trade_date)")
    sig_dated_df = _read_csv_any(signal_dated)
    _ensure_cols(
        sig_dated_df,
        ["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"],
        f"signals_dated(top10_{trade_date}.csv)",
    )

    # ---- candidates_snapshot 用 trade_date（signal_date）
    candidates_snapshot = Path(f"data/decision/decision_candidates_{trade_date}.csv")
    _ensure_exists(candidates_snapshot, "decision_candidates(trade_date)")
    cand_df = _read_csv_any(candidates_snapshot)
    _ensure_cols(
        cand_df,
        ["ts_code", "name", "p_fill_pred", "e_ret_pred", "cost_est", "risk_penalty", "ev_pred", "signal_date", "exec_date"],
        f"decision_candidates_{trade_date}.csv",
    )

    # ---- weights/report/eval/execution 用 exec_date
    weights_dated = Path(f"docs/weights/weights_{exec_date}.csv")
    report_md = Path(f"outputs/decision/decision_report_{exec_date}.md")
    eval_json = Path(f"outputs/decision/eval_{exec_date}.json")
    execution_table = Path(f"data/decision/decision_execution_{exec_date}.csv")

    _ensure_exists(weights_dated, "weights_dated(exec_date)")
    _ensure_exists(report_md, "decision_report(exec_date)")
    _ensure_exists(eval_json, "eval_json(exec_date)")
    _ensure_exists(execution_table, "decision_execution(exec_date)")

    exec_df = _read_csv_any(execution_table)
    _ensure_cols(
        exec_df,
        ["exec_date", "ts_code", "jq_code", "filled_flag", "buy_time", "buy_price", "fail_reason", "buy_slippage_bp"],
        f"decision_execution_{exec_date}.csv",
    )

    # ---- eval json 基础结构
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception:
        _fail(f"无法读取/解析 eval JSON：{eval_json.as_posix()}")

    if _norm_ymd(payload.get("exec_date", "")) != exec_date:
        _warn(f"eval.exec_date 与 weights_latest.exec_date 不一致：payload={payload.get('exec_date')} weights={exec_date}")
    else:
        _ok("eval.exec_date 与 exec_date 一致")

    if "paths" not in payload or not isinstance(payload["paths"], dict):
        _fail("eval JSON 缺少 paths 字段或格式不对")
    _ok("eval JSON 结构验收通过")

    # ---- 兜底：确保 outputs/decision 至少有内容
    decision_dir = Path("outputs/decision")
    if not decision_dir.exists():
        _fail("outputs/decision 目录不存在")
    if not list(decision_dir.glob("*")):
        _fail("outputs/decision 目录为空（不应发生）")
    _ok("outputs/decision 目录非空")

    print(f"[CONTRACT][PASS] IO 契约验收通过：trade_date={trade_date} exec_date={exec_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
