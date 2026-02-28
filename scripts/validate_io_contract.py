#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_io_contract.py

目标：锁死 top10-decision 的 IO 契约（路径/命名/字段不允许悄悄改变）

检查内容（P0 必须）：
- 产物文件存在（latest + dated）
- 关键 CSV 必要列存在（允许额外列）
- eval/report 与 exec_date 对齐

注意：
- 这是“验收闸门”，只读文件，不改任何内容。
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


def _pick_exec_date_from_weights(weights_latest: Path) -> str:
    df = _read_csv_any(weights_latest)
    _ensure_cols(df, ["exec_date"], "weights_latest.csv")
    s = df["exec_date"].dropna()
    if s.empty:
        return ""
    return _norm_ymd(s.iloc[0])


def _glob_one(pattern: str) -> List[Path]:
    return sorted(Path(".").glob(pattern))


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
        ["signal_date", "exec_date", "exit_date", "ts_code", "jq_code", "filled_flag", "buy_price", "sell_price", "ret_exec",
         "p_fill_pred", "e_ret_pred", "ev_pred"],
        "decision_learning.csv",
    )

    # ---- dated：用 exec_date 推导（必须能推导出 8 位）
    exec_date = _pick_exec_date_from_weights(weights_latest)
    if not exec_date or len(exec_date) != 8:
        _fail(f"无法从 weights_latest.csv 推导 exec_date（得到：{exec_date}），无法校验 dated 文件。")

    signal_dated = Path(f"docs/signals/top10_{exec_date}.csv")
    weights_dated = Path(f"docs/weights/weights_{exec_date}.csv")
    report_md = Path(f"outputs/decision/decision_report_{exec_date}.md")
    eval_json = Path(f"outputs/decision/eval_{exec_date}.json")
    execution_table = Path(f"data/decision/decision_execution_{exec_date}.csv")
    candidates_snapshot = Path(f"data/decision/decision_candidates_{exec_date}.csv")

    # 这里按你的契约：都必须存在
    _ensure_exists(signal_dated, "signals_dated")
    _ensure_exists(weights_dated, "weights_dated")
    _ensure_exists(report_md, "decision_report")
    _ensure_exists(eval_json, "eval_json")
    _ensure_exists(execution_table, "decision_execution")
    _ensure_exists(candidates_snapshot, "decision_candidates")

    # ---- candidates/execution 字段（最小必要列验收，不约束额外列）
    cand_df = _read_csv_any(candidates_snapshot)
    _ensure_cols(
        cand_df,
        ["ts_code", "name", "p_fill_pred", "e_ret_pred", "cost_est", "risk_penalty", "ev_pred", "signal_date", "exec_date"],
        "decision_candidates_YYYYMMDD.csv",
    )

    exec_df = _read_csv_any(execution_table)
    _ensure_cols(
        exec_df,
        ["exec_date", "ts_code", "jq_code", "filled_flag", "buy_time", "buy_price", "fail_reason", "buy_slippage_bp"],
        "decision_execution_YYYYMMDD.csv",
    )

    # ---- eval json 基础结构
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception:
        _fail(f"无法读取/解析 eval JSON：{eval_json.as_posix()}")

    if str(payload.get("exec_date", "")) != exec_date:
        _warn(f"eval.exec_date 与文件名 exec_date 不一致：payload={payload.get('exec_date')} file={exec_date}")
    else:
        _ok("eval.exec_date 与文件名一致")

    if "paths" not in payload or not isinstance(payload["paths"], dict):
        _fail("eval JSON 缺少 paths 字段或格式不对")
    _ok("eval JSON 结构验收通过")

    # ---- 兜底：确保 outputs/decision 至少有内容
    decision_dir = Path("outputs/decision")
    if not decision_dir.exists():
        _fail("outputs/decision 目录不存在")
    any_files = list(decision_dir.glob("*"))
    if not any_files:
        _fail("outputs/decision 目录为空（不应发生）")
    _ok("outputs/decision 目录非空")

    print(f"[CONTRACT][PASS] IO 契约验收通过：exec_date={exec_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
