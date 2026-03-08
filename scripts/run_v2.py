#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 runner (Orchestrator Only)

硬规则符合性：
1) 数据入口只允许一个：src/top10decision/ingest.py（本文件不读 URL/不读旧文件）
2) sync 独立：scripts/sync_pred_source.py（本文件不做跨仓库拉取）
3) adapters 仅字段映射：src/top10decision/adapters/decisio_adapter.py
4) models / engines 只算分数/概率：src/top10decision/models/*, src/top10decision/engines/*
5) 写文件只在 writers：src/top10decision/writers/*
6) run_v2.py 只编排：不再包含业务细节函数

本次升级：
- 优先使用 ingest.build_model_input() 读取 pred + FS 的统一输入
- 若 FS 缺失，则自动降级回 pred_only
- P_fill / E_ret 不再直接写死规则函数，升级为优先 engine 推理、失败自动回退规则
- 将输入层状态与 engine 审计状态写入 report / eval，便于验收
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd

from top10decision.ingest import (
    build_model_input,
    get_input_status,
    load_pred_snapshot,
)
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router

from top10decision.models.fill_model import fill_model_rule
from top10decision.models.overnight_model import overnight_model_rule
from top10decision.models.costs import cost_estimate_rule, risk_penalty_rule

from top10decision.weights.engine import WeightCaps, build_weights_with_backups

from top10decision.writers.filesystem import (
    ensure_dirs,
    ensure_execution_table,
    ensure_learning_table,
)
from top10decision.writers.artifacts import (
    write_candidates_snapshot,
    write_weights,
    write_signals,
    build_signal_df_for_joinquant,
)
from top10decision.writers.reports import write_decision_report, write_eval_json
from top10decision.writers.io_contract import (
    TOPK_DEFAULT,
    TOPN_DEFAULT,
    W_MAX_DEFAULT,
    THEME_CAP_DEFAULT,
    GROSS_CAP_DEFAULT,
)
from top10decision.writers.io_contract import (
    norm_ymd,
    get_first_value,
    choose_exec_date,
    fmt_num,
)


def _ensure_required_cols(df: pd.DataFrame, required_cols: list[str]) -> None:
    if df is None or df.empty:
        raise RuntimeError("输入数据为空，无法继续运行。")
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"缺少必要字段 {c}，请检查 ingest / adapter / pred_source_latest 输入链路。")


def _build_input_bundle() -> tuple[pd.DataFrame, dict[str, Any], str, str]:
    """
    返回：
    - input_df: 本轮实际使用的输入表
    - input_status: ingest 层状态
    - input_mode: pred_plus_fs / pred_only
    - fs_degrade_reason: 降级原因
    """
    input_status = get_input_status()
    merged_df = build_model_input(include_limit=True, include_truth=False)

    fs_base_loaded = bool(input_status.get("features_base_loaded", False))
    fs_limit_loaded = bool(input_status.get("features_limit_loaded", False))

    if merged_df is not None and not merged_df.empty and fs_base_loaded:
        input_mode = "pred_plus_fs"
        fs_degrade_reason = ""
        input_df = merged_df.copy()
    else:
        input_df = load_pred_snapshot()
        input_mode = "pred_only"
        missing_parts = []
        if not fs_base_loaded:
            missing_parts.append("features_base_missing")
        if not fs_limit_loaded:
            missing_parts.append("features_limit_missing")
        fs_degrade_reason = ",".join(missing_parts) if missing_parts else "merged_input_unavailable"

    if input_df is None or input_df.empty:
        raise RuntimeError("ingest 返回空数据：pred / FS 输入均为空或不可读。")

    return input_df, input_status, input_mode, fs_degrade_reason


def _prepare_runtime_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    运行前最小清洗：
    - 保障必要字段
    - 尽量按单一 trade_date 过滤
    """
    _ensure_required_cols(input_df, ["ts_code", "name"])

    out = input_df.copy()

    if "trade_date" in out.columns:
        td_vals = (
            out["trade_date"]
            .dropna()
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        td_vals = td_vals[td_vals != ""]
        if not td_vals.empty:
            mode_vals = td_vals.mode()
            trade_date = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else str(td_vals.max())
            filtered = out.loc[
                out["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True) == trade_date
            ].copy()
            if not filtered.empty:
                out = filtered

    return out


def _load_engine_apply_func(
    engine_file_name: str,
    func_name: str,
) -> Callable[..., pd.DataFrame] | None:
    """
    先尝试常规 import。
    若 engines/ 目录尚未放 __init__.py，允许退化为按文件路径加载。
    这样本次接线不被包结构细节卡死。
    """
    try:
        if engine_file_name == "pfill_engine.py" and func_name == "apply_pfill_engine":
            from top10decision.engines.pfill_engine import apply_pfill_engine  # type: ignore
            return apply_pfill_engine
        if engine_file_name == "eret_engine.py" and func_name == "apply_eret_engine":
            from top10decision.engines.eret_engine import apply_eret_engine  # type: ignore
            return apply_eret_engine
    except Exception:
        pass

    try:
        root = Path(__file__).resolve().parent.parent
        engine_path = root / "src" / "top10decision" / "engines" / engine_file_name
        if not engine_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(
            f"_dynamic_{engine_file_name.replace('.py', '')}",
            engine_path,
        )
        if spec is None or spec.loader is None:
            return None

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name, None)
    except Exception:
        return None


def _run_pfill_engine(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    apply_func = _load_engine_apply_func("pfill_engine.py", "apply_pfill_engine")
    out = df.copy()

    if apply_func is None:
        out["p_fill_pred"] = fill_model_rule(out)
        out["p_fill_pred_rule"] = out["p_fill_pred"]
        out["p_fill_pred_model"] = pd.NA
        out["p_fill_pred_final"] = out["p_fill_pred"]
        out["p_fill_model_loaded"] = False
        out["p_fill_model_kind"] = ""
        out["p_fill_model_path"] = ""
        out["p_fill_model_feature_mode"] = ""
        out["p_fill_pred_src"] = "rule_direct_fallback"
        out["p_fill_degrade_reason"] = "engine_import_failed"
    else:
        out = apply_func(out)

    audit = {
        "p_fill_pred_src": str(get_first_value(out, "p_fill_pred_src")),
        "p_fill_model_loaded": bool(get_first_value(out, "p_fill_model_loaded", default=False)),
        "p_fill_model_kind": str(get_first_value(out, "p_fill_model_kind")),
        "p_fill_model_path": str(get_first_value(out, "p_fill_model_path")),
        "p_fill_degrade_reason": str(get_first_value(out, "p_fill_degrade_reason")),
    }
    return out, audit


def _run_eret_engine(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    apply_func = _load_engine_apply_func("eret_engine.py", "apply_eret_engine")
    out = df.copy()

    if apply_func is None:
        out["e_ret_pred"] = overnight_model_rule(
            out,
            regime=str(get_first_value(out, "regime_name", default="RISK_ON")),
        )
        out["eret_pred"] = out["e_ret_pred"]
        out["eret_pred_rule"] = out["e_ret_pred"]
        out["eret_pred_model"] = pd.NA
        out["eret_pred_final"] = out["e_ret_pred"]
        out["eret_model_loaded"] = False
        out["eret_model_kind"] = ""
        out["eret_model_path"] = ""
        out["eret_model_feature_mode"] = ""
        out["eret_pred_src"] = "rule_direct_fallback"
        out["eret_degrade_reason"] = "engine_import_failed"
    else:
        out = apply_func(out)

    if "e_ret_pred" not in out.columns and "eret_pred" in out.columns:
        out["e_ret_pred"] = out["eret_pred"]
    if "eret_pred" not in out.columns and "e_ret_pred" in out.columns:
        out["eret_pred"] = out["e_ret_pred"]

    audit = {
        "eret_pred_src": str(get_first_value(out, "eret_pred_src")),
        "eret_model_loaded": bool(get_first_value(out, "eret_model_loaded", default=False)),
        "eret_model_kind": str(get_first_value(out, "eret_model_kind")),
        "eret_model_path": str(get_first_value(out, "eret_model_path")),
        "eret_degrade_reason": str(get_first_value(out, "eret_degrade_reason")),
    }
    return out, audit


def main() -> int:
    ensure_dirs()

    # ✅ 唯一入口：统一通过 ingest 层取数
    input_df, input_status, input_mode, fs_degrade_reason = _build_input_bundle()
    input_df = _prepare_runtime_input(input_df)

    # 日志：用于验收“是否仍在吃 Top10(10行)”
    n_rows = int(len(input_df))
    if 0 < n_rows <= TOPN_DEFAULT:
        print(
            f"[WARN] input_df rows={n_rows} <= TOPN({TOPN_DEFAULT}). "
            "这通常意味着数据源仍是 Top10（10行）而非 decisio 全量。"
        )

    print(f"[run_v2] input_mode={input_mode}")
    if fs_degrade_reason:
        print(f"[run_v2] fs_degrade_reason={fs_degrade_reason}")

    reg = simple_regime(input_df)
    gr = guardrails(input_df)

    regime_name = str(getattr(reg, "regime", "RISK_ON"))
    risk_budget = float(getattr(reg, "risk_budget", 1.0))

    # topk：不超过数据本身行数；guardrails 若不给/给0，则用默认
    gr_topk = int(getattr(gr, "topk", TOPK_DEFAULT)) if hasattr(gr, "topk") else TOPK_DEFAULT
    if gr_topk <= 0:
        gr_topk = TOPK_DEFAULT
    topk = min(max(10, gr_topk), max(10, len(input_df)))

    routed_df = score_router(input_df).head(topk).copy()

    trade_date = get_first_value(routed_df, "trade_date")
    target_trade_date = get_first_value(routed_df, "target_trade_date")
    exec_date = choose_exec_date(trade_date, target_trade_date)
    exit_date = ""

    # STOP 分支：保持原逻辑行为（只输出空 weights + 基础表 + 报告）
    if getattr(gr, "stop_trading", False):
        stop_note = getattr(gr, "reason", "STOP_TRADING")

        exec_path = ensure_execution_table(exec_date=exec_date)
        learning_path = ensure_learning_table()

        cand_snapshot = routed_df.copy()
        cand_snapshot["signal_date"] = norm_ymd(trade_date)
        cand_snapshot["exec_date"] = norm_ymd(exec_date)
        cand_snapshot["exit_date"] = norm_ymd(exit_date)
        cand_snapshot["p_fill_pred"] = 0.0
        cand_snapshot["e_ret_pred"] = 0.0
        cand_snapshot["cost_est"] = cost_estimate_rule()
        cand_snapshot["risk_penalty"] = risk_penalty_rule(regime_name)
        cand_snapshot["ev_pred"] = 0.0
        cand_snapshot["input_mode"] = input_mode
        cand_snapshot["fs_degrade_reason"] = fs_degrade_reason
        cand_snapshot["p_fill_pred_src"] = "stop_zero"
        cand_snapshot["eret_pred_src"] = "stop_zero"
        cand_path = write_candidates_snapshot(cand_snapshot, signal_date=trade_date)

        weights_df = pd.DataFrame(
            columns=["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"]
        )
        weights_latest_path, weights_dated_path = write_weights(weights_df, exec_date=exec_date)

        report_lines: List[str] = []
        report_lines.append(f"# Decision Report ({exec_date or 'unknown'})\n\n")
        report_lines.append(f"**停手：{stop_note}**\n\n")
        report_lines.append("## Input Status\n\n")
        report_lines.append(f"- input_mode: **{input_mode}**\n")
        report_lines.append(f"- fs_degrade_reason: **{fs_degrade_reason or 'none'}**\n")
        report_lines.append(f"- pred_loaded: **{input_status.get('pred_loaded', False)}**\n")
        report_lines.append(f"- features_base_loaded: **{input_status.get('features_base_loaded', False)}**\n")
        report_lines.append(f"- features_limit_loaded: **{input_status.get('features_limit_loaded', False)}**\n")
        report_lines.append(f"- truth_close_loaded: **{input_status.get('truth_close_loaded', False)}**\n")
        report_lines.append(f"- meta_loaded: **{input_status.get('meta_loaded', False)}**\n")
        report_lines.append("- p_fill_pred_src: **stop_zero**\n")
        report_lines.append("- eret_pred_src: **stop_zero**\n")
        report_path = write_decision_report(exec_date, "".join(report_lines))

        write_eval_json(
            exec_date,
            {
                "exec_date": exec_date,
                "signal_date": trade_date,
                "stop_trading": True,
                "reason": stop_note,
                "input_mode": input_mode,
                "fs_degrade_reason": fs_degrade_reason,
                "input_status": input_status,
                "engine_status": {
                    "p_fill_pred_src": "stop_zero",
                    "eret_pred_src": "stop_zero",
                },
            },
        )

        return 0

    # ===== 正常分支：接入 P_fill / E_ret engine =====
    routed_df = routed_df.copy()
    routed_df["signal_date"] = norm_ymd(trade_date)
    routed_df["exec_date"] = norm_ymd(exec_date)
    routed_df["exit_date"] = norm_ymd(exit_date)
    routed_df["regime_name"] = regime_name

    routed_df, pfill_audit = _run_pfill_engine(routed_df)
    routed_df, eret_audit = _run_eret_engine(routed_df)

    cost_est = cost_estimate_rule()
    risk_pen = risk_penalty_rule(regime_name)

    routed_df["cost_est"] = cost_est
    routed_df["risk_penalty"] = risk_pen
    routed_df["ev_pred"] = (
        pd.to_numeric(routed_df["p_fill_pred"], errors="coerce").fillna(0.0).astype(float)
        * pd.to_numeric(routed_df["e_ret_pred"], errors="coerce").fillna(0.0).astype(float)
        - cost_est
        - risk_pen
    )
    routed_df["input_mode"] = input_mode
    routed_df["fs_degrade_reason"] = fs_degrade_reason

    cand_path = write_candidates_snapshot(routed_df.copy(), signal_date=trade_date)

    caps = WeightCaps(
        w_max=W_MAX_DEFAULT,
        theme_cap=THEME_CAP_DEFAULT,
        gross_cap=GROSS_CAP_DEFAULT,
    )
    targets, backups = build_weights_with_backups(routed_df, topn=TOPN_DEFAULT, caps=caps)

    # weights：目标 + 候补（同一文件，候补 weight=0）
    weights_out = pd.concat([targets, backups], ignore_index=True)
    weights_out["exec_date"] = norm_ymd(exec_date)
    weights_out = weights_out[
        ["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"]
    ].copy()
    weights_latest_path, weights_dated_path = write_weights(weights_out, exec_date=exec_date)

    # signals：只输出目标（weight>0）——保持旧 joinquant 契约
    signal_df = build_signal_df_for_joinquant(
        weights_df=weights_out,
        risk_budget=risk_budget,
        regime_name=regime_name,
        trade_date=trade_date,
        target_trade_date=target_trade_date,
    )
    write_signals(signal_df, trade_date=trade_date)

    exec_path = ensure_execution_table(exec_date=exec_date)
    learning_path = ensure_learning_table()

    # decision report（内容/表结构保持原脚本逻辑，仅增加输入审计）
    top_targets = (
        weights_out[weights_out["weight"].astype(float) > 0]
        .copy()
        .sort_values("target_rank")
    )

    lines: List[str] = []
    lines.append(f"# Decision Report ({exec_date or 'unknown'})\n\n")
    lines.append(f"- signal_date: **{trade_date or 'unknown'}**\n")
    lines.append(f"- exec_date: **{exec_date or 'unknown'}**\n")
    lines.append(f"- regime: **{regime_name}**\n")
    lines.append(f"- risk_budget: **{fmt_num(risk_budget, 4)}**\n")
    lines.append(f"- input_mode: **{input_mode}**\n")
    lines.append(f"- fs_degrade_reason: **{fs_degrade_reason or 'none'}**\n\n")

    lines.append("## Input Status\n\n")
    lines.append(f"- pred_loaded: **{input_status.get('pred_loaded', False)}**\n")
    lines.append(f"- pred_rows: **{input_status.get('pred_rows', 0)}**\n")
    lines.append(f"- features_base_loaded: **{input_status.get('features_base_loaded', False)}**\n")
    lines.append(f"- features_base_rows: **{input_status.get('features_base_rows', 0)}**\n")
    lines.append(f"- features_limit_loaded: **{input_status.get('features_limit_loaded', False)}**\n")
    lines.append(f"- features_limit_rows: **{input_status.get('features_limit_rows', 0)}**\n")
    lines.append(f"- truth_close_loaded: **{input_status.get('truth_close_loaded', False)}**\n")
    lines.append(f"- truth_close_rows: **{input_status.get('truth_close_rows', 0)}**\n")
    lines.append(f"- meta_loaded: **{input_status.get('meta_loaded', False)}**\n\n")

    lines.append("## Engine Status\n\n")
    lines.append(f"- p_fill_pred_src: **{pfill_audit.get('p_fill_pred_src', '') or 'unknown'}**\n")
    lines.append(f"- p_fill_model_loaded: **{pfill_audit.get('p_fill_model_loaded', False)}**\n")
    lines.append(f"- p_fill_model_kind: **{pfill_audit.get('p_fill_model_kind', '') or 'none'}**\n")
    lines.append(f"- p_fill_degrade_reason: **{pfill_audit.get('p_fill_degrade_reason', '') or 'none'}**\n")
    lines.append(f"- eret_pred_src: **{eret_audit.get('eret_pred_src', '') or 'unknown'}**\n")
    lines.append(f"- eret_model_loaded: **{eret_audit.get('eret_model_loaded', False)}**\n")
    lines.append(f"- eret_model_kind: **{eret_audit.get('eret_model_kind', '') or 'none'}**\n")
    lines.append(f"- eret_degrade_reason: **{eret_audit.get('eret_degrade_reason', '') or 'none'}**\n\n")

    lines.append("## Artifacts\n\n")
    lines.append(f"- candidates_snapshot: `{cand_path}`\n")
    lines.append(f"- execution_table: `{exec_path}`\n")
    lines.append(f"- learning_table: `{learning_path}`\n")
    lines.append(f"- weights_latest: `{weights_latest_path}`\n")
    lines.append(f"- weights_dated: `{weights_dated_path}`\n\n")

    lines.append("## TopN Targets\n\n")
    lines.append("| rank | ts_code | name | weight | EV | P_fill | E_ret |\n")
    lines.append("|---:|---|---|---:|---:|---:|---:|\n")
    merged_targets = top_targets.merge(
        routed_df[["ts_code", "ev_pred", "p_fill_pred", "e_ret_pred"]],
        on=["ts_code", "ev_pred"],
        how="left",
    )
    for _, r in merged_targets.iterrows():
        lines.append(
            f"| {int(r.get('target_rank', 0))} | {r.get('ts_code', '')} | {r.get('name', '')} | "
            f"{fmt_num(r.get('weight', 0.0), 6)} | {fmt_num(r.get('ev_pred', ''), 6)} | "
            f"{fmt_num(r.get('p_fill_pred', ''), 6)} | {fmt_num(r.get('e_ret_pred', ''), 6)} |\n"
        )

    report_path = write_decision_report(exec_date, "".join(lines))

    eval_payload = {
        "signal_date": trade_date,
        "exec_date": exec_date,
        "regime": regime_name,
        "risk_budget": risk_budget,
        "topk": int(len(routed_df)),
        "picked": int(len(top_targets)),
        "cost_est": cost_est,
        "risk_penalty": risk_pen,
        "input_mode": input_mode,
        "fs_degrade_reason": fs_degrade_reason,
        "input_status": input_status,
        "engine_status": {
            "p_fill_pred_src": pfill_audit.get("p_fill_pred_src", ""),
            "p_fill_model_loaded": pfill_audit.get("p_fill_model_loaded", False),
            "p_fill_model_kind": pfill_audit.get("p_fill_model_kind", ""),
            "p_fill_model_path": pfill_audit.get("p_fill_model_path", ""),
            "p_fill_degrade_reason": pfill_audit.get("p_fill_degrade_reason", ""),
            "eret_pred_src": eret_audit.get("eret_pred_src", ""),
            "eret_model_loaded": eret_audit.get("eret_model_loaded", False),
            "eret_model_kind": eret_audit.get("eret_model_kind", ""),
            "eret_model_path": eret_audit.get("eret_model_path", ""),
            "eret_degrade_reason": eret_audit.get("eret_degrade_reason", ""),
        },
        "paths": {
            "candidates": cand_path,
            "execution": exec_path,
            "learning": learning_path,
            "weights_latest": weights_latest_path,
            "weights_dated": weights_dated_path,
            "decision_report": report_path,
        },
    }
    write_eval_json(exec_date, eval_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
