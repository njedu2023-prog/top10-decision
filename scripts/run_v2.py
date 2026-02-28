#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 runner (Orchestrator Only)

硬规则符合性：
1) 数据入口只允许一个：src/top10decision/ingest.py（本文件不读 URL/不读旧文件）
2) sync 独立：scripts/sync_pred_source.py（本文件不做跨仓库拉取）
3) adapters 仅字段映射：src/top10decision/adapters/decisio_adapter.py
4) models 只算分数/概率：src/top10decision/models/*
5) 写文件只在 writers：src/top10decision/writers/*
6) run_v2.py 只编排：不再包含业务细节函数
"""

from __future__ import annotations

from typing import List

import pandas as pd

from top10decision.ingest import load_pred_snapshot
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router

from top10decision.models.fill_model import fill_model_rule
from top10decision.models.overnight_model import overnight_model_rule
from top10decision.models.costs import cost_estimate_rule, risk_penalty_rule

from top10decision.weights.engine import WeightCaps, build_weights_with_backups

from top10decision.writers.filesystem import ensure_dirs, ensure_execution_table, ensure_learning_table
from top10decision.writers.artifacts import (
    write_candidates_snapshot,
    write_weights,
    write_signals,
    build_signal_df_for_joinquant,
)
from top10decision.writers.reports import write_decision_report, write_eval_json
from top10decision.writers.io_contract import TOPK_DEFAULT, TOPN_DEFAULT, W_MAX_DEFAULT, THEME_CAP_DEFAULT, GROSS_CAP_DEFAULT
from top10decision.writers.io_contract import norm_ymd, get_first_value, choose_exec_date, fmt_num


def main() -> int:
    ensure_dirs()

    # ✅ 唯一入口：只读固定快照 data/pred/pred_source_latest.csv
    pred_df = load_pred_snapshot()
    if pred_df is None or pred_df.empty:
        raise RuntimeError("ingest 返回空数据：data/pred/pred_source_latest.csv 为空或不可读。")

    # 基础字段保障（入口已做适配，但这里再做显式保护，避免 silent）
    for c in ("ts_code", "name"):
        if c not in pred_df.columns:
            raise RuntimeError(f"缺少必要字段 {c}，请检查 pred_source_latest.csv 以及 decisio_adapter 映射。")

    # 日志：用于验收“是否仍在吃 Top10(10行)”
    n_rows = int(len(pred_df))
    if 0 < n_rows <= TOPN_DEFAULT:
        print(f"[WARN] pred_df rows={n_rows} <= TOPN({TOPN_DEFAULT}). 这通常意味着数据源仍是 Top10（10行）而非 decisio 全量。")

    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    regime_name = str(getattr(reg, "regime", "RISK_ON"))
    risk_budget = float(getattr(reg, "risk_budget", 1.0))

    # topk：不超过数据本身行数；guardrails 若不给/给0，则用默认
    gr_topk = int(getattr(gr, "topk", TOPK_DEFAULT)) if hasattr(gr, "topk") else TOPK_DEFAULT
    if gr_topk <= 0:
        gr_topk = TOPK_DEFAULT
    topk = min(max(10, gr_topk), max(10, len(pred_df)))

    routed_df = score_router(pred_df).head(topk).copy()

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
        cand_path = write_candidates_snapshot(cand_snapshot, signal_date=trade_date)

        weights_df = pd.DataFrame(columns=["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"])
        weights_latest_path, weights_dated_path = write_weights(weights_df, exec_date=exec_date)

        report_path = write_decision_report(exec_date, f"# Decision Report ({exec_date or 'unknown'})\n\n**停手：{stop_note}**\n")
        eval_path = write_eval_json(exec_date, {"exec_date": exec_date, "signal_date": trade_date, "stop_trading": True, "reason": stop_note})

        return 0

    # ===== 正常分支：P0.1 =====
    routed_df = routed_df.copy()
    routed_df["signal_date"] = norm_ymd(trade_date)
    routed_df["exec_date"] = norm_ymd(exec_date)
    routed_df["exit_date"] = norm_ymd(exit_date)

    routed_df["p_fill_pred"] = fill_model_rule(routed_df)
    routed_df["e_ret_pred"] = overnight_model_rule(routed_df, regime=regime_name)

    cost_est = cost_estimate_rule()
    risk_pen = risk_penalty_rule(regime_name)

    routed_df["cost_est"] = cost_est
    routed_df["risk_penalty"] = risk_pen
    routed_df["ev_pred"] = routed_df["p_fill_pred"].astype(float) * routed_df["e_ret_pred"].astype(float) - cost_est - risk_pen

    cand_path = write_candidates_snapshot(routed_df.copy(), signal_date=trade_date)

    caps = WeightCaps(w_max=W_MAX_DEFAULT, theme_cap=THEME_CAP_DEFAULT, gross_cap=GROSS_CAP_DEFAULT)
    targets, backups = build_weights_with_backups(routed_df, topn=TOPN_DEFAULT, caps=caps)

    # weights：目标 + 候补（同一文件，候补 weight=0）
    weights_out = pd.concat([targets, backups], ignore_index=True)
    weights_out["exec_date"] = norm_ymd(exec_date)
    weights_out = weights_out[["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"]].copy()
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

    # decision report（内容/表结构保持原脚本逻辑）
    top_targets = weights_out[weights_out["weight"].astype(float) > 0].copy().sort_values("target_rank")

    lines: List[str] = []
    lines.append(f"# Decision Report ({exec_date or 'unknown'})\n\n")
    lines.append(f"- signal_date: **{trade_date or 'unknown'}**\n")
    lines.append(f"- exec_date: **{exec_date or 'unknown'}**\n")
    lines.append(f"- regime: **{regime_name}**\n")
    lines.append(f"- risk_budget: **{fmt_num(risk_budget, 4)}**\n\n")

    lines.append("## Artifacts\n\n")
    lines.append(f"- candidates_snapshot: `{cand_path}`\n")
    lines.append(f"- execution_table: `{exec_path}`\n")
    lines.append(f"- learning_table: `{learning_path}`\n")
    lines.append(f"- weights_latest: `{weights_latest_path}`\n")
    lines.append(f"- weights_dated: `{weights_dated_path}`\n\n")

    lines.append("## TopN Targets\n\n")
    lines.append("| rank | ts_code | name | weight | EV |\n")
    lines.append("|---:|---|---|---:|---:|\n")
    for _, r in top_targets.iterrows():
        lines.append(
            f"| {int(r.get('target_rank', 0))} | {r.get('ts_code','')} | {r.get('name','')} | "
            f"{fmt_num(r.get('weight', 0.0), 6)} | {fmt_num(r.get('ev_pred', ''), 6)} |\n"
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
