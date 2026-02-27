#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 runner (Professional Skeleton)

核心目标（主线契约）：
- 输入：T 日 a-top10 Full TopK 候选（默认 TopK=100）
- 输出：在 T+1 可成交约束下，输出“买得到且隔夜正期望”的权重给聚宽执行

P0（先跑通闭环）：
- fill_model：规则版 P_fill_pred
- overnight_model：规则版 E_ret_pred
- weight_engine：EV -> weights（含上限约束的最小实现）
- 三张表落库（保证自学习不跑偏）：
  1) data/decision/decision_candidates_{T}.csv
  2) data/decision/decision_execution_{exec_date}.csv（P0 先落空表，等聚宽回填）
  3) data/decision/decision_learning.csv（P0 先建表，后续接执行一致标签）

兼容旧输出（不破坏现有聚宽读取逻辑）：
- docs/signals/top10_latest.csv
- docs/signals/top10_YYYYMMDD.csv
- docs/reports/daily_latest.md
- docs/reports/daily_YYYYMMDD.md

新增输出（本次主线）：
- docs/weights/weights_latest.csv
- docs/weights/weights_YYYYMMDD.csv
- outputs/decision/decision_report_YYYYMMDD.md
- outputs/decision/eval_YYYYMMDD.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.utils import to_jq_code
from top10decision.position.allocator import allocate_equal_weight
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal


# =========================
# Config (P0)
# =========================

TOPK_DEFAULT = 100
TOPN_DEFAULT = 10

# 权重/暴露上限（P0 最小实现）
W_MAX_DEFAULT = 0.12          # 单票上限
THEME_CAP_DEFAULT = 0.35      # 题材暴露上限（同一主题/板块合计）
GROSS_CAP_DEFAULT = 1.00      # 总仓位上限（目标总和）

# 成本与风险（P0 先用定值/简单规则）
COST_BP_DEFAULT = 8.0         # 交易成本估计（bp），用于 EV 扣减
RISK_PENALTY_OFF = 0.00       # risk_on 下风险惩罚
RISK_PENALTY_ON = 0.02        # risk_off 下风险惩罚（用作 EV 线性惩罚）


# =========================
# Helpers
# =========================

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


def _norm_ymd(v) -> str:
    """把日期规范成 YYYYMMDD（去掉 .0 / 科学计数法 / 空值）"""
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


def _get_first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    if col in ("trade_date", "target_trade_date", "exec_date", "exit_date", "signal_date"):
        return _norm_ymd(s.iloc[0])
    return str(s.iloc[0])


def _fmt_num(x, nd=6):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "" if x is None else str(x)


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _pick_theme(row: pd.Series) -> str:
    # 题材字段：优先 theme / board / industry
    for k in ("theme", "Theme", "board", "industry", "sector"):
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _ensure_dirs() -> None:
    Path("data/pred").mkdir(parents=True, exist_ok=True)
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    Path("docs/signals").mkdir(parents=True, exist_ok=True)
    Path("docs/reports").mkdir(parents=True, exist_ok=True)
    Path("docs/weights").mkdir(parents=True, exist_ok=True)
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    Path("outputs/learning").mkdir(parents=True, exist_ok=True)


def _choose_exec_date(trade_date: str, target_trade_date: str) -> str:
    """
    文件命名与执行日优先级：
    - 优先使用 target_trade_date 作为 exec_date
    - 否则 fallback 到 trade_date（P0 先不引入交易日历推导）
    """
    td = _norm_ymd(trade_date)
    ttd = _norm_ymd(target_trade_date)
    return ttd or td


# =========================
# P0 Models (rule-based)
# =========================

def fill_model_rule(df: pd.DataFrame) -> pd.Series:
    """
    P_fill(i | exec_date) 规则版（P0）
    - 若字段缺失，则给一个保守的中性值 0.35
    - 若能拿到 open_times / seal_amount / turnover_rate 等字段，则做简单映射
    """
    base = 0.35

    # 尝试使用可用字段（可空）
    open_times = df.get("open_times", pd.Series([None] * len(df)))
    seal_amount = df.get("seal_amount", pd.Series([None] * len(df)))
    turnover = df.get("turnover_rate", pd.Series([None] * len(df)))

    p = []
    for i in range(len(df)):
        ot = _safe_float(open_times.iloc[i], default=float("nan"))
        sa = _safe_float(seal_amount.iloc[i], default=float("nan"))
        tr = _safe_float(turnover.iloc[i], default=float("nan"))

        pi = base

        # 开板次数越多，越容易买到（但也可能弱，这里只做成交性）
        if not pd.isna(ot):
            pi += min(max(ot, 0.0), 5.0) * 0.06  # 0~0.30

        # 封单金额越大，越不容易买到（提高一字/秒板概率）
        if not pd.isna(sa):
            # 做一个缩放：金额很大时扣到较低
            pi -= min(sa / 1e8, 5.0) * 0.05  # 最多 -0.25

        # 换手率高 -> 可成交性更强（轻微加成）
        if not pd.isna(tr):
            pi += min(max(tr, 0.0), 20.0) * 0.005  # 0~0.10

        # 裁剪到 [0.02, 0.98]
        pi = max(0.02, min(0.98, pi))
        p.append(pi)

    return pd.Series(p, index=df.index, name="p_fill_pred")


def overnight_model_rule(df: pd.DataFrame, regime: str) -> pd.Series:
    """
    E_ret(i | buy=T+1, sell=T+2) 规则版（P0）
    - 使用 a-top10 的 Probability / StrengthScore / ThemeBoost 做 proxy
    - regime=RISK_OFF 时整体下调
    返回：e_ret_pred（单位：小数收益，如 0.02 表示 +2%）
    """
    # 获取可用字段
    prob = df.get("Probability", df.get("prob", df.get("probability", pd.Series([None] * len(df)))))
    strength = df.get("StrengthScore", df.get("strength", pd.Series([None] * len(df))))
    theme = df.get("ThemeBoost", df.get("theme", pd.Series([None] * len(df))))

    e = []
    for i in range(len(df)):
        p = _safe_float(prob.iloc[i], default=0.3)
        s = _safe_float(strength.iloc[i], default=0.0)
        t = _safe_float(theme.iloc[i], default=0.0)

        # 基础：概率映射到一个 0~3% 的范围
        ei = (max(0.0, min(1.0, p)) - 0.2) * 0.03  # p=0.2 ->0, p=1 -> 2.4%
        # 强度与题材加成：轻微加成
        ei += max(-2.0, min(10.0, s)) * 0.001      # -0.2% ~ +1.0%
        ei += max(-1.0, min(3.0, t)) * 0.003       # -0.3% ~ +0.9%

        # regime 风险下调
        if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
            ei -= 0.006  # -0.6%

        # 裁剪
        ei = max(-0.05, min(0.08, ei))  # [-5%, +8%]
        e.append(ei)

    return pd.Series(e, index=df.index, name="e_ret_pred")


def risk_penalty_rule(regime: str) -> float:
    if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
        return float(RISK_PENALTY_ON)
    return float(RISK_PENALTY_OFF)


def cost_estimate_rule() -> float:
    # bp -> 小数收益
    return float(COST_BP_DEFAULT) / 10000.0


# =========================
# Weight Engine (P0)
# =========================

@dataclass
class WeightCaps:
    w_max: float = W_MAX_DEFAULT
    theme_cap: float = THEME_CAP_DEFAULT
    gross_cap: float = GROSS_CAP_DEFAULT


def build_weights_topn(
    candidates: pd.DataFrame,
    topn: int,
    caps: WeightCaps,
) -> pd.DataFrame:
    """
    输入：带 ev_pred/theme 的候选
    输出：TopN 权重表（含 target_rank/backup_rank）
    P0：先按 EV 排序，逐个塞入，满足单票上限与题材暴露上限与总仓位上限
    """
    _ensure_cols(candidates, ["ts_code", "name", "ev_pred"])

    df = candidates.sort_values("ev_pred", ascending=False).reset_index(drop=True).copy()
    df["theme"] = df.apply(_pick_theme, axis=1)

    picked_rows = []
    theme_used: Dict[str, float] = {}
    gross_used = 0.0

    # 先用等权作为初始目标权重，然后用 caps 裁剪
    # 目标总仓位 caps.gross_cap 内，分配给最多 topn 只
    if topn <= 0:
        topn = TOPN_DEFAULT
    base_w = min(caps.gross_cap, 1.0) / float(topn)

    for i in range(len(df)):
        if len(picked_rows) >= topn:
            break

        row = df.loc[i].copy()
        th = row.get("theme", "") or ""

        w = min(base_w, caps.w_max)
        # 题材暴露上限
        if th:
            used = theme_used.get(th, 0.0)
            if used + w > caps.theme_cap:
                continue

        if gross_used + w > caps.gross_cap + 1e-9:
            break

        row["weight"] = w
        picked_rows.append(row)
        gross_used += w
        if th:
            theme_used[th] = theme_used.get(th, 0.0) + w

    out = pd.DataFrame(picked_rows).copy()
    if out.empty:
        return out

    out["target_rank"] = list(range(1, len(out) + 1))
    out["backup_rank"] = ""  # P0：先不输出候补序；候补由 candidates 的剩余 EV 排序隐含
    return out


# =========================
# Writers
# =========================

def _write_human_report(pred_topk: pd.DataFrame, out_path: str, title: str, stop_note: Optional[str] = None) -> None:
    """
    生成“人类可读 Top10 表格”
    列：排名 / 代码 / 股票 / Probability / 强度得分 / 题材加成 / 板块
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trade_date = _get_first_value(pred_topk, "trade_date")
    target_trade_date = _get_first_value(pred_topk, "target_trade_date")

    lines: List[str] = []
    lines.append(f"# {title}\n\n")
    lines.append(f"- trade_date（信号生成日）: **{trade_date if trade_date else '未知'}**\n")
    lines.append(f"- target_trade_date（执行交易日）: **{target_trade_date if target_trade_date else '未知/未填'}**\n\n")

    if stop_note:
        lines.append(f"**停手：{stop_note}**\n\n")

    lines.append("| 排名 | 代码 | 股票 | Probability | 强度得分 | 题材加成 | 板块 |\n")
    lines.append("|---:|---|---|---:|---:|---:|---|\n")

    d = pred_topk.head(10).copy()
    if "rank" not in d.columns:
        d["rank"] = list(range(1, len(d) + 1))

    for _, r in d.iterrows():
        rank = r.get("rank", "")
        ts_code = r.get("ts_code", "")
        name = r.get("name", "")
        prob = r.get("Probability", r.get("prob", r.get("probability", "")))
        strength = r.get("StrengthScore", r.get("strength", ""))
        theme = r.get("ThemeBoost", r.get("theme", ""))
        board = r.get("board", r.get("industry", ""))

        lines.append(
            f"| {rank} | {ts_code} | {name} | {_fmt_num(prob, 6)} | {_fmt_num(strength, 4)} | {_fmt_num(theme, 6)} | {board} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_signals(latest_df: pd.DataFrame, trade_date: str) -> None:
    """
    兼容旧信号输出（给聚宽策略读取）
    - latest：docs/signals/top10_latest.csv（覆盖）
    - dated：docs/signals/top10_YYYYMMDD.csv（永远新增）
    """
    write_latest_signal(latest_df, out_path="docs/signals/top10_latest.csv")
    td = _norm_ymd(trade_date)
    if td:
        write_latest_signal(latest_df, out_path=f"docs/signals/top10_{td}.csv")


def _write_weights(weights_df: pd.DataFrame, exec_date: str) -> Tuple[str, str]:
    """
    输出权重（主线）：
    - docs/weights/weights_latest.csv（覆盖）
    - docs/weights/weights_YYYYMMDD.csv（归档，exec_date 优先）
    返回：两个路径
    """
    Path("docs/weights").mkdir(parents=True, exist_ok=True)

    latest_path = "docs/weights/weights_latest.csv"
    dated = _norm_ymd(exec_date)
    dated_path = f"docs/weights/weights_{dated}.csv" if dated else "docs/weights/weights_unknown.csv"

    weights_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(dated_path, index=False, encoding="utf-8-sig")
    return latest_path, dated_path


def _write_candidates_snapshot(cand_df: pd.DataFrame, signal_date: str) -> str:
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    sd = _norm_ymd(signal_date) or "unknown"
    path = f"data/decision/decision_candidates_{sd}.csv"
    cand_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _ensure_execution_table(exec_date: str) -> str:
    """
    P0：先落空表，等聚宽回填真实执行结果
    """
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    ed = _norm_ymd(exec_date) or "unknown"
    path = f"data/decision/decision_execution_{ed}.csv"

    if not Path(path).exists():
        empty = pd.DataFrame(
            columns=[
                "exec_date",
                "ts_code",
                "jq_code",
                "filled_flag",
                "buy_time",
                "buy_price",
                "fail_reason",
                "buy_slippage_bp",
            ]
        )
        empty.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _ensure_learning_table() -> str:
    """
    P0：只建表。后续在聚宽回填 execution 后，再做执行一致标签并增量写入。
    """
    Path("data/decision").mkdir(parents=True, exist_ok=True)
    path = "data/decision/decision_learning.csv"
    if not Path(path).exists():
        empty = pd.DataFrame(
            columns=[
                "signal_date",
                "exec_date",
                "exit_date",
                "ts_code",
                "jq_code",
                "filled_flag",
                "buy_price",
                "sell_price",
                "ret_exec",
                # features snapshot (minimal)
                "p_fill_pred",
                "e_ret_pred",
                "ev_pred",
            ]
        )
        empty.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_decision_report(exec_date: str, report_md: str) -> str:
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    ed = _norm_ymd(exec_date) or "unknown"
    path = f"outputs/decision/decision_report_{ed}.md"
    Path(path).write_text(report_md, encoding="utf-8")
    return path


def _write_eval_json(exec_date: str, payload: dict) -> str:
    Path("outputs/decision").mkdir(parents=True, exist_ok=True)
    ed = _norm_ymd(exec_date) or "unknown"
    path = f"outputs/decision/eval_{ed}.json"
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =========================
# Main
# =========================

def build_signal_df_for_joinquant(weights_df: pd.DataFrame, risk_budget: float, regime_name: str, trade_date: str, target_trade_date: str) -> pd.DataFrame:
    """
    把 weights_df 映射回旧版 joinquant 信号表格式（兼容你的现有聚宽策略读取）
    """
    _ensure_cols(weights_df, ["ts_code", "weight"])
    df = weights_df.copy()
    df["jq_code"] = df["ts_code"].apply(to_jq_code)
    df["trade_date"] = _norm_ymd(trade_date)
    df["target_trade_date"] = _norm_ymd(target_trade_date)
    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)
    df["reason"] = "P0_EV_weight"

    # allocator 期望列 target_weight
    df["target_weight"] = df["weight"].astype(float)

    # 保证列顺序
    return df[["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]].copy()


def main() -> int:
    _ensure_dirs()

    # 1) load pred (a-top10 latest 或手动覆盖后的 pred_top10_latest.csv)
    pred_df = load_latest_pred()
    _ensure_cols(pred_df, ["ts_code", "name"])

    # 2) regime / guardrails
    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    regime_name = str(getattr(reg, "regime", "RISK_ON"))
    risk_budget = float(getattr(reg, "risk_budget", 1.0))

    # 3) score router -> Full TopK (not only top10)
    topk = int(getattr(gr, "topk", TOPK_DEFAULT)) if hasattr(gr, "topk") else TOPK_DEFAULT
    routed_df = score_router(pred_df).head(max(10, topk)).copy()

    # trade_date / target_trade_date（来自 a-top10 输出，如果存在）
    trade_date = _get_first_value(routed_df, "trade_date")
    target_trade_date = _get_first_value(routed_df, "target_trade_date")

    exec_date = _choose_exec_date(trade_date, target_trade_date)
    exit_date = ""  # P0 先不推导 T+2（后续可接交易日历）

    # reports（兼容旧）
    dated_name = f"daily_{trade_date}.md" if trade_date else "daily_unknown.md"

    # ===== STOP 分支：仍然落盘空信号 + 报告（并且确保三张表存在）=====
    if getattr(gr, "stop_trading", False):
        stop_note = getattr(gr, "reason", "STOP_TRADING")

        empty_signal = pd.DataFrame(
            columns=["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]
        )
        _write_signals(empty_signal, trade_date=trade_date)

        _write_human_report(
            routed_df,
            out_path="docs/reports/daily_latest.md",
            title="Daily Decision Report (latest)",
            stop_note=stop_note,
        )
        _write_human_report(
            routed_df,
            out_path=f"docs/reports/{dated_name}",
            title=f"Daily Decision Report ({trade_date})" if trade_date else "Daily Decision Report (unknown)",
            stop_note=stop_note,
        )

        # 三张表也要存在（防止流程断裂）
        _ensure_execution_table(exec_date=exec_date)
        _ensure_learning_table()

        # 候选快照（即使停手也保留可追溯）
        cand_snapshot = routed_df.copy()
        cand_snapshot["signal_date"] = _norm_ymd(trade_date)
        cand_snapshot["exec_date"] = _norm_ymd(exec_date)
        cand_snapshot["exit_date"] = _norm_ymd(exit_date)
        cand_snapshot["p_fill_pred"] = 0.0
        cand_snapshot["e_ret_pred"] = 0.0
        cand_snapshot["cost_est"] = cost_estimate_rule()
        cand_snapshot["risk_penalty"] = risk_penalty_rule(regime_name)
        cand_snapshot["ev_pred"] = 0.0
        _write_candidates_snapshot(cand_snapshot, signal_date=trade_date)

        # 主线 weights 也落空（保持契约）
        weights_df = pd.DataFrame(columns=["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"])
        _write_weights(weights_df, exec_date=exec_date)

        report_md = (
            f"# Decision Report ({exec_date or 'unknown'})\n\n"
            f"- signal_date: **{trade_date or 'unknown'}**\n"
            f"- exec_date: **{exec_date or 'unknown'}**\n"
            f"- regime: **{regime_name}**\n\n"
            f"**停手：{stop_note}**\n"
        )
        _write_decision_report(exec_date, report_md)
        _write_eval_json(exec_date, {"exec_date": exec_date, "signal_date": trade_date, "stop_trading": True, "reason": stop_note})

        return 0

    # ===== 正常分支：P0 生成三件套 =====

    # 4) 规则模型预测
    routed_df = routed_df.copy()
    routed_df["signal_date"] = _norm_ymd(trade_date)
    routed_df["exec_date"] = _norm_ymd(exec_date)
    routed_df["exit_date"] = _norm_ymd(exit_date)

    routed_df["p_fill_pred"] = fill_model_rule(routed_df)
    routed_df["e_ret_pred"] = overnight_model_rule(routed_df, regime=regime_name)

    cost_est = cost_estimate_rule()
    risk_pen = risk_penalty_rule(regime_name)

    routed_df["cost_est"] = cost_est
    routed_df["risk_penalty"] = risk_pen
    routed_df["ev_pred"] = routed_df["p_fill_pred"].astype(float) * routed_df["e_ret_pred"].astype(float) - cost_est - risk_pen

    # 5) 候选快照落库（T 日）
    cand_cols = list(routed_df.columns)
    cand_path = _write_candidates_snapshot(routed_df[cand_cols].copy(), signal_date=trade_date)

    # 6) 选 TopN 权重（可成交约束 + EV）
    caps = WeightCaps(w_max=W_MAX_DEFAULT, theme_cap=THEME_CAP_DEFAULT, gross_cap=GROSS_CAP_DEFAULT)
    picked = build_weights_topn(routed_df, topn=TOPN_DEFAULT, caps=caps)

    if picked.empty:
        # 极端情况：没有票满足约束 -> 落空并写报告
        _ensure_execution_table(exec_date=exec_date)
        _ensure_learning_table()
        weights_df = pd.DataFrame(columns=["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"])
        _write_weights(weights_df, exec_date=exec_date)

        report_md = (
            f"# Decision Report ({exec_date or 'unknown'})\n\n"
            f"- signal_date: **{trade_date or 'unknown'}**\n"
            f"- exec_date: **{exec_date or 'unknown'}**\n"
            f"- regime: **{regime_name}**\n\n"
            f"候选已生成，但在 P0 约束下未选出可用 TopN。\n"
            f"- candidates_snapshot: `{cand_path}`\n"
        )
        _write_decision_report(exec_date, report_md)
        _write_eval_json(exec_date, {"exec_date": exec_date, "signal_date": trade_date, "picked": 0, "candidates_path": cand_path})
        return 0

    # 7) 输出 weights（主线）
    weights_df = picked.copy()
    weights_df["exec_date"] = _norm_ymd(exec_date)
    weights_out = weights_df[["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"]].copy()
    weights_latest_path, weights_dated_path = _write_weights(weights_out, exec_date=exec_date)

    # 8) 兼容旧 joinquant signals 输出
    # 这里用 weights 转成 target_weight
    signal_df = build_signal_df_for_joinquant(
        weights_df=weights_out,
        risk_budget=risk_budget,
        regime_name=regime_name,
        trade_date=trade_date,
        target_trade_date=target_trade_date,
    )
    _write_signals(signal_df, trade_date=trade_date)

    # 9) 旧版人类报告仍然写（用 routed_df 的 Top10）
    trade_date2 = trade_date
    dated_name2 = f"daily_{trade_date2}.md" if trade_date2 else dated_name

    _write_human_report(
        routed_df,
        out_path="docs/reports/daily_latest.md",
        title="Daily Decision Report (latest)",
        stop_note=None,
    )
    _write_human_report(
        routed_df,
        out_path=f"docs/reports/{dated_name2}",
        title=f"Daily Decision Report ({trade_date2})" if trade_date2 else "Daily Decision Report (unknown)",
        stop_note=None,
    )

    # 10) 确保 execution/learning 表存在（P0 空表/建表）
    exec_path = _ensure_execution_table(exec_date=exec_date)
    learning_path = _ensure_learning_table()

    # 11) 输出 decision_report + eval
    lines: List[str] = []
    lines.append(f"# Decision Report ({exec_date or 'unknown'})\n\n")
    lines.append(f"- signal_date: **{trade_date or 'unknown'}**\n")
    lines.append(f"- exec_date: **{exec_date or 'unknown'}**\n")
    lines.append(f"- regime: **{regime_name}**\n")
    lines.append(f"- risk_budget: **{_fmt_num(risk_budget, 4)}**\n\n")

    lines.append("## Artifacts\n\n")
    lines.append(f"- candidates_snapshot: `{cand_path}`\n")
    lines.append(f"- execution_table: `{exec_path}`\n")
    lines.append(f"- learning_table: `{learning_path}`\n")
    lines.append(f"- weights_latest: `{weights_latest_path}`\n")
    lines.append(f"- weights_dated: `{weights_dated_path}`\n\n")

    lines.append("## TopN by EV (P0)\n\n")
    lines.append("| rank | ts_code | name | weight | p_fill | e_ret | EV |\n")
    lines.append("|---:|---|---|---:|---:|---:|---:|\n")

    # 把 topn 的 p_fill/e_ret/ev 一并展示（从 routed_df 合并）
    show = weights_out.merge(
        routed_df[["ts_code", "p_fill_pred", "e_ret_pred", "ev_pred"]].copy(),
        on="ts_code",
        how="left",
    ).copy()

    show = show.sort_values("target_rank", ascending=True)
    for _, r in show.iterrows():
        lines.append(
            f"| {int(r.get('target_rank', 0))} | {r.get('ts_code','')} | {r.get('name','')} | "
            f"{_fmt_num(r.get('weight', 0.0), 6)} | {_fmt_num(r.get('p_fill_pred', ''), 4)} | "
            f"{_fmt_num(r.get('e_ret_pred', ''), 4)} | {_fmt_num(r.get('ev_pred', ''), 6)} |\n"
        )

    report_path = _write_decision_report(exec_date, "".join(lines))

    eval_payload = {
        "signal_date": trade_date,
        "exec_date": exec_date,
        "regime": regime_name,
        "risk_budget": risk_budget,
        "topk": int(len(routed_df)),
        "picked": int(len(weights_out)),
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
    _write_eval_json(exec_date, eval_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
