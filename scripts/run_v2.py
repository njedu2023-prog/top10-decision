#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10-decision — V2 P0 runner (Professional Skeleton)

P0.1 关键增强：
- weights 输出加入“候补递补序列”：
  - 目标行：weight>0, target_rank=1..TopN
  - 候补行：weight=0, backup_rank=1..(TopK-TopN)
- 兼容旧 joinquant signals：只输出 weight>0 的目标行，避免影响现有聚宽策略

【本次修复：数据来源路径（PRED source path/url）】
- 支持从环境变量读取数据源：
  1) TOP10_PRED_URL  : 直接拉取远端 CSV（GitHub Raw）
  2) TOP10_PRED_PATH : 读取本地路径 CSV
  3) fallback        : 调用原 load_latest_pred()
- 自动兼容 a-top10 decisio 输出字段：
  - target_trade_date <- verify_date
  - Probability <- prob
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from top10decision.ingest import load_latest_pred
from top10decision.regime.simple_regime import simple_regime
from top10decision.risk.guardrails import guardrails
from top10decision.strategies.score_router import score_router
from top10decision.utils import to_jq_code
from top10decision.adapters.joinquant.write_latest_signal import write_latest_signal


# =========================
# Config (P0)
# =========================

TOPK_DEFAULT = 100
TOPN_DEFAULT = 10

W_MAX_DEFAULT = 0.12          # 单票上限
THEME_CAP_DEFAULT = 0.35      # 题材暴露上限
GROSS_CAP_DEFAULT = 1.00      # 总仓位上限

COST_BP_DEFAULT = 8.0         # 成本估计 bp
RISK_PENALTY_OFF = 0.00
RISK_PENALTY_ON = 0.02


# =========================
# Helpers
# =========================

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要字段：{miss}. 现有字段：{list(df.columns)}")


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


def _get_first_value(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    if col in ("trade_date", "target_trade_date", "exec_date", "exit_date", "signal_date", "verify_date"):
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
    td = _norm_ymd(trade_date)
    ttd = _norm_ymd(target_trade_date)
    return ttd or td


def _read_csv_any(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _download_to(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "top10-decision"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    dst.write_bytes(data)


def _load_pred_df() -> pd.DataFrame:
    """
    数据源优先级：
    1) TOP10_PRED_URL  : 远端 CSV（GitHub raw）
    2) TOP10_PRED_PATH : 本地 CSV
    3) fallback        : load_latest_pred()
    """
    url = (os.getenv("TOP10_PRED_URL") or "").strip()
    path = (os.getenv("TOP10_PRED_PATH") or "").strip()

    cache_path = Path("data/pred/pred_source_latest.csv")

    if url:
        print(f"[INGEST] use TOP10_PRED_URL={url}")
        _download_to(url, cache_path)
        df = _read_csv_any(cache_path)
        df.attrs["pred_source"] = f"url:{url}"
    elif path:
        p = Path(path)
        print(f"[INGEST] use TOP10_PRED_PATH={p}")
        df = _read_csv_any(p)
        # 同样缓存一份，便于验收
        try:
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        except Exception:
            pass
        df.attrs["pred_source"] = f"path:{p}"
    else:
        print("[INGEST] fallback to load_latest_pred()")
        df = load_latest_pred()
        df.attrs["pred_source"] = "fallback:load_latest_pred"

    if df is None:
        df = pd.DataFrame()
    return df


def _normalize_pred_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容 a-top10 decisio 输出字段（pred_decisio_latest.csv）：
    - target_trade_date <- verify_date
    - Probability <- prob
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # 必要字段兜底
    if "ts_code" not in d.columns:
        # 尝试兼容 code
        if "code" in d.columns:
            d["ts_code"] = d["code"]
    if "name" not in d.columns:
        if "stock_name" in d.columns:
            d["name"] = d["stock_name"]

    # 交易日字段
    if "target_trade_date" not in d.columns:
        if "verify_date" in d.columns:
            d["target_trade_date"] = d["verify_date"]
        else:
            d["target_trade_date"] = ""

    if "trade_date" not in d.columns:
        # 极端兜底
        d["trade_date"] = ""

    # 概率字段
    if "Probability" not in d.columns:
        if "prob" in d.columns:
            d["Probability"] = d["prob"]

    # 保证存在，避免后续逻辑空引用（值允许为空）
    for c in ("prob", "StrengthScore", "ThemeBoost", "board"):
        if c not in d.columns:
            d[c] = ""

    return d


# =========================
# P0 Models (rule-based)
# =========================

def fill_model_rule(df: pd.DataFrame) -> pd.Series:
    base = 0.35
    open_times = df.get("open_times", pd.Series([None] * len(df)))
    seal_amount = df.get("seal_amount", pd.Series([None] * len(df)))
    turnover = df.get("turnover_rate", pd.Series([None] * len(df)))

    p = []
    for i in range(len(df)):
        ot = _safe_float(open_times.iloc[i], default=float("nan"))
        sa = _safe_float(seal_amount.iloc[i], default=float("nan"))
        tr = _safe_float(turnover.iloc[i], default=float("nan"))

        pi = base
        if not pd.isna(ot):
            pi += min(max(ot, 0.0), 5.0) * 0.06
        if not pd.isna(sa):
            pi -= min(sa / 1e8, 5.0) * 0.05
        if not pd.isna(tr):
            pi += min(max(tr, 0.0), 20.0) * 0.005

        pi = max(0.02, min(0.98, pi))
        p.append(pi)

    return pd.Series(p, index=df.index, name="p_fill_pred")


def overnight_model_rule(df: pd.DataFrame, regime: str) -> pd.Series:
    prob = df.get("Probability", df.get("prob", df.get("probability", pd.Series([None] * len(df)))))
    strength = df.get("StrengthScore", df.get("strength", pd.Series([None] * len(df))))
    theme = df.get("ThemeBoost", df.get("theme", pd.Series([None] * len(df))))

    e = []
    for i in range(len(df)):
        p = _safe_float(prob.iloc[i], default=0.3)
        s = _safe_float(strength.iloc[i], default=0.0)
        t = _safe_float(theme.iloc[i], default=0.0)

        ei = (max(0.0, min(1.0, p)) - 0.2) * 0.03
        ei += max(-2.0, min(10.0, s)) * 0.001
        ei += max(-1.0, min(3.0, t)) * 0.003

        if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
            ei -= 0.006

        ei = max(-0.05, min(0.08, ei))
        e.append(ei)

    return pd.Series(e, index=df.index, name="e_ret_pred")


def risk_penalty_rule(regime: str) -> float:
    if str(regime).upper().strip() in ("RISK_OFF", "OFF", "DEFENSE"):
        return float(RISK_PENALTY_ON)
    return float(RISK_PENALTY_OFF)


def cost_estimate_rule() -> float:
    return float(COST_BP_DEFAULT) / 10000.0


# =========================
# Weight Engine (P0.1)
# =========================

@dataclass
class WeightCaps:
    w_max: float = W_MAX_DEFAULT
    theme_cap: float = THEME_CAP_DEFAULT
    gross_cap: float = GROSS_CAP_DEFAULT


def build_weights_with_backups(
    candidates: pd.DataFrame,
    topn: int,
    caps: WeightCaps,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
    - targets：TopN 目标（weight>0, target_rank）
    - backups：候补池（weight=0, backup_rank）
    """
    _ensure_cols(candidates, ["ts_code", "name", "ev_pred"])

    df = candidates.sort_values("ev_pred", ascending=False).reset_index(drop=True).copy()
    df["theme"] = df.apply(_pick_theme, axis=1)

    picked_idx = []
    theme_used: Dict[str, float] = {}
    gross_used = 0.0

    if topn <= 0:
        topn = TOPN_DEFAULT
    base_w = min(caps.gross_cap, 1.0) / float(topn)

    for i in range(len(df)):
        if len(picked_idx) >= topn:
            break

        th = df.loc[i, "theme"] or ""
        w = min(base_w, caps.w_max)

        if th:
            used = theme_used.get(th, 0.0)
            if used + w > caps.theme_cap:
                continue

        if gross_used + w > caps.gross_cap + 1e-9:
            break

        picked_idx.append(i)
        gross_used += w
        if th:
            theme_used[th] = theme_used.get(th, 0.0) + w

    targets = df.loc[picked_idx].copy()
    if targets.empty:
        backups = df.copy()
        return targets, backups

    targets["weight"] = min(base_w, caps.w_max)
    targets["target_rank"] = list(range(1, len(targets) + 1))
    targets["backup_rank"] = ""

    # P0.1：候补 = 除 targets 外的剩余 EV 序列
    rest = df.drop(index=picked_idx).reset_index(drop=True).copy()
    rest["weight"] = 0.0
    rest["target_rank"] = ""
    rest["backup_rank"] = list(range(1, len(rest) + 1))

    return targets.reset_index(drop=True), rest


# =========================
# Writers
# =========================

def _write_signals(latest_df: pd.DataFrame, trade_date: str) -> None:
    write_latest_signal(latest_df, out_path="docs/signals/top10_latest.csv")
    td = _norm_ymd(trade_date)
    if td:
        write_latest_signal(latest_df, out_path=f"docs/signals/top10_{td}.csv")


def _write_weights(weights_df: pd.DataFrame, exec_date: str) -> Tuple[str, str]:
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


def build_signal_df_for_joinquant(
    weights_df: pd.DataFrame,
    risk_budget: float,
    regime_name: str,
    trade_date: str,
    target_trade_date: str,
) -> pd.DataFrame:
    """
    兼容旧 joinquant 信号格式：
    注意：只输出 weight>0 的目标行（P0.1 关键），候补行不会进入 signals
    """
    _ensure_cols(weights_df, ["ts_code", "weight"])
    df = weights_df.copy()
    df = df[df["weight"].astype(float) > 0].copy()

    df["jq_code"] = df["ts_code"].apply(to_jq_code)
    df["trade_date"] = _norm_ymd(trade_date)
    df["target_trade_date"] = _norm_ymd(target_trade_date)
    df["risk_budget"] = float(risk_budget)
    df["regime"] = str(regime_name)
    df["reason"] = "P0_EV_weight"
    df["target_weight"] = df["weight"].astype(float)

    return df[["trade_date", "target_trade_date", "jq_code", "target_weight", "risk_budget", "regime", "reason"]].copy()


# =========================
# Main
# =========================

def main() -> int:
    _ensure_dirs()

    pred_df = _load_pred_df()
    pred_df = _normalize_pred_fields(pred_df)

    _ensure_cols(pred_df, ["ts_code", "name"])

    # 标记来源，写入日志，便于你验收
    src = getattr(pred_df, "attrs", {}).get("pred_source", "")
    if src:
        print(f"[INGEST] pred_source={src}")

    reg = simple_regime(pred_df)
    gr = guardrails(pred_df)

    regime_name = str(getattr(reg, "regime", "RISK_ON"))
    risk_budget = float(getattr(reg, "risk_budget", 1.0))

    topk = int(getattr(gr, "topk", TOPK_DEFAULT)) if hasattr(gr, "topk") else TOPK_DEFAULT
    routed_df = score_router(pred_df).head(max(10, topk)).copy()

    trade_date = _get_first_value(routed_df, "trade_date")
    target_trade_date = _get_first_value(routed_df, "target_trade_date")

    exec_date = _choose_exec_date(trade_date, target_trade_date)
    exit_date = ""

    # STOP 分支：保留原有逻辑（略）
    if getattr(gr, "stop_trading", False):
        stop_note = getattr(gr, "reason", "STOP_TRADING")
        _ensure_execution_table(exec_date=exec_date)
        _ensure_learning_table()

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

        weights_df = pd.DataFrame(columns=["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"])
        _write_weights(weights_df, exec_date=exec_date)

        _write_decision_report(exec_date, f"# Decision Report ({exec_date or 'unknown'})\n\n**停手：{stop_note}**\n")
        _write_eval_json(exec_date, {"exec_date": exec_date, "signal_date": trade_date, "stop_trading": True, "reason": stop_note})
        return 0

    # ===== 正常分支：P0.1 =====
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

    cand_path = _write_candidates_snapshot(routed_df.copy(), signal_date=trade_date)

    caps = WeightCaps(w_max=W_MAX_DEFAULT, theme_cap=THEME_CAP_DEFAULT, gross_cap=GROSS_CAP_DEFAULT)
    targets, backups = build_weights_with_backups(routed_df, topn=TOPN_DEFAULT, caps=caps)

    # 输出 weights：目标 + 候补（同一文件，候补 weight=0）
    weights_out = pd.concat([targets, backups], ignore_index=True)
    weights_out["exec_date"] = _norm_ymd(exec_date)

    weights_out = weights_out[["exec_date", "ts_code", "name", "weight", "target_rank", "backup_rank", "ev_pred"]].copy()
    weights_latest_path, weights_dated_path = _write_weights(weights_out, exec_date=exec_date)

    # signals：只输出目标（weight>0）
    signal_df = build_signal_df_for_joinquant(
        weights_df=weights_out,
        risk_budget=risk_budget,
        regime_name=regime_name,
        trade_date=trade_date,
        target_trade_date=target_trade_date,
    )
    _write_signals(signal_df, trade_date=trade_date)

    exec_path = _ensure_execution_table(exec_date=exec_date)
    learning_path = _ensure_learning_table()

    # decision report
    top_targets = weights_out[weights_out["weight"].astype(float) > 0].copy().sort_values("target_rank")
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

    lines.append("## TopN Targets\n\n")
    lines.append("| rank | ts_code | name | weight | EV |\n")
    lines.append("|---:|---|---|---:|---:|\n")
    for _, r in top_targets.iterrows():
        lines.append(
            f"| {int(r.get('target_rank', 0))} | {r.get('ts_code','')} | {r.get('name','')} | "
            f"{_fmt_num(r.get('weight', 0.0), 6)} | {_fmt_num(r.get('ev_pred', ''), 6)} |\n"
        )

    report_path = _write_decision_report(exec_date, "".join(lines))

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
    _write_eval_json(exec_date, eval_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
