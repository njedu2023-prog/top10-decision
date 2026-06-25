#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Premium final executable decision layer.

This is a Premium-only post-processing layer. It keeps the existing Premium
ranking intact, then creates a separate executable buy/watch/reject view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BUY_ACTIONS = {"BUY_MARKET", "BUY_LIMIT", "SMALL_BUY"}


@dataclass(frozen=True)
class FinalDecisionStats:
    trade_date: str
    final_buy_count: int
    watch_count: int
    reject_count: int
    market_mode: str
    max_trade_count: int
    t1_weight_mode: str
    t1_rank_ic: float

    def as_dict(self) -> Dict[str, object]:
        return dict(
            trade_date=self.trade_date,
            final_buy_count=self.final_buy_count,
            watch_count=self.watch_count,
            reject_count=self.reject_count,
            market_mode=self.market_mode,
            max_trade_count=self.max_trade_count,
            t1_weight_mode=self.t1_weight_mode,
            t1_rank_ic=self.t1_rank_ic,
        )


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in names:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


def _num(df: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    col = _first_col(df, names)
    if not col:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _txt(df: pd.DataFrame, names: Sequence[str], default: str = "") -> pd.Series:
    col = _first_col(df, names)
    if not col:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].astype(str).fillna(default)


def _clean(v: object) -> str:
    s = str(v if v is not None else "").strip()
    return "" if s.lower() in {"nan", "none", "<na>", "nat"} else s


def _prob(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(~(x > 1.0), x / 100.0)
    return x.clip(0.0, 1.0)


def _score(s: pd.Series, default: float = 50.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(~(x <= 1.0), x * 100.0)
    return x.clip(0.0, 100.0).fillna(default) / 100.0


def _action(v: object) -> str:
    s = _clean(v).lower()
    if not s:
        return "WATCH_ONLY"
    if "放弃" in s or "reject" in s or "禁止" in s:
        return "REJECT"
    if "只观察" in s or "观察" in s or "watch" in s:
        return "WATCH_ONLY"
    if "市价" in s or "market" in s:
        return "BUY_MARKET"
    if "小仓" in s or "small" in s:
        return "SMALL_BUY"
    if "限价" in s or "limit" in s:
        return "BUY_LIMIT"
    return "WATCH_ONLY"


def _action_score(s: pd.Series) -> pd.Series:
    return s.map({"BUY_MARKET": 1.0, "BUY_LIMIT": 0.88, "SMALL_BUY": 0.66, "WATCH_ONLY": 0.18, "REJECT": 0.0}).fillna(0.1)


def _t1_rank_ic(history: Optional[pd.DataFrame]) -> float:
    if history is None or history.empty:
        return float("nan")
    score_col = _first_col(history, ["t1_continue_up_rate", "t1_continue_up_rate_rule", "t1_up_prob_model", "T+1延续上涨率"])
    ret_col = _first_col(history, ["t1_close_ret", "t1_ret", "t1_return", "real_premium_ret"])
    date_col = _first_col(history, ["d_trade_date", "trade_date", "base_date", "d_analysis_trade_date"])
    if not score_col or not ret_col:
        return float("nan")
    tmp = pd.DataFrame(
        {
            "score": pd.to_numeric(history[score_col], errors="coerce"),
            "ret": pd.to_numeric(history[ret_col], errors="coerce"),
            "date": history[date_col].astype(str) if date_col else "",
        }
    ).dropna(subset=["score", "ret"])
    if tmp.empty:
        return float("nan")
    if date_col:
        dates = sorted(tmp["date"].astype(str).unique().tolist())[-20:]
        tmp = tmp[tmp["date"].isin(dates)]
    cors: List[float] = []
    groups = tmp.groupby("date") if date_col else [("", tmp)]
    for _, g in groups:
        if len(g) < 3 or g["score"].nunique() < 2 or g["ret"].nunique() < 2:
            continue
        c = g["score"].rank().corr(g["ret"].rank())
        if pd.notna(c):
            cors.append(float(c))
    return float(np.mean(cors)) if cors else float("nan")


def _market(df: pd.DataFrame) -> Tuple[str, int, float]:
    emotion = _num(df, ["mkt_emotion_score"]).dropna()
    up_ratio = _num(df, ["mkt_up_ratio"]).dropna()
    v = float(emotion.iloc[0]) if len(emotion) else float("nan")
    if not np.isfinite(v):
        v = float(up_ratio.iloc[0]) if len(up_ratio) else float("nan")
    if not np.isfinite(v):
        return "NORMAL", 2, 0.70
    if v < 0.18:
        return "NO_TRADE", 0, 0.0
    if v < 0.30:
        return "DEFENSIVE", 1, 0.35
    if v < 0.48:
        return "CAUTION", 2, 0.55
    return "NORMAL", 3, 0.85


def build_final_decisions(
    df: pd.DataFrame,
    *,
    trade_date: str = "",
    history: Optional[pd.DataFrame] = None,
    max_buys: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FinalDecisionStats]:
    if df is None or df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, FinalDecisionStats(trade_date, 0, 0, 0, "NO_DATA", 0, "UNKNOWN", float("nan"))

    out = df.copy().reset_index(drop=True)
    market_mode, market_cap, position_budget = _market(out)
    max_buys = market_cap if max_buys is None else min(int(max_buys), market_cap)

    ic = _t1_rank_ic(history)
    if np.isfinite(ic) and ic < 0:
        t1_weight, t1_mode = 0.03, "FROZEN_NEGATIVE_IC"
    elif np.isfinite(ic) and ic < 0.05:
        t1_weight, t1_mode = 0.06, "LOW_IC"
    else:
        t1_weight, t1_mode = 0.12, "NORMAL"

    raw_action = _txt(out, ["T日建议买入方式", "t_buy_method", "T+1建议买入方式", "t1_buy_method"])
    out["trade_action_raw"] = raw_action
    out["trade_action"] = raw_action.map(_action)

    t_up = _prob(_num(out, ["t_limitup_prob", "T日涨停概率"])).fillna(0.0)
    strength = _score(_num(out, ["t_limitup_strength", "T日涨停强度"]))
    attack = _score(_num(out, ["t_up_attack_score", "T-Attack"]))
    t1_up = _prob(_num(out, ["t1_continue_up_rate", "T+1延续上涨率", "t1_up_prob_model"])).fillna(0.0)
    t1_accept = _score(_num(out, ["t1_accept_score", "t1_accept_prob_model"]))
    exec_score = _action_score(out["trade_action"])

    dec_can_buy = _txt(out, ["dec_can_buy"]).str.strip().str.lower()
    dec_weight = _num(out, ["dec_weight"]).fillna(0.0)
    dec_rank = _num(out, ["dec_rank"])
    dec = pd.Series(0.50, index=out.index)
    dec = dec.where(~dec_can_buy.isin({"1", "true", "yes", "y", "是", "可买"}), 0.90)
    dec = dec.where(~(dec_weight > 0), 0.75)
    dec = dec.where(~(dec_rank <= 10), 0.80)
    dec = dec.where(~dec_can_buy.isin({"0", "false", "no", "n", "否", "不可买"}), 0.05)

    hard = _num(out, ["intraday_hard_risk_flag", "hard_risk_flag"], 0.0).fillna(0.0).clip(0, 1)
    penalty = _num(out, ["intraday_risk_penalty", "risk_penalty_score", "risk_penalty"], 0.0).fillna(0.0).clip(lower=0)
    fail = _prob(_num(out, ["t1_fail_prob_model"])).fillna(0.0)
    drawdown = _prob(_num(out, ["t1_big_drawdown_prob_model"])).fillna(0.0)
    risk = (1.0 - (0.45 * hard + 0.30 * penalty.clip(0, 1) + 0.15 * fail + 0.10 * drawdown)).clip(0, 1)

    base_weight = 1.0 - t1_weight
    out["final_trade_score"] = (
        base_weight * (0.34 * t_up + 0.15 * strength + 0.12 * attack + 0.15 * exec_score + 0.12 * dec + 0.12 * risk)
        + t1_weight * (0.55 * t1_up + 0.45 * t1_accept)
    ) * 100.0
    out["execution_score"] = exec_score * 100.0
    out["decision_support_score"] = dec * 100.0
    out["risk_quality_score"] = risk * 100.0
    out["t1_weight_mode"] = t1_mode
    out["market_mode"] = market_mode

    bucket = _txt(out, ["premium_bucket"]).str.upper()
    exclude = _txt(out, ["premium_exclude_reason"]).str.strip()
    forced = _num(out, ["premium_force_excluded"], 0.0).fillna(0.0)

    final_actions: List[str] = []
    reasons: List[str] = []
    for i, row in out.iterrows():
        act = str(row["trade_action"])
        r: List[str] = []
        if market_mode == "NO_TRADE":
            r.append("market_no_trade")
        if act == "REJECT":
            r.append("premium_reject")
        if act == "WATCH_ONLY":
            r.append("premium_watch_only")
        if bucket.iloc[i] == "EXCLUDED" or forced.iloc[i] >= 1:
            r.append("premium_excluded")
        er = _clean(exclude.iloc[i])
        if er and er.lower() not in {"ok", "normal", "none", "-"}:
            r.append(f"premium_gate:{er}")
        if dec_can_buy.iloc[i] in {"0", "false", "no", "n", "否", "不可买"}:
            r.append("decision_veto")
        if hard.iloc[i] >= 1:
            r.append("minute_hard_risk")
        if penalty.iloc[i] >= 0.45:
            r.append("minute_risk_penalty")
        if fail.iloc[i] >= 0.60:
            r.append("t1_fail_risk")
        if drawdown.iloc[i] >= 0.45:
            r.append("t1_drawdown_risk")

        hard_reject = {"market_no_trade", "premium_reject", "premium_excluded", "decision_veto", "minute_hard_risk"}
        if hard_reject.intersection(r):
            final_actions.append("REJECT")
        elif act in BUY_ACTIONS and not r:
            final_actions.append(act)
        else:
            final_actions.append("WATCH_ONLY")
        reasons.append(";".join(r) if r else "pass")

    out["final_action"] = final_actions
    out["final_reason"] = reasons
    buy = out[out["final_action"].isin(BUY_ACTIONS)].sort_values("final_trade_score", ascending=False).copy()
    watch = out[out["final_action"].eq("WATCH_ONLY")].sort_values("final_trade_score", ascending=False).copy()
    reject = out[out["final_action"].eq("REJECT")].sort_values("final_trade_score", ascending=False).copy()

    if len(buy) > max_buys:
        overflow = buy.iloc[max_buys:].copy()
        overflow["final_action"] = "WATCH_ONLY"
        overflow["final_reason"] = overflow["final_reason"].where(overflow["final_reason"].ne("pass"), "market_cap_overflow")
        watch = pd.concat([overflow, watch], ignore_index=True).sort_values("final_trade_score", ascending=False)
        buy = buy.iloc[:max_buys].copy()

    buy, watch, reject = buy.reset_index(drop=True), watch.reset_index(drop=True), reject.reset_index(drop=True)
    if len(buy):
        pos = position_budget / max(1, len(buy))
        buy["suggested_position"] = np.where(buy["final_action"].eq("SMALL_BUY"), min(pos, 0.10), pos)
    else:
        buy["suggested_position"] = pd.Series(dtype=float)
    for frame in (buy, watch, reject):
        if len(frame):
            frame["final_rank"] = np.arange(1, len(frame) + 1)
            frame["final_trade_score"] = pd.to_numeric(frame["final_trade_score"], errors="coerce").round(4)

    stats = FinalDecisionStats(trade_date, len(buy), len(watch), len(reject), market_mode, int(max_buys), t1_mode, ic)
    return buy, watch, reject, stats


def final_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    col_map = [
        ("final_rank", "FinalRank"),
        ("rank", "PremiumRank"),
        ("ts_code", "Code"),
        ("name", "Name"),
        ("sector", "Sector"),
        ("final_action", "Action"),
        ("final_trade_score", "FinalScore"),
        ("suggested_position", "Position"),
        ("T日可接受买入价", "MaxBuyPrice"),
        ("t_max_buy_price", "MaxBuyPrice"),
        ("t_limitup_prob", "T-Up"),
        ("t_limitup_strength", "T-Strength"),
        ("t1_continue_up_rate", "T1-Up"),
        ("premium_bucket", "Bucket"),
        ("trade_action_raw", "PremiumAction"),
        ("final_reason", "Reason"),
        ("T+1卖出计划", "T+1 SellPlan"),
        ("t1_sell_plan", "T+1 SellPlan"),
    ]
    out = pd.DataFrame()
    used = set()
    for src, dst in col_map:
        if src in df.columns and dst not in used:
            out[dst] = df[src]
            used.add(dst)
    return out


__all__ = ["BUY_ACTIONS", "FinalDecisionStats", "build_final_decisions", "final_display_columns"]
