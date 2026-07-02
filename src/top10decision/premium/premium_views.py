#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium views and validation helpers.

This module keeps human-facing Top10/Top20 tables and post-trade validation
out of the core scoring path. The trading date sequence is still produced by
predict.py through the strict A-share calendar.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LimitupValidationStats:
    ready: bool
    reason: str
    top10_total: int = 0
    top10_hits: int = 0
    top10_hit_rate: float = float("nan")
    top20_total: int = 0
    top20_hits: int = 0
    top20_hit_rate: float = float("nan")

    def as_dict(self) -> Dict[str, object]:
        return {
            "limitup_validation_ready": self.ready,
            "limitup_validation_reason": self.reason,
            "top10_limitup_total": self.top10_total,
            "top10_limitup_hits": self.top10_hits,
            "top10_limitup_hit_rate": (
                "" if not np.isfinite(self.top10_hit_rate) else round(float(self.top10_hit_rate), 6)
            ),
            "top20_limitup_total": self.top20_total,
            "top20_limitup_hits": self.top20_hits,
            "top20_limitup_hit_rate": (
                "" if not np.isfinite(self.top20_hit_rate) else round(float(self.top20_hit_rate), 6)
            ),
        }


def _num(s: object) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(pd.Series(s), errors="coerce")


def _limit_rate_for_code(ts_code: object) -> float:
    s = str(ts_code or "").strip().upper()
    raw = s.split(".")[0] if "." in s else "".join(ch for ch in s if ch.isdigit())[-6:]
    suffix = s.split(".")[-1] if "." in s else ""
    if suffix == "BJ" or raw.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if raw.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def _fmt_pct(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


def _fmt_num(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _clean_text(x: object, default: str = "-") -> str:
    s = str(x if x is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "<na>", "nat"}:
        return default
    return s


def add_rank_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit Top10/Top20 group flags to Premium output tables."""
    if df is None or df.empty:
        return df
    out = df.copy()
    rank = _num(out.get("rank")).reindex(out.index)
    out["rank_group"] = np.select(
        [rank <= 10, rank <= 20],
        ["TOP10", "TOP20"],
        default="FULL",
    )
    out["is_top10"] = (rank <= 10).astype("Int64")
    out["is_top20"] = (rank <= 20).astype("Int64")
    out["榜单分组"] = out["rank_group"].map({"TOP10": "TOP10", "TOP20": "TOP20", "FULL": "Outside TOP20"})
    return out


def attach_limitup_validation(
    df_verify: pd.DataFrame,
    daily_t: Optional[pd.DataFrame],
    trade_date: str,
    buy_date: str,
) -> tuple[pd.DataFrame, LimitupValidationStats]:
    """
    Attach real T-day limit-up labels to the verify table.

    The close limit-up threshold is derived from D close and A-share board rules
    when a formal up-limit column is unavailable in the cached daily file.
    """
    if df_verify is None or df_verify.empty:
        return df_verify, LimitupValidationStats(False, "verify_table_empty")
    if daily_t is None or daily_t.empty:
        out = df_verify.copy()
        out["t_limitup_actual"] = pd.NA
        out["t_touch_limitup_actual"] = pd.NA
        out["t_limitup_verify_ready"] = 0
        out["t_limitup_verify_reason"] = "t_daily_not_ready"
        return out, LimitupValidationStats(False, "t_daily_not_ready")

    t = daily_t.copy()
    if "ts_code" not in t.columns:
        out = df_verify.copy()
        out["t_limitup_actual"] = pd.NA
        out["t_touch_limitup_actual"] = pd.NA
        out["t_limitup_verify_ready"] = 0
        out["t_limitup_verify_reason"] = "t_daily_missing_ts_code"
        return out, LimitupValidationStats(False, "t_daily_missing_ts_code")

    rename = {}
    for c in ("open", "high", "close"):
        if c in t.columns:
            rename[c] = f"{c}_T_actual"
    t = t.rename(columns=rename)
    keep = ["ts_code", *rename.values()]
    t = t[[c for c in keep if c in t.columns]].copy()

    out = df_verify.copy().merge(t, on="ts_code", how="left")
    d_close = _num(out.get("close_T")).reindex(out.index)
    t_open = _num(out.get("open_T_actual")).reindex(out.index)
    t_high = _num(out.get("high_T_actual")).reindex(out.index)
    t_close = _num(out.get("close_T_actual")).reindex(out.index)
    rates = out["ts_code"].map(_limit_rate_for_code).astype(float)
    limit_px = (d_close * (1.0 + rates)).round(2)

    ready = d_close.notna() & t_high.notna() & t_close.notna() & (d_close > 0)
    out["t_limit_price_est"] = limit_px
    out["t_open_ret"] = np.where(ready & t_open.notna(), t_open / d_close - 1.0, pd.NA)
    out["t_intraday_ret"] = np.where(ready, t_high / d_close - 1.0, pd.NA)
    out["t_close_ret"] = np.where(ready, t_close / d_close - 1.0, pd.NA)
    out["t_up_actual"] = np.where(ready, (t_close > d_close).astype(int), pd.NA)
    out["t_high_profit_hit"] = np.where(ready, (pd.to_numeric(out["t_intraday_ret"], errors="coerce") >= 0.02).astype(int), pd.NA)
    out["t_limitup_actual"] = np.where(ready, (t_close >= limit_px * 0.9985).astype(int), pd.NA)
    out["t_touch_limitup_actual"] = np.where(ready, (t_high >= limit_px * 0.9985).astype(int), pd.NA)
    out["t_limitup_verify_ready"] = ready.astype(int)
    out["t_limitup_verify_reason"] = np.where(ready, "ok", "missing_D_or_T_price")
    out["t_limitup_verify_trade_date"] = str(buy_date)
    out["d_analysis_trade_date"] = str(trade_date)

    stats = _limitup_stats(out)
    return out, stats


def _limitup_stats(df_verify: pd.DataFrame) -> LimitupValidationStats:
    if df_verify is None or df_verify.empty:
        return LimitupValidationStats(False, "verify_table_empty")
    if "t_limitup_actual" not in df_verify.columns:
        return LimitupValidationStats(False, "limitup_actual_missing")

    ready = _num(df_verify.get("t_limitup_verify_ready")).reindex(df_verify.index).fillna(0).astype(int).eq(1)
    actual = _num(df_verify.get("t_limitup_actual")).reindex(df_verify.index)
    rank = _num(df_verify.get("rank")).reindex(df_verify.index)

    def calc(n: int) -> tuple[int, int, float]:
        m = ready & (rank <= n) & actual.notna()
        total = int(m.sum())
        hits = int((actual[m] == 1).sum())
        rate = hits / total if total > 0 else float("nan")
        return total, hits, rate

    top10_total, top10_hits, top10_rate = calc(10)
    top20_total, top20_hits, top20_rate = calc(20)
    return LimitupValidationStats(
        ready=bool(top10_total > 0 or top20_total > 0),
        reason="ok" if (top10_total > 0 or top20_total > 0) else "no_ready_limitup_rows",
        top10_total=top10_total,
        top10_hits=top10_hits,
        top10_hit_rate=top10_rate,
        top20_total=top20_total,
        top20_hits=top20_hits,
        top20_hit_rate=top20_rate,
    )


def limitup_stats_from_verify(df_verify: pd.DataFrame) -> LimitupValidationStats:
    return _limitup_stats(df_verify)


def _html_escape(x: object) -> str:
    return html.escape(_clean_text(x, ""), quote=True)


def _stage_text(row: pd.Series) -> str:
    text = _clean_text(row.get("晋阶", row.get("advance_stage", row.get("stage", row.get("limit_stage", "")))), "")
    if text and text not in {"nan", "None", "<NA>"}:
        return text
    raw = row.get("limit_times", row.get("连板数", None))
    try:
        if pd.isna(raw):
            return "-"
        n = int(float(raw))
    except Exception:
        return "-"
    if n <= 0:
        return "-"
    return f"{n}→{n + 1}"


def _display_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    show = df.head(n).copy()
    rows: List[Dict[str, object]] = []
    for _, r in show.iterrows():
        rows.append(
            {
                "Rank": _clean_text(r.get("rank")),
                "Code": _clean_text(r.get("ts_code")),
                "Name": _clean_text(r.get("name")),
                "晋阶": _stage_text(r),
                "Sector": _clean_text(
                    r.get(
                        "sector",
                        r.get(
                            "板块",
                            r.get("board", r.get("industry", r.get("行业", r.get("所属板块", r.get("所属行业", "-"))))),
                        ),
                    ),
                    "-",
                ),
                "D Close": _fmt_num(r.get("close_T"), 2),
                "Bucket": _clean_text(r.get("premium_bucket"), "WATCH"),
                "T-Up": _fmt_pct(r.get("t_limitup_prob"), 2),
                "T-Strength": _fmt_num(r.get("t_limitup_strength"), 2),
                "T-Attack": _fmt_num(r.get("t_up_attack_score"), 2),
                "T1-Up": _fmt_pct(r.get("t1_continue_up_rate"), 2),
                "T1-Accept": _fmt_num(r.get("t1_accept_score"), 2),
                "T1-Relay": _fmt_num(r.get("limitup_continuation_score"), 2),
                "Score": _fmt_num(r.get("premium_final_score", r.get("premium_adaptive_score", r.get("自适应排序评分"))), 2),
                "Gate": _clean_text(r.get("premium_exclude_reason"), "ok"),
                "T Auction Action": _clean_text(
                    r.get("T日建议买入方式", r.get("T+1建议买入方式", r.get("t1_buy_method")))
                ),
                "Price": _clean_text(
                    r.get("T日可接受买入价", r.get("T+1可接受买入价", r.get("t1_max_buy_price")))
                ),
                "T+1 Sell Plan": _clean_text(r.get("T+1卖出计划", r.get("t1_sell_plan"))),
            }
        )
    return pd.DataFrame(rows)


def _score_class(x: object, good_at: float, mid_at: float) -> str:
    try:
        raw = str(x).strip().replace("%", "")
        v = float(raw)
        if "%" in str(x):
            v = v / 100.0
        if v >= good_at:
            return "good"
        if v >= mid_at:
            return "mid"
    except Exception:
        pass
    return "quiet"


def _table_html(df: pd.DataFrame, table_id: str = "") -> str:
    if df is None or df.empty:
        return '<p class="empty">No data available</p>'
    head = "".join(f"<th>{_html_escape(c)}</th>" for c in df.columns)
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            val = r.get(c, "")
            cls = ""
            if c == "T-Up":
                cls = f' class="num {_score_class(val, 0.70, 0.45)}"'
            elif c == "T1-Up":
                cls = f' class="num {_score_class(val, 0.70, 0.50)}"'
            elif c in {"T-Strength", "T-Attack", "T1-Accept", "T1-Relay", "Score"}:
                cls = f' class="num {_score_class(val, 70.0, 55.0)}"'
            elif c in {"Rank", "D Close"}:
                cls = ' class="num"'
            cells.append(f"<td{cls}>{_html_escape(val)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    tid = f' id="{_html_escape(table_id)}"' if table_id else ""
    return f"<div class=\"table-wrap\"><table{tid}><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _metric_card(label: str, value: str, note: str = "") -> str:
    return (
        f"<div class=\"metric\"><span>{_html_escape(label)}</span>"
        f"<strong>{_html_escape(value)}</strong><small>{_html_escape(note)}</small></div>"
    )


def _rate_rows_card(title: str, rows: Sequence[Dict[str, object]]) -> str:
    parts = []
    for row in rows:
        label = _clean_text(row.get("label"))
        rate = _clean_text(row.get("rate"))
        hits = _clean_text(row.get("hits"))
        total = _clean_text(row.get("total"))
        note = _clean_text(row.get("note"), "")
        suffix = f"/{_html_escape(total)}" if total else ""
        note_html = f"<small>{_html_escape(hits)}{suffix}{(', ' + _html_escape(note)) if note else ''}</small>"
        parts.append(
            '<div class="metric-line">'
            f'<span>{_html_escape(label)}</span>'
            f'<strong>{_html_escape(rate)}</strong>'
            f"{note_html}</div>"
        )
    return f'<div class="metric metric-wide"><span>{_html_escape(title)}</span><div class="metric-lines">{"".join(parts)}</div></div>'


def _link_button(label: str, href: str, enabled: bool, kind: str = "") -> str:
    cls = "nav-btn"
    if kind:
        cls += f" {kind}"
    if not enabled:
        cls += " disabled"
        return f'<span class="{cls}">{_html_escape(label)}</span>'
    return f'<a class="{cls}" href="{_html_escape(href)}">{_html_escape(label)}</a>'


def _report_nav_html(trade_date: str, report_dates: Optional[Sequence[str]]) -> str:
    dates = sorted({str(x) for x in (report_dates or []) if str(x).isdigit() and len(str(x)) == 8})
    if str(trade_date).isdigit() and str(trade_date) not in dates:
        dates.append(str(trade_date))
        dates = sorted(dates)
    if not dates:
        return ""

    try:
        idx = dates.index(str(trade_date))
    except ValueError:
        idx = len(dates) - 1

    prev_date = dates[idx - 1] if idx > 0 else ""
    next_date = dates[idx + 1] if idx + 1 < len(dates) else ""
    recent = dates[-6:]
    chips = "".join(
        f'<a class="date-chip{" active" if d == str(trade_date) else ""}" href="premium_{_html_escape(d)}.html">{_html_escape(d)}</a>'
        for d in recent
    )
    return f"""
      <nav class="report-nav" aria-label="Report navigation">
        <div class="nav-actions">
          {_link_button("Previous Report", f"premium_{prev_date}.html", bool(prev_date))}
          {_link_button("Latest Report", "premium_latest.html", True, "primary")}
          {_link_button("Next Report", f"premium_{next_date}.html", bool(next_date))}
        </div>
        <div class="date-chips">{chips}</div>
      </nav>
    """


def render_premium_report_html(
    trade_date: str,
    buy_date: str,
    target_date: str,
    df_top: pd.DataFrame,
    df_verify: pd.DataFrame,
    verify_pending: bool,
    verify_reason: str,
    gen_ts: str,
    model_version: str,
    audit_notes: Optional[Iterable[str]] = None,
    report_dates: Optional[Sequence[str]] = None,
    historical_limitup_stats: Optional[Dict[str, object]] = None,
) -> str:
    """Render the human-friendly Premium HTML report."""
    stats = limitup_stats_from_verify(df_verify)
    top10 = _display_table(df_top, 10)
    top20 = _display_table(df_top, 20)
    hist = historical_limitup_stats or {}
    hist_ready = bool(hist.get("ready", False))
    hist_top1_rate = pd.to_numeric(pd.Series([hist.get("top1_hit_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top1_hits = int(hist.get("top1_hits", 0) or 0)
    hist_top1_total = int(hist.get("top1_total", 0) or 0)
    hist_top3_rate = pd.to_numeric(pd.Series([hist.get("top3_hit_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top3_hits = int(hist.get("top3_hits", 0) or 0)
    hist_top3_total = int(hist.get("top3_total", 0) or 0)
    hist_top5_rate = pd.to_numeric(pd.Series([hist.get("top5_hit_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top5_hits = int(hist.get("top5_hits", 0) or 0)
    hist_top5_total = int(hist.get("top5_total", 0) or 0)
    hist_top1_up_rate = pd.to_numeric(pd.Series([hist.get("top1_up_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top1_up_hits = int(hist.get("top1_up_hits", 0) or 0)
    hist_top1_up_total = int(hist.get("top1_up_total", 0) or 0)
    hist_top3_up_rate = pd.to_numeric(pd.Series([hist.get("top3_up_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top3_up_hits = int(hist.get("top3_up_hits", 0) or 0)
    hist_top3_up_total = int(hist.get("top3_up_total", 0) or 0)
    hist_top5_up_rate = pd.to_numeric(pd.Series([hist.get("top5_up_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top5_up_hits = int(hist.get("top5_up_hits", 0) or 0)
    hist_top5_up_total = int(hist.get("top5_up_total", 0) or 0)
    hist_top10_rate = pd.to_numeric(pd.Series([hist.get("top10_hit_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top10_hits = int(hist.get("top10_hits", 0) or 0)
    hist_top10_total = int(hist.get("top10_total", 0) or 0)
    hist_top20_rate = pd.to_numeric(pd.Series([hist.get("top20_hit_rate", np.nan)]), errors="coerce").iloc[0]
    hist_top20_hits = int(hist.get("top20_hits", 0) or 0)
    hist_top20_total = int(hist.get("top20_total", 0) or 0)
    hist_days = int(hist.get("n_days", 0) or 0)
    hist_source = _clean_text(hist.get("source"), "-")
    hist_reason = _clean_text(hist.get("reason"), "-")
    hist_5d = pd.to_numeric(pd.Series([hist.get("top10_hit_rate_5d", np.nan)]), errors="coerce").iloc[0]
    hist_20d = pd.to_numeric(pd.Series([hist.get("top10_hit_rate_20d", np.nan)]), errors="coerce").iloc[0]
    hist_60d = pd.to_numeric(pd.Series([hist.get("top10_hit_rate_60d", np.nan)]), errors="coerce").iloc[0]
    cal_brier = pd.to_numeric(pd.Series([hist.get("calibration_brier", np.nan)]), errors="coerce").iloc[0]
    cal_ece = pd.to_numeric(pd.Series([hist.get("calibration_ece", np.nan)]), errors="coerce").iloc[0]
    cal_rows = int(hist.get("calibration_rows", 0) or 0)
    limitup_ic = pd.to_numeric(pd.Series([hist.get("limitup_spearman_ic_mean", np.nan)]), errors="coerce").iloc[0]
    limitup_ic_20d = pd.to_numeric(pd.Series([hist.get("limitup_spearman_ic_20d", np.nan)]), errors="coerce").iloc[0]
    limitup_ic_pos = pd.to_numeric(pd.Series([hist.get("limitup_spearman_ic_positive_rate", np.nan)]), errors="coerce").iloc[0]
    limitup_tau = pd.to_numeric(pd.Series([hist.get("limitup_kendall_tau_mean", np.nan)]), errors="coerce").iloc[0]
    limitup_ic_days = int(hist.get("limitup_ic_days", 0) or 0)
    t1_ret_ic = pd.to_numeric(pd.Series([hist.get("t1_ret_spearman_ic_mean", np.nan)]), errors="coerce").iloc[0]
    t1_ret_ic_20d = pd.to_numeric(pd.Series([hist.get("t1_ret_spearman_ic_20d", np.nan)]), errors="coerce").iloc[0]
    t1_ret_ic_pos = pd.to_numeric(pd.Series([hist.get("t1_ret_spearman_ic_positive_rate", np.nan)]), errors="coerce").iloc[0]
    t1_ret_ic_days = int(hist.get("t1_ret_ic_days", 0) or 0)
    tier_top10_rate = pd.to_numeric(pd.Series([hist.get("tier_top10_hit_rate", np.nan)]), errors="coerce").iloc[0]
    tier_11_20_rate = pd.to_numeric(pd.Series([hist.get("tier_top20_tail_hit_rate", np.nan)]), errors="coerce").iloc[0]
    tier_spread = pd.to_numeric(pd.Series([hist.get("tier_top10_vs_11_20_hit_spread", np.nan)]), errors="coerce").iloc[0]
    tier_summary = _clean_text(hist.get("tier_summary"), "-")
    mkt_emotion = pd.to_numeric(df_top.get("mkt_emotion_score", pd.Series([np.nan])), errors="coerce").dropna()
    mkt_up = pd.to_numeric(df_top.get("mkt_up_ratio", pd.Series([np.nan])), errors="coerce").dropna()
    mkt_strong = pd.to_numeric(df_top.get("mkt_strong_count", pd.Series([np.nan])), errors="coerce").dropna()
    mkt_emotion_v = float(mkt_emotion.iloc[0]) if len(mkt_emotion) else float("nan")
    mkt_up_v = float(mkt_up.iloc[0]) if len(mkt_up) else float("nan")
    mkt_strong_v = int(mkt_strong.iloc[0]) if len(mkt_strong) else 0
    w_limitup = pd.to_numeric(df_top.get("adaptive_weight_limitup", pd.Series([np.nan])), errors="coerce").dropna()
    w_t1 = pd.to_numeric(df_top.get("adaptive_weight_t1", pd.Series([np.nan])), errors="coerce").dropna()
    w_strength = pd.to_numeric(df_top.get("adaptive_weight_strength", pd.Series([np.nan])), errors="coerce").dropna()
    w_exec = pd.to_numeric(df_top.get("adaptive_weight_execution", pd.Series([np.nan])), errors="coerce").dropna()
    w_limitup_v = float(w_limitup.iloc[0]) if len(w_limitup) else float("nan")
    w_t1_v = float(w_t1.iloc[0]) if len(w_t1) else float("nan")
    w_strength_v = float(w_strength.iloc[0]) if len(w_strength) else float("nan")
    w_exec_v = float(w_exec.iloc[0]) if len(w_exec) else float("nan")
    buckets = df_top.get("premium_bucket", pd.Series([], dtype="object")).astype(str) if df_top is not None else pd.Series([], dtype="object")
    bucket_eligible = int((buckets == "ELIGIBLE").sum()) if len(buckets) else 0
    bucket_watch = int((buckets == "WATCH").sum()) if len(buckets) else 0
    bucket_excluded = int((buckets == "EXCLUDED").sum()) if len(buckets) else 0
    rank_mode_s = df_top.get("premium_rank_mode", pd.Series([], dtype="object")).dropna() if df_top is not None else pd.Series([], dtype="object")
    rank_mode = _clean_text(rank_mode_s.iloc[0] if len(rank_mode_s) else "-", "-")
    model_mode_s = df_top.get("model_rank_mode", pd.Series([], dtype="object")).dropna() if df_top is not None else pd.Series([], dtype="object")
    model_mode = _clean_text(model_mode_s.iloc[0] if len(model_mode_s) else "-", "-")

    cards = [
        _metric_card("D Analysis Date", str(trade_date), "Uses post-close data from D only"),
        _metric_card("T Auction Buy Date", str(buy_date), "Strict A-share trading calendar"),
        _metric_card("T+1 Timing Exit Date", str(target_date), "Continuation and validation date"),
        _metric_card(
            "Current TOP10 Hit Rate",
            "-" if not stats.ready or not np.isfinite(stats.top10_hit_rate) else _fmt_pct(stats.top10_hit_rate),
            f"{stats.top10_hits}/{stats.top10_total}, {stats.reason}",
        ),
        _metric_card(
            "Historical TOP10 Limit-up Hit Rate",
            "-" if not hist_ready or not np.isfinite(hist_top10_rate) else _fmt_pct(hist_top10_rate),
            f"{hist_top10_hits}/{hist_top10_total}, trading days {hist_days}, {hist_source}",
        ),
        _rate_rows_card(
            "Head Historical Limit-up Hit Rate",
            [
                {"label": "TOP1 Historical Limit-up Hit Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top1_rate) else _fmt_pct(hist_top1_rate), "hits": hist_top1_hits, "total": hist_top1_total},
                {"label": "TOP3 Historical Limit-up Hit Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top3_rate) else _fmt_pct(hist_top3_rate), "hits": hist_top3_hits, "total": hist_top3_total},
                {"label": "TOP5 Historical Limit-up Hit Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top5_rate) else _fmt_pct(hist_top5_rate), "hits": hist_top5_hits, "total": hist_top5_total},
            ],
        ),
        _rate_rows_card(
            "Head Historical Up Rate",
            [
                {"label": "TOP1 Historical Up Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top1_up_rate) else _fmt_pct(hist_top1_up_rate), "hits": hist_top1_up_hits, "total": hist_top1_up_total, "note": "T close > D close"},
                {"label": "TOP3 Historical Up Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top3_up_rate) else _fmt_pct(hist_top3_up_rate), "hits": hist_top3_up_hits, "total": hist_top3_up_total, "note": "T close > D close"},
                {"label": "TOP5 Historical Up Rate", "rate": "-" if not hist_ready or not np.isfinite(hist_top5_up_rate) else _fmt_pct(hist_top5_up_rate), "hits": hist_top5_up_hits, "total": hist_top5_up_total, "note": "T close > D close"},
            ],
        ),
        _metric_card(
            "20D TOP10 Hit Rate",
            "-" if not hist_ready or not np.isfinite(hist_20d) else _fmt_pct(hist_20d),
            f"5D {_fmt_pct(hist_5d) if np.isfinite(hist_5d) else '-'}; 60D {_fmt_pct(hist_60d) if np.isfinite(hist_60d) else '-'}",
        ),
        _metric_card(
            "Probability Calibration Quality",
            "-" if not np.isfinite(cal_brier) else f"Brier {cal_brier:.4f}",
            f"ECE {_fmt_num(cal_ece, 4) if np.isfinite(cal_ece) else '-'}; samples {cal_rows}",
        ),
        _metric_card(
            "Limit-up Rank IC",
            "-" if not np.isfinite(limitup_ic) else _fmt_num(limitup_ic, 4),
            f"20D {_fmt_num(limitup_ic_20d, 4) if np.isfinite(limitup_ic_20d) else '-'}; positive rate {_fmt_pct(limitup_ic_pos) if np.isfinite(limitup_ic_pos) else '-'}; {limitup_ic_days} days",
        ),
        _metric_card(
            "T+1 Return Rank IC",
            "-" if not np.isfinite(t1_ret_ic) else _fmt_num(t1_ret_ic, 4),
            f"20D {_fmt_num(t1_ret_ic_20d, 4) if np.isfinite(t1_ret_ic_20d) else '-'}; positive rate {_fmt_pct(t1_ret_ic_pos) if np.isfinite(t1_ret_ic_pos) else '-'}; {t1_ret_ic_days} days",
        ),
        _metric_card(
            "Adaptive Ranking Weights",
            f"Limit-up {_fmt_pct(w_limitup_v) if np.isfinite(w_limitup_v) else '-'} / T+1 {_fmt_pct(w_t1_v) if np.isfinite(w_t1_v) else '-'}",
            f"Strength {_fmt_pct(w_strength_v) if np.isfinite(w_strength_v) else '-'}; Execution {_fmt_pct(w_exec_v) if np.isfinite(w_exec_v) else '-'}",
        ),
        _metric_card(
            "Professional Gate",
            f"{bucket_eligible} eligible / {bucket_watch} watch",
            f"Excluded {bucket_excluded}; rank {rank_mode}; model {model_mode}",
        ),
        _metric_card(
            "Tier Effectiveness",
            "-" if not np.isfinite(tier_spread) else _fmt_pct(tier_spread),
            f"TOP10 {_fmt_pct(tier_top10_rate) if np.isfinite(tier_top10_rate) else '-'}; 11-20 {_fmt_pct(tier_11_20_rate) if np.isfinite(tier_11_20_rate) else '-'}",
        ),
        _metric_card(
            "D Market Sentiment",
            "-" if not np.isfinite(mkt_emotion_v) else _fmt_pct(mkt_emotion_v),
            f"Up ratio {_fmt_pct(mkt_up_v) if np.isfinite(mkt_up_v) else '-'}; strong stocks {mkt_strong_v}",
        ),
    ]
    notes = "".join(f"<li>{_html_escape(x)}</li>" for x in (audit_notes or []))
    verify_badge = "PENDING" if verify_pending else "READY"
    nav = _report_nav_html(str(trade_date), report_dates)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Premium Limit-up Relay Forecast {html.escape(str(trade_date))}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink:#142033; --muted:#667085; --line:#d7dde7; --soft:#f5f7fb;
      --accent:#b42318; --accent2:#0f6b4f; --warn:#b7791f; --panel:#ffffff;
      --shadow:0 12px 32px rgba(20,32,51,.08);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; color:var(--ink); background:#eef2f6; }}
    header {{ padding:24px 28px 18px; background:#ffffff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }}
    .topbar {{ display:flex; align-items:flex-start; justify-content:space-between; gap:18px; max-width:1480px; margin:0 auto; }}
    .kicker {{ color:var(--accent); font-weight:700; font-size:13px; letter-spacing:0; }}
    h1 {{ margin:7px 0 0; font-size:24px; line-height:1.18; letter-spacing:0; }}
    .sub {{ margin:0; color:var(--muted); line-height:1.65; max-width:900px; }}
    .status-pill {{ display:inline-flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:999px; padding:8px 12px; color:var(--muted); font-size:13px; white-space:nowrap; background:#fff; }}
    .status-pill b {{ color:var(--ink); }}
    main {{ padding:18px 28px 36px; max-width:1480px; margin:0 auto; }}
    .report-nav {{ display:flex; align-items:center; justify-content:space-between; gap:14px; margin:0 0 14px; }}
    .nav-actions, .date-chips, .tabs {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .nav-btn, .date-chip, .tab-btn {{ border:1px solid var(--line); background:#fff; color:#334155; text-decoration:none; border-radius:8px; padding:8px 11px; font-size:13px; line-height:1; cursor:pointer; }}
    .nav-btn:hover, .date-chip:hover, .tab-btn:hover {{ border-color:#b6c0d0; background:#f8fafc; }}
    .nav-btn.primary, .date-chip.active, .tab-btn.active {{ border-color:#1f6f54; color:#0f5b43; background:#edf8f3; font-weight:700; }}
    .nav-btn.disabled {{ color:#a0a8b5; background:#f4f6f9; cursor:not-allowed; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; margin-bottom:16px; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; min-height:98px; box-shadow:var(--shadow); }}
    .metric-wide {{ grid-column:span 2; }}
    .metric span {{ display:block; color:var(--muted); font-size:13px; }}
    .metric strong {{ display:block; margin-top:8px; font-size:23px; line-height:1.2; }}
    .metric small {{ display:block; margin-top:8px; color:var(--muted); line-height:1.35; }}
    .metric-lines {{ margin-top:10px; display:grid; gap:8px; }}
    .metric-line {{ display:grid; grid-template-columns:minmax(0,1fr) auto auto; align-items:center; gap:10px; padding:8px 0; border-top:1px solid #edf0f5; }}
    .metric-line:first-child {{ border-top:0; padding-top:0; }}
    .metric-line span {{ color:#344054; font-weight:650; line-height:1.3; }}
    .metric-line strong {{ margin:0; font-size:20px; color:var(--accent); font-variant-numeric:tabular-nums; }}
    .metric-line small {{ margin:0; color:var(--muted); font-size:12px; white-space:nowrap; }}
    .toolbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; background:#fff; border:1px solid var(--line); border-radius:8px; margin-bottom:14px; }}
    .hint {{ color:var(--muted); font-size:13px; line-height:1.35; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-top:14px; overflow:hidden; box-shadow:var(--shadow); }}
    section.hidden {{ display:none; }}
    .section-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:16px 18px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
    .badge {{ border:1px solid var(--line); border-radius:999px; padding:5px 10px; font-size:12px; color:var(--muted); white-space:nowrap; background:#fff; }}
    .table-wrap {{ overflow:auto; width:100%; max-height:72vh; }}
    table {{ width:100%; border-collapse:collapse; min-width:1100px; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid #edf0f5; text-align:left; white-space:nowrap; font-size:13px; vertical-align:top; }}
    th {{ background:var(--soft); color:#384256; font-weight:700; position:sticky; top:0; z-index:2; }}
    tbody tr:nth-child(even) td {{ background:#fbfcfe; }}
    tbody tr:hover td {{ background:#fff8f2; }}
    th:first-child, td:first-child {{ position:sticky; left:0; z-index:1; background:inherit; }}
    th:first-child {{ z-index:3; }}
    .num {{ font-variant-numeric:tabular-nums; font-weight:650; }}
    .good {{ color:var(--accent); }}
    .mid {{ color:var(--warn); }}
    .quiet {{ color:#475467; }}
    .explain {{ padding:14px 18px; color:var(--muted); line-height:1.7; }}
    .explain ul {{ margin:8px 0 0; padding-left:18px; }}
    .empty {{ margin:0; padding:16px 18px; color:var(--muted); }}
    .footnote {{ color:var(--muted); font-size:12px; margin:14px 0 0; line-height:1.5; }}
    @media (max-width: 900px) {{
      header {{ position:static; }}
      header, main {{ padding-left:16px; padding-right:16px; }}
      .topbar, .report-nav, .toolbar {{ align-items:flex-start; flex-direction:column; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .metric-wide {{ grid-column:1 / -1; }}
      h1 {{ font-size:24px; }}
    }}
    @media (max-width: 560px) {{
      .metrics {{ grid-template-columns:1fr; }}
      .metric-line {{ grid-template-columns:1fr auto; }}
      .metric-line small {{ grid-column:1 / -1; }}
      .section-head {{ align-items:flex-start; flex-direction:column; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <div class="kicker">Premium V4 Quant Engine</div>
        <h1>Premium Limit-up Relay TOP10 / TOP20</h1>
      </div>
      <div class="status-pill">Validation <b>{_html_escape(verify_badge)}</b></div>
    </div>
  </header>
  <main>
    {nav}
    <div class="metrics">{''.join(cards)}</div>
    <div class="toolbar">
      <div class="tabs" role="tablist" aria-label="List switcher">
        <button class="tab-btn active" type="button" data-target="top10-panel">TOP10 Execution List</button>
        <button class="tab-btn" type="button" data-target="top20-panel">TOP20 Watch List</button>
        <button class="tab-btn" type="button" data-target="verify-panel">Validation & Learning</button>
      </div>
      <div class="hint">Tables scroll horizontally with the first column pinned; stronger colors indicate higher probability or score.</div>
    </div>
    <section id="top10-panel">
      <div class="section-head"><h2>TOP10: Highest T-day Limit-up Probability</h2><span class="badge">Core Execution List</span></div>
      {_table_html(top10, "top10-table")}
    </section>
    <section id="top20-panel" class="hidden">
      <div class="section-head"><h2>TOP20: T+1 Continuation Candidates</h2><span class="badge">Extended Watch List</span></div>
      {_table_html(top20, "top20-table")}
    </section>
    <section id="verify-panel" class="hidden">
      <div class="section-head"><h2>Validation & Learning</h2><span class="badge">{_html_escape(verify_badge)}</span></div>
      <div class="explain">
        <div>Validation status: {_html_escape(verify_reason)}</div>
        <div>Current TOP10 limit-up prediction hit rate: {_html_escape('-' if not stats.ready or not np.isfinite(stats.top10_hit_rate) else _fmt_pct(stats.top10_hit_rate))} ({stats.top10_hits}/{stats.top10_total})</div>
        <div>Current TOP20 limit-up prediction hit rate: {_html_escape('-' if not stats.ready or not np.isfinite(stats.top20_hit_rate) else _fmt_pct(stats.top20_hit_rate))} ({stats.top20_hits}/{stats.top20_total})</div>
        <div>Historical TOP1 / TOP3 / TOP5 cumulative limit-up hit rate: TOP1 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top1_rate) else _fmt_pct(hist_top1_rate))} ({hist_top1_hits}/{hist_top1_total}); TOP3 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top3_rate) else _fmt_pct(hist_top3_rate))} ({hist_top3_hits}/{hist_top3_total}); TOP5 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top5_rate) else _fmt_pct(hist_top5_rate))} ({hist_top5_hits}/{hist_top5_total})</div>
        <div>Historical TOP1 / TOP3 / TOP5 cumulative up rate: TOP1 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top1_up_rate) else _fmt_pct(hist_top1_up_rate))} ({hist_top1_up_hits}/{hist_top1_up_total}); TOP3 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top3_up_rate) else _fmt_pct(hist_top3_up_rate))} ({hist_top3_up_hits}/{hist_top3_up_total}); TOP5 {_html_escape('-' if not hist_ready or not np.isfinite(hist_top5_up_rate) else _fmt_pct(hist_top5_up_rate))} ({hist_top5_up_hits}/{hist_top5_up_total})</div>
        <div>Historical TOP10 cumulative limit-up hit rate: {_html_escape('-' if not hist_ready or not np.isfinite(hist_top10_rate) else _fmt_pct(hist_top10_rate))} ({hist_top10_hits}/{hist_top10_total})</div>
        <div>Historical TOP20 cumulative limit-up hit rate: {_html_escape('-' if not hist_ready or not np.isfinite(hist_top20_rate) else _fmt_pct(hist_top20_rate))} ({hist_top20_hits}/{hist_top20_total})</div>
        <div>Rolling TOP10 hit rate: 5D {_html_escape('-' if not np.isfinite(hist_5d) else _fmt_pct(hist_5d))}; 20D {_html_escape('-' if not np.isfinite(hist_20d) else _fmt_pct(hist_20d))}; 60D {_html_escape('-' if not np.isfinite(hist_60d) else _fmt_pct(hist_60d))}</div>
        <div>Probability calibration: Brier {_html_escape('-' if not np.isfinite(cal_brier) else f'{cal_brier:.4f}')}; ECE {_html_escape('-' if not np.isfinite(cal_ece) else f'{cal_ece:.4f}')}; samples {cal_rows}</div>
        <div>Limit-up Rank IC: Spearman mean {_html_escape('-' if not np.isfinite(limitup_ic) else f'{limitup_ic:.4f}')}; 20D {_html_escape('-' if not np.isfinite(limitup_ic_20d) else f'{limitup_ic_20d:.4f}')}; Kendall Tau {_html_escape('-' if not np.isfinite(limitup_tau) else f'{limitup_tau:.4f}')}; positive rate {_html_escape('-' if not np.isfinite(limitup_ic_pos) else _fmt_pct(limitup_ic_pos))}; valid days {limitup_ic_days}</div>
        <div>T+1 Return Rank IC: Spearman mean {_html_escape('-' if not np.isfinite(t1_ret_ic) else f'{t1_ret_ic:.4f}')}; 20D {_html_escape('-' if not np.isfinite(t1_ret_ic_20d) else f'{t1_ret_ic_20d:.4f}')}; positive rate {_html_escape('-' if not np.isfinite(t1_ret_ic_pos) else _fmt_pct(t1_ret_ic_pos))}; valid days {t1_ret_ic_days}</div>
        <div>Adaptive ranking weights: limit-up {_html_escape('-' if not np.isfinite(w_limitup_v) else _fmt_pct(w_limitup_v))}; T+1 {_html_escape('-' if not np.isfinite(w_t1_v) else _fmt_pct(w_t1_v))}; strength {_html_escape('-' if not np.isfinite(w_strength_v) else _fmt_pct(w_strength_v))}; execution {_html_escape('-' if not np.isfinite(w_exec_v) else _fmt_pct(w_exec_v))}. T+1 weight is reduced when T+1 Rank IC is negative, then increases gradually after it turns positive.</div>
        <div>Tier hit/return summary: {_html_escape(tier_summary)}</div>
        <div>D-day market sentiment: sentiment {_html_escape('-' if not np.isfinite(mkt_emotion_v) else _fmt_pct(mkt_emotion_v))}; up ratio {_html_escape('-' if not np.isfinite(mkt_up_v) else _fmt_pct(mkt_up_v))}; strong stocks {mkt_strong_v}</div>
        <div>Historical statistics sample: valid trading days {hist_days}; source: {_html_escape(hist_source)}; status: {_html_escape(hist_reason)}</div>
        <div>Model version: {_html_escape(model_version)}; generated at: {_html_escape(gen_ts)}</div>
        {('<ul>' + notes + '</ul>') if notes else ''}
      </div>
    </section>
    <p class="footnote">This report is generated automatically by Premium. Previous/next navigation is based on historical HTML report dates already present in the repository.</p>
  </main>
  <script>
    document.querySelectorAll('.tab-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        const target = btn.getAttribute('data-target');
        document.querySelectorAll('.tab-btn').forEach((x) => x.classList.remove('active'));
        document.querySelectorAll('main > section').forEach((panel) => panel.classList.add('hidden'));
        btn.classList.add('active');
        const panel = document.getElementById(target);
        if (panel) panel.classList.remove('hidden');
      }});
    }});
  </script>
</body>
</html>
"""


__all__ = [
    "LimitupValidationStats",
    "add_rank_groups",
    "attach_limitup_validation",
    "limitup_stats_from_verify",
    "render_premium_report_html",
]
