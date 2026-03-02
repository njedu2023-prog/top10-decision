#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Markdown 报告渲染（去模板化版本）

A) ✅ 新口径（Premium V2 主线，锁死）：
- 报告包含两张表：预测表 + 验证表（输出为 Markdown 内嵌 HTML 表格）
- 不依赖 PremiumA.html / PremiumB.html（彻底去模板化）
- 若 T+2 行情未到：pending（不得报错卡死），验证表显示 PENDING 原因
- 验证表顺序必须与预测表 Top30 完全一致（由上游保证）

入口函数：
- render_premium_report_md(...)

B) ♻️ 旧口径（PredEV/TopK/RankIC）：
- 保留 render_premium_md(...)，避免旧链路引用时炸裂
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import re

import numpy as np
import pandas as pd

from .config import PremiumConfig


# =========================
# 新口径（V2 报告：去模板化 + 内嵌 HTML）
# =========================

def _fmt_prob(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


def _fmt_pct_ratio(x: object, digits: int = 2) -> str:
    """
    x 是 ratio（例如 0.0911 表示 +9.11%）
    """
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v * 100:+.{digits}f}%"
    except Exception:
        return "-"


def _fmt_float(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _fmt_price(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _fmt_bool(x: object) -> str:
    if x is True or str(x).lower() == "true":
        return "✅"
    if x is False or str(x).lower() == "false":
        return "❌"
    return "-"


def _detect_quantile_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    r_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("r_p")]
    p_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("close_T2_p")]

    def _key(c: str) -> int:
        digits = "".join([ch for ch in c if ch.isdigit()])
        try:
            return int(digits[-2:]) if len(digits) >= 2 else int(digits)
        except Exception:
            return 999

    return sorted(r_cols, key=_key), sorted(p_cols, key=_key)


def _select_cols_exist(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# =========================
# 展示层：中文表头 + 不换行 + 字号控制（仅影响报告渲染，不影响计算/CSV）
# =========================

_CN_FIXED = {
    "rank": "排名",
    "trade_date": "交易日(T)",
    "target_date": "目标交易日(T+2)",
    "ts_code": "代码",
    "name": "名称",
    "close_T": "收盘价(T)",
    "close_T2_actual": "收盘价(T+2)",
    "r_actual": "实际ln收益",
    "rank_r_p50": "中位ln收益排名",
    "p_premium": "上涨概率",
    "e_premium": "预期溢价",
    "score_ev": "综合评分",
    "dec_rank": "决策排名",
    "dec_weight": "决策权重",
    "dec_can_buy": "可买",
    "dec_p_fill": "可成交概率",
    "dec_reason": "决策原因",
    "in_p10": "命中P10区间",
    "in_p50": "命中P50区间",
    "err_r_p50": "ln收益误差(中位)",
    "err_close_p50": "价格误差(中位)",
    "actual_ret": "实际收益",
    "hit_up": "是否上涨",
}


def _cn_col(col: str) -> str:
    c = str(col)
    if c in _CN_FIXED:
        return _CN_FIXED[c]
    m = re.match(r"^r_p(\d{2})$", c)
    if m:
        return f"ln收益分位{m.group(1)}"
    m = re.match(r"^close_T2_p(\d{2})$", c)
    if m:
        return f"T+2价格分位{m.group(1)}"
    return c


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    show = df.copy()
    show.columns = [_cn_col(c) for c in show.columns]
    html = show.to_html(index=False, escape=False, border=0)
    return f'<div class="tbl-wrap">{html}</div>'


def _report_style_block() -> str:
    # GitHub Markdown 支持内嵌 HTML/CSS（用于字号与不换行控制）
    return """<style>
.premium-report { font-size: 12px; line-height: 1.45; }
.premium-report h1, .premium-report h2, .premium-report h3 { font-size: 14px; margin: 10px 0 6px; }
.premium-report p, .premium-report li { font-size: 12px; margin: 4px 0; }
.premium-report .tbl-wrap { overflow-x: auto; padding: 4px 0; }
.premium-report table { border-collapse: collapse; font-size: 12px; }
.premium-report th, .premium-report td { border: 1px solid #e5e7eb; padding: 4px 6px; white-space: nowrap; vertical-align: middle; }
.premium-report th { font-weight: 600; }
</style>"""


def _format_pred_table(df_top30: pd.DataFrame) -> pd.DataFrame:
    df = df_top30.copy()

    r_cols, p_cols = _detect_quantile_cols(df)

    base_cols = [
        "rank", "trade_date", "target_date", "ts_code", "name",
        "close_T",
    ]
    cols = base_cols + r_cols + p_cols
    cols += _select_cols_exist(df, ["rank_r_p50"])
    cols += _select_cols_exist(df, ["p_premium", "e_premium", "score_ev"])
    cols += _select_cols_exist(df, ["dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason"])

    cols = _select_cols_exist(df, cols)
    out = df[cols].copy()

    if "p_premium" in out.columns:
        out["p_premium"] = out["p_premium"].map(lambda x: _fmt_prob(x, 2))
    if "e_premium" in out.columns:
        out["e_premium"] = out["e_premium"].map(lambda x: _fmt_pct_ratio(x, 2))

    # ✅ close_T 只展示一个收盘价（且不换行由 CSS 控制）
    if "close_T" in out.columns:
        out["close_T"] = out["close_T"].map(lambda x: _fmt_price(x, 2))

    for c in r_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: _fmt_float(x, 4))

    for c in p_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: _fmt_price(x, 3))

    if "score_ev" in out.columns:
        out["score_ev"] = out["score_ev"].map(lambda x: _fmt_float(x, 4))

    return out


def _format_verify_table(df_verify: pd.DataFrame) -> pd.DataFrame:
    df = df_verify.copy()

    cols = [
        "rank", "trade_date", "target_date", "ts_code", "name",
        "close_T", "close_T2_actual", "r_actual",
        "in_p10", "in_p50",
        "err_r_p50", "err_close_p50",
        "actual_ret", "hit_up",
    ]
    cols = _select_cols_exist(df, cols)
    out = df[cols].copy()

    if "close_T" in out.columns:
        out["close_T"] = out["close_T"].map(lambda x: _fmt_price(x, 2))
    if "close_T2_actual" in out.columns:
        out["close_T2_actual"] = out["close_T2_actual"].map(lambda x: _fmt_price(x, 2))
    if "r_actual" in out.columns:
        out["r_actual"] = out["r_actual"].map(lambda x: _fmt_float(x, 4))
    if "in_p10" in out.columns:
        out["in_p10"] = out["in_p10"].map(_fmt_bool)
    if "in_p50" in out.columns:
        out["in_p50"] = out["in_p50"].map(_fmt_bool)
    if "err_r_p50" in out.columns:
        out["err_r_p50"] = out["err_r_p50"].map(lambda x: _fmt_float(x, 4))
    if "err_close_p50" in out.columns:
        out["err_close_p50"] = out["err_close_p50"].map(lambda x: _fmt_price(x, 3))

    if "actual_ret" in out.columns:
        out["actual_ret"] = out["actual_ret"].map(lambda x: _fmt_pct_ratio(x, 2))

    return out


def render_premium_report_md(
    trade_date: str,
    target_date: str,
    df_top30: pd.DataFrame,
    df_verify: pd.DataFrame,
    verify_pending: bool,
    verify_reason: str,
    gen_ts: str,
) -> str:
    """
    ✅ 新口径：生成 Premium V2 报告（Markdown 内嵌 HTML）
    - 表头中文（仅展示，不改 CSV 字段与计算）
    - 表格不换行（横向滚动）
    - 标题 14px，正文/表格 12px
    """
    cfg = PremiumConfig.load()

    # 统一兜底：target_date 永远非空
    trade_date = str(trade_date or "").strip()
    target_date = str(target_date or trade_date).strip()

    parts: List[str] = []
    parts.append(_report_style_block())
    parts.append('<div class="premium-report">')

    parts.append(f"<h1>Premium（溢价预测）V2（Close[T+2] 分布预测）</h1>")
    parts.append("<ul>")
    parts.append(f"<li>交易日（T）：<b>{trade_date}</b></li>")
    parts.append(f"<li>目标交易日（T+2）：<b>{target_date}</b></li>")
    parts.append("<li>周期：<b>2 个交易日（T→T+2）</b></li>")
    parts.append(f"<li>生成时间：{gen_ts}</li>")
    parts.append(f"<li>模型版本：<b>{getattr(cfg, 'model_version', '-') }</b></li>")
    parts.append("</ul>")

    parts.append("<h2>预测表（Top30）</h2>")
    pred_show = _format_pred_table(df_top30)
    if pred_show is None or pred_show.empty:
        parts.append("<p>（预测表为空，属于异常；请检查 pred_source_latest 输入）</p>")
    else:
        parts.append(_df_to_html_table(pred_show))

    parts.append("<h2>验证表（Top30）</h2>")
    if verify_pending:
        parts.append(f"<p><b>状态：PENDING</b>（原因：{verify_reason}）</p>")
        parts.append("<p>说明：这属于正常状态（T+2 真值未到）。系统仍会持续输出预测表并落盘缓存。</p>")
        verify_show = _format_verify_table(df_verify) if df_verify is not None else pd.DataFrame()
        if verify_show is not None and not verify_show.empty:
            parts.append(_df_to_html_table(verify_show))
    else:
        verify_show = _format_verify_table(df_verify)
        if verify_show is None or verify_show.empty:
            parts.append("<p>（验证表为空；可能真值列未能 merge 成功）</p>")
        else:
            parts.append(_df_to_html_table(verify_show))

        if df_verify is not None and "hit_up" in df_verify.columns:
            hit = (df_verify["hit_up"].astype(str) == "是").sum()
            total = int(len(df_verify)) if len(df_verify) else 0
            hit_rate = (hit / total * 100.0) if total > 0 else 0.0
            parts.append(f"<p>命中（旧口径：actual_ret>0）：{hit}/{total}（{hit_rate:.2f}%）</p>")

        if df_verify is not None and "in_p10" in df_verify.columns:
            cov10 = float(pd.to_numeric(df_verify["in_p10"], errors="coerce").fillna(False).mean())
            parts.append(f"<p>覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：{cov10*100:.2f}%</p>")
        if df_verify is not None and "in_p50" in df_verify.columns:
            cov50 = float(pd.to_numeric(df_verify["in_p50"], errors="coerce").fillna(False).mean())
            parts.append(f"<p>覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：{cov50*100:.2f}%</p>")
        if df_verify is not None and "err_r_p50" in df_verify.columns:
            mae_r = float(pd.to_numeric(df_verify["err_r_p50"], errors="coerce").abs().mean())
            parts.append(f"<p>MAE（V2：|err_r_p50|）：{mae_r:.6f}</p>")
        if df_verify is not None and "err_close_p50" in df_verify.columns:
            mae_c = float(pd.to_numeric(df_verify["err_close_p50"], errors="coerce").abs().mean())
            parts.append(f"<p>MAE（V2：|err_close_p50|）：{mae_c:.6f}</p>")

    parts.append("<h2>文件输出</h2>")
    parts.append("<p>（说明：候选=pred_source_latest 全量，不做过滤；排序同 Top30。）</p>")
    parts.append("<ul>")
    parts.append(f"<li>full 文件：<code>outputs/premium/premium_full_{trade_date}.csv</code></li>")
    parts.append(f"<li>top30 文件：<code>outputs/premium/premium_top30_{trade_date}.csv</code></li>")
    parts.append(f"<li>verify 文件：<code>outputs/premium/premium_verify_{trade_date}.csv</code></li>")
    parts.append("</ul>")

    parts.append("<h2>字段说明（V2 核心）</h2>")
    parts.append("<ul>")
    parts.append("<li>r_pXX：log-return 分位点，r = ln(Close[T+2]/Close[T])</li>")
    parts.append("<li>close_T2_pXX：价格分位点 = close_T * exp(r_pXX)</li>")
    parts.append("<li>in_p10：r_actual 是否落在 [p05, p95]</li>")
    parts.append("<li>in_p50：r_actual 是否落在 [p25, p75]</li>")
    parts.append("</ul>")

    parts.append("</div>")
    return "\n".join(parts)


# =========================
# 旧口径（保留，避免历史引用炸裂）
# =========================

def _fmt_pct(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v*100:.{digits}f}%"
    except Exception:
        return "-"


def _spearman_rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def rank(x: np.ndarray) -> np.ndarray:
        x2 = np.where(np.isnan(x), -1e18, x)
        order = np.argsort(x2)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x2), dtype=float)
        return r

    ra = rank(a)
    rb = rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _fmt_float_old(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def render_premium_md(df_rank: pd.DataFrame, cfg: PremiumConfig, trade_date: str) -> str:
    """
    ♻️ 旧口径：渲染 Premium 报告（Markdown）
    （保留用于历史链路；新主线请用 render_premium_report_md）
    """
    topk = int(getattr(cfg, "topk", 10))
    df = df_rank.copy()

    next_td = "-"
    if "next_trade_date" in df.columns and df["next_trade_date"].notna().any():
        next_td = str(df["next_trade_date"].dropna().iloc[0])

    df_top = df.head(topk).copy()

    show_cols = [
        "rank_pred_ev",
        "ts_code",
        "name",
        "pred_up_prob",
        "pred_ret_mean",
        "pred_ev",
        "risk_liquidity",
        "fill_risk_hint",
        "real_premium_ret",
    ]
    cols_exist = [c for c in show_cols if c in df_top.columns]
    df_show = df_top[cols_exist].copy()

    if "pred_up_prob" in df_show.columns:
        df_show["pred_up_prob"] = df_show["pred_up_prob"].map(lambda x: _fmt_pct(x, 1))
    if "pred_ret_mean" in df_show.columns:
        df_show["pred_ret_mean"] = df_show["pred_ret_mean"].map(lambda x: _fmt_pct(x, 2))
    if "pred_ev" in df_show.columns:
        df_show["pred_ev"] = df_show["pred_ev"].map(lambda x: _fmt_pct(x, 2))
    if "real_premium_ret" in df_show.columns:
        df_show["real_premium_ret"] = df_show["real_premium_ret"].map(lambda x: _fmt_pct(x, 2))

    risk_summary = []
    if "risk_liquidity" in df.columns:
        n_high = int((df["risk_liquidity"].astype(str) == "HIGH").sum())
        risk_summary.append(f"- 流动性风险 HIGH：{n_high} / {len(df)}")
    if "fill_risk_hint" in df.columns and df["fill_risk_hint"].notna().any():
        risk_summary.append("- 买不到风险：已从上游字段透传（本模块不参与排序）")

    eval_lines = []
    has_real = "real_premium_ret" in df.columns and pd.to_numeric(df["real_premium_ret"], errors="coerce").notna().any()
    if has_real:
        rr = pd.to_numeric(df_top["real_premium_ret"], errors="coerce").values
        hit = float(np.nanmean(rr > 0.0)) if len(rr) > 0 else float("nan")
        mean_ret = float(np.nanmean(rr)) if len(rr) > 0 else float("nan")

        pred_ev = pd.to_numeric(df["pred_ev"], errors="coerce").values
        real_all = pd.to_numeric(df["real_premium_ret"], errors="coerce").values
        ric = _spearman_rank_ic(pred_ev, real_all)

        eval_lines += [
            f"- HitRate@{topk}（Top{topk} 真实收益>0 比例）：{hit*100:.1f}%",
            f"- Top{topk} 真实平均收益：{mean_ret*100:.2f}%",
            f"- RankIC（pred_ev vs real）：{_fmt_float_old(ric, 4)}",
        ]
    else:
        eval_lines.append("- 真实对照（real_premium_ret）尚未产生：当前为预测版报告（正常）")

    lines = []
    lines.append(f"# Premium 溢价预测排序（{trade_date}）")
    lines.append("")
    lines.append("- （旧口径报告渲染，保留用于历史链路；新主线已迁移至 V2）")
    lines.append(f"- trade_date：**{trade_date}**")
    lines.append(f"- next_trade_date：**{next_td}**")
    lines.append(f"- 模型版本：**{getattr(cfg, 'model_version', '-') }**")
    lines.append("")

    lines.append("## Top 排序（按 PredEV）")
    lines.append("")
    lines.append(df_show.to_markdown(index=False))
    lines.append("")

    lines.append("## 风险提示摘要")
    lines.append("")
    lines.extend(risk_summary if risk_summary else ["- （暂无）"])
    lines.append("")

    lines.append("## 评估（若有真实对照）")
    lines.append("")
    lines.extend(eval_lines)
    lines.append("")

    return "\n".join(lines)


__all__ = ["render_premium_report_md", "render_premium_md"]
