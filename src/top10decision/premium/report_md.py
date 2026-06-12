#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Markdown 报告渲染（V3.5：涨停接力实盘表头版）

当前锁死口径：
- D：分析基准日，使用 Close[D]
- T：下一交易日集合竞价买入日
- T+1：买入后的预测到期 / 盘中择时卖出日

主表只展示人工下隔夜单最需要的 9 列：
操作排名｜代码｜名称｜D日收盘价｜T日涨停概率｜T日涨停强度｜T+1延续上涨率｜涨停接力评分｜T日建议买入方式

说明：
- 操作排名 = 上游 predict.py 已按“涨停接力评分”降序重排后的 rank。
- E_ret_plus、价格区间、卖出计划、置信度等工程/辅助字段仍保留在 CSV，不在主报告表展开。
- 保留 render_premium_md(...) 旧入口，避免历史链路引用时报错。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import PremiumConfig


# =========================
# 基础格式化
# =========================

def _fmt_prob(x: object, digits: int = 2) -> str:
    """x 是 0~1 概率 / 比率。"""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        # 兼容已经传入 0~100 的分值，但主概率字段标准应为 0~1。
        if v > 1.0 and v <= 100.0:
            return f"{v:.{digits}f}%"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


def _fmt_score(x: object, digits: int = 2) -> str:
    """x 是 0~100 分值。"""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _fmt_pct_ratio(x: object, digits: int = 2) -> str:
    """x 是 ratio（例如 0.0911 表示 +9.11%）。保留给旧摘要/旧入口。"""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v * 100:+.{digits}f}%"
    except Exception:
        return "-"


def _fmt_price(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _num(row: pd.Series, col: str, default: float = np.nan) -> float:
    if col not in row.index:
        return default
    try:
        v = float(row.get(col))
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _str_val(row: pd.Series, col: str, default: str = "") -> str:
    if col not in row.index:
        return default
    s = str(row.get(col, default)).strip()
    if s.lower() in ("nan", "none", "<na>", "nat", ""):
        return default
    return s


def _first_existing_str(row: pd.Series, cols: Sequence[str], default: str = "-") -> str:
    for c in cols:
        v = _str_val(row, c, "")
        if v:
            return v
    return default


def _cn_col(col: str) -> str:
    return str(col)


def _html_escape(x: object) -> str:
    import html

    if x is None:
        return ""
    s = str(x)
    if s.lower() in ("nan", "none", "<na>", "nat"):
        s = ""
    return html.escape(s, quote=True)


def _df_to_html_table(df: pd.DataFrame) -> str:
    """
    输出横向撑开的 HTML 表格。

    GitHub 会过滤 <style> 块，所以使用内联 style + nowrap 双保险：
    - table 使用 width:max-content，允许内容自然撑开；
    - th/td 使用 white-space:nowrap 和 nowrap，避免中文表头/名称被强制换行；
    - 外层 div 使用 overflow-x:auto，列多时横向滚动。
    """
    if df is None or df.empty:
        return ""

    show = df.copy()
    show.columns = [_cn_col(c) for c in show.columns]

    table_style = "width:max-content;min-width:100%;border-collapse:collapse;white-space:nowrap;"
    cell_style = "white-space:nowrap;padding:6px 10px;"
    div_style = "overflow-x:auto;width:100%;"

    rows: List[str] = []
    header = "".join(
        f'<th nowrap="nowrap" style="{cell_style}">{_html_escape(c)}</th>'
        for c in show.columns
    )
    rows.append(f"<tr>{header}</tr>")

    for _, r in show.iterrows():
        cells = "".join(
            f'<td nowrap="nowrap" style="{cell_style}">{_html_escape(r.get(c, ""))}</td>'
            for c in show.columns
        )
        rows.append(f"<tr>{cells}</tr>")

    return f'<div style="{div_style}"><table style="{table_style}">' + "".join(rows) + "</table></div>"


# =========================
# A股交易日口径显示
# =========================

def _clean_yyyymmdd(x: object) -> str:
    s = str(x or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return ""


def _parse_yyyymmdd(x: object) -> datetime | None:
    s = _clean_yyyymmdd(x)
    if len(s) != 8:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d")
    except Exception:
        return None


def _fmt_yyyymmdd(dt: datetime | None) -> str:
    return dt.strftime("%Y%m%d") if dt is not None else "-"


def _next_weekday_yyyymmdd(base_date: str) -> str:
    """
    兜底函数：只有在上游没有提供买入日字段时使用。
    严格 A 股交易日历应由 predict.py 输出 buy_date。
    """
    dt = _parse_yyyymmdd(base_date)
    if dt is None:
        return "-"
    dt = dt + timedelta(days=1)
    while dt.weekday() >= 5:
        dt = dt + timedelta(days=1)
    return _fmt_yyyymmdd(dt)


def _extract_date_from_frames(frames: Iterable[pd.DataFrame], cols: Sequence[str]) -> str:
    for df in frames:
        if df is None or df.empty:
            continue
        for c in cols:
            if c not in df.columns:
                continue
            vals = df[c].dropna().astype(str).map(_clean_yyyymmdd)
            vals = vals[vals.str.len() == 8]
            if not vals.empty:
                return str(vals.iloc[0])
    return ""


def _resolve_buy_date(trade_date: str, target_date: str, df_top30: pd.DataFrame, df_verify: pd.DataFrame) -> str:
    """T = 竞价买入日。优先读取上游按中国 A 股交易日历算好的字段。"""
    buy_date = _extract_date_from_frames(
        [df_top30, df_verify],
        ["buy_date", "t_buy_date", "T_date", "t_date", "next_trade_date", "next_td", "trade_date_T"],
    )
    if buy_date:
        return buy_date
    return _next_weekday_yyyymmdd(trade_date)


# =========================
# 涨停接力实盘主表
# =========================

_OPER_COLS = [
    "操作排名",
    "代码",
    "名称",
    "D日收盘价",
    "T日涨停概率",
    "T日涨停强度",
    "T+1延续上涨率",
    "涨停接力评分",
    "T日建议买入方式",
]


def _op_rank(row: pd.Series) -> object:
    # 注意：rank 已由 predict.py 按“涨停接力评分”重排，报告层必须优先使用 rank。
    for c in ("rank", "rank_limitup_continuation", "rank_eret_plus", "rank_r_p50"):
        if c in row.index and pd.notna(row.get(c)):
            try:
                return int(float(row.get(c)))
            except Exception:
                return row.get(c)
    return "-"


def _t_buy_method(row: pd.Series) -> str:
    # 报告展示为 T日建议买入方式；兼容上游 CSV 旧字段 T+1建议买入方式。
    return _first_existing_str(
        row,
        ["T日建议买入方式", "T+1建议买入方式", "t_buy_method", "t1_buy_method", "buy_method"],
    )


def _limitup_prob_text(row: pd.Series) -> str:
    for c in ("T日涨停概率", "t_limitup_prob", "limitup_prob", "t_limit_up_prob"):
        if c in row.index:
            return _fmt_prob(_num(row, c), 2)
    return "-"


def _limitup_strength_text(row: pd.Series) -> str:
    for c in ("T日涨停强度", "t_limitup_strength", "limitup_strength", "t_limit_up_strength"):
        if c in row.index:
            return _fmt_score(_num(row, c), 2)
    return "-"


def _continue_up_text(row: pd.Series) -> str:
    for c in ("T+1延续上涨率", "t1_continue_up_rate", "continue_up_rate", "t1_up_rate"):
        if c in row.index:
            return _fmt_prob(_num(row, c), 2)
    return "-"


def _continuation_score_text(row: pd.Series) -> str:
    for c in ("涨停接力评分", "limitup_continuation_score", "continuation_score"):
        if c in row.index:
            return _fmt_score(_num(row, c), 2)
    return "-"


def _format_operation_table(df: pd.DataFrame, verify: bool = False) -> pd.DataFrame:
    """
    预测表与验证表统一使用涨停接力实盘表头：
    操作排名｜代码｜名称｜D日收盘价｜T日涨停概率｜T日涨停强度｜T+1延续上涨率｜涨停接力评分｜T日建议买入方式
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_OPER_COLS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "操作排名": _op_rank(row),
                "代码": _str_val(row, "ts_code", "-"),
                "名称": _str_val(row, "name", "-"),
                "D日收盘价": _fmt_price(_num(row, "close_T"), 2),
                "T日涨停概率": _limitup_prob_text(row),
                "T日涨停强度": _limitup_strength_text(row),
                "T+1延续上涨率": _continue_up_text(row),
                "涨停接力评分": _continuation_score_text(row),
                "T日建议买入方式": _t_buy_method(row),
            }
        )

    return pd.DataFrame(rows, columns=_OPER_COLS)


def _format_pred_table(df_top30: pd.DataFrame) -> pd.DataFrame:
    return _format_operation_table(df_top30, verify=False)


def _format_verify_table(df_verify: pd.DataFrame) -> pd.DataFrame:
    return _format_operation_table(df_verify, verify=True)


def _append_ehx_summary(parts: List[str], df_verify: pd.DataFrame) -> None:
    """验证完成后，追加 raw vs plus 误差改善摘要。"""
    if df_verify is None or df_verify.empty:
        return
    if "raw_abs_err" not in df_verify.columns or "plus_abs_err" not in df_verify.columns:
        return

    raw_err = pd.to_numeric(df_verify["raw_abs_err"], errors="coerce")
    plus_err = pd.to_numeric(df_verify["plus_abs_err"], errors="coerce")
    valid = raw_err.notna() & plus_err.notna()
    if not bool(valid.any()):
        return

    raw_mae = float(raw_err[valid].mean())
    plus_mae = float(plus_err[valid].mean())
    improve_rate = float((plus_err[valid] < raw_err[valid]).mean())
    delta = raw_mae - plus_mae

    parts.append("")
    parts.append("## E_ret_plus / EHX 验证摘要")
    parts.append("")
    parts.append(f"- Raw MAE：{raw_mae * 100:.4f}%")
    parts.append(f"- Plus MAE：{plus_mae * 100:.4f}%")
    parts.append(f"- MAE 改善：{delta * 100:+.4f}%")
    parts.append(f"- Plus 优于 Raw 比例：{improve_rate * 100:.2f}%")


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
    生成 Premium V3.5 报告（Markdown + 内嵌 HTML table）
    - 不输出 <style>，避免 GitHub 过滤/转义。
    - 主表只展示涨停接力实盘 9 列。
    - 工程字段不进入主表，继续保留在 CSV 与验证摘要中。
    """
    cfg = PremiumConfig.load()

    trade_date = _clean_yyyymmdd(trade_date) or str(trade_date or "").strip()
    target_date = _clean_yyyymmdd(target_date) or str(target_date or trade_date).strip()
    buy_date = _resolve_buy_date(trade_date, target_date, df_top30, df_verify)

    parts: List[str] = []
    parts.append("# Premium（涨停接力实盘预测）V3.5（D→T→T+1）")
    parts.append("")
    parts.append(
        "> 注：D 为本次预测的**分析基准日**（使用 Close[D]）；"
        "T 为下一交易日**集合竞价买入日**；"
        "T+1 为买入后的**延续上涨验证/盘中择时卖出日**。"
    )
    parts.append("")
    parts.append(f"- 预测日（D）：**{trade_date}**")
    parts.append(f"- 竞价买入日（T）：**{buy_date}**")
    parts.append(f"- 预测到期日（T+1）：**{target_date}**")
    parts.append("- 周期：**2 个交易日（D→T+1）**")
    parts.append(f"- 生成时间：{gen_ts}")
    parts.append(f"- 模型版本：**{getattr(cfg, 'model_version', '-') }**")
    parts.append("")

    parts.append("## 预测表（Top30）")
    parts.append("")
    pred_show = _format_pred_table(df_top30)
    if pred_show is None or pred_show.empty:
        parts.append("（预测表为空，属于异常；请检查 pred_source_latest 输入）")
    else:
        parts.append(_df_to_html_table(pred_show))
    parts.append("")

    parts.append("## 验证表（Top30）")
    parts.append("")
    if verify_pending:
        parts.append(f"**状态：PENDING**（原因：{verify_reason}）")
        parts.append("")
        parts.append("说明：这属于正常状态（T+1 真值未到）。系统仍会持续输出预测表并落盘缓存。")
        parts.append("")
        verify_show = _format_verify_table(df_verify) if df_verify is not None else pd.DataFrame()
        if verify_show is not None and not verify_show.empty:
            parts.append(_df_to_html_table(verify_show))
            parts.append("")
    else:
        verify_show = _format_verify_table(df_verify)
        if verify_show is None or verify_show.empty:
            parts.append("（验证表为空；可能真值列未能 merge 成功）")
        else:
            parts.append(_df_to_html_table(verify_show))
            parts.append("")

        if df_verify is not None and "hit_up" in df_verify.columns:
            hit = (df_verify["hit_up"].astype(str) == "是").sum()
            total = int(len(df_verify)) if len(df_verify) else 0
            hit_rate = (hit / total * 100.0) if total > 0 else 0.0
            parts.append(f"- 命中（旧口径：actual_ret>0）：{hit}/{total}（{hit_rate:.2f}%）")

        if df_verify is not None and "in_p10" in df_verify.columns:
            cov10 = float(pd.to_numeric(df_verify["in_p10"], errors="coerce").fillna(False).mean())
            parts.append(f"- 覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：{cov10*100:.2f}%")
        if df_verify is not None and "in_p50" in df_verify.columns:
            cov50 = float(pd.to_numeric(df_verify["in_p50"], errors="coerce").fillna(False).mean())
            parts.append(f"- 覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：{cov50*100:.2f}%")
        if df_verify is not None and "err_r_p50" in df_verify.columns:
            mae_r = float(pd.to_numeric(df_verify["err_r_p50"], errors="coerce").abs().mean())
            parts.append(f"- MAE（V2：|err_r_p50|）：{mae_r:.6f}")
        if df_verify is not None and "err_close_p50" in df_verify.columns:
            mae_c = float(pd.to_numeric(df_verify["err_close_p50"], errors="coerce").abs().mean())
            parts.append(f"- MAE（V2：|err_close_p50|）：{mae_c:.6f}")

        _append_ehx_summary(parts, df_verify)

    parts.append("")
    parts.append("## 字段说明（V3.5 涨停接力实盘口径）")
    parts.append("")
    parts.append("- 操作排名：按上游 `涨停接力评分` 降序排列后的名次；同分再参考 T日涨停概率、T+1延续上涨率、T日涨停强度。")
    parts.append("- D日收盘价：分析基准日 D 的收盘价。")
    parts.append("- T日涨停概率：模型/规则层对 T 日冲击涨停可能性的评分化概率。")
    parts.append("- T日涨停强度：衡量 T 日涨停攻击质量与封板强弱的 0~100 分。")
    parts.append("- T+1延续上涨率：T 日走强/涨停后，T+1 继续上涨并给出溢价的倾向率。")
    parts.append("- 涨停接力评分：综合 T日涨停概率、T日涨停强度、T+1延续上涨率与执行安全分后的核心排序分。")
    parts.append("- T日建议买入方式：面向 T 日集合竞价的买入方式建议。")
    parts.append("")
    parts.append(
        "> 注：E_ret_plus、价格区间、置信度、买入价、卖出计划、Raw/Plus误差等辅助/审计字段仍保留在 CSV 与验证摘要中，主表不再展开展示。"
    )
    if buy_date == _next_weekday_yyyymmdd(trade_date):
        parts.append(
            "> 注：竞价买入日优先读取上游交易日历字段；若 CSV 暂无 buy_date/next_trade_date，则本报告用工作日兜底。严格 A 股节假日口径建议由 predict.py 输出 buy_date。"
        )
    parts.append("")

    return "\n".join(parts)


# =========================
# 旧口径（保留，避免历史引用炸裂）
# =========================

def _fmt_pct(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
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
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def render_premium_md(df_rank: pd.DataFrame, cfg: PremiumConfig, trade_date: str) -> str:
    """
    旧口径：渲染 Premium 报告（Markdown）。
    保留用于历史链路；新主线请用 render_premium_report_md。
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
    lines.append("- （旧口径报告渲染，保留用于历史链路；新主线已迁移至 V3.5）")
    lines.append(f"- trade_date：**{trade_date}**")
    lines.append(f"- next_trade_date：**{next_td}**")
    lines.append(f"- 模型版本：**{getattr(cfg, 'model_version', '-') }**")
    lines.append("")
    lines.append("## Top 排序（按 PredEV）")
    lines.append("")
    lines.append(df_show.to_markdown(index=False) if not df_show.empty else "（暂无）")
    lines.append("")
    lines.append("## 风险摘要")
    lines.append("")
    lines.extend(risk_summary if risk_summary else ["- （暂无）"])
    lines.append("")
    lines.append("## 评估（若有真实对照）")
    lines.append("")
    lines.extend(eval_lines)
    lines.append("")

    return "\n".join(lines)


__all__ = ["render_premium_report_md", "render_premium_md"]
