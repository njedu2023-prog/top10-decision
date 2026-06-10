#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Markdown 报告渲染（V3.2：人类操作表头版）

A) ✅ 新口径（Premium V3 主线，锁死）：
- 报告包含两张表：预测表 + 验证表（输出为 Markdown + 内嵌 HTML table）。
- 不依赖 PremiumA.html / PremiumB.html（彻底去模板化）。
- 若 T+2 行情未到：pending（不得报错卡死），验证表显示 PENDING 原因。
- 验证表顺序必须与预测表 Top30 完全一致（由上游保证）。
- 报告主表面向人类操作，不再展示工程字段长表头。
- 预测表与验证表统一展示：
  操作排名｜代码｜名称｜收盘价｜T+2预期收益｜预期价格区间｜T+2上涨概率｜模型置信度

入口函数：
- render_premium_report_md(...)

B) ♻️ 旧口径（PredEV/TopK/RankIC）：
- 保留 render_premium_md(...)，避免旧链路引用时炸裂。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import re

import numpy as np
import pandas as pd

from .config import PremiumConfig


# =========================
# 新口径（V3 报告：去模板化 + 内嵌 HTML）
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


def _fmt_price(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _fmt_bool(x: object) -> str:
    if x is True or str(x).lower() == "true" or str(x) == "1":
        return "✅"
    if x is False or str(x).lower() == "false" or str(x) == "0":
        return "❌"
    return "-"


def _fmt_direction(x: object) -> str:
    s = str(x or "").strip().lower()
    if s == "up":
        return "上修"
    if s == "down":
        return "下修"
    if s == "flat":
        return "持平"
    return str(x) if str(x).strip() else "-"


def _fmt_conf_label(x: object) -> str:
    s = str(x or "").strip().lower()
    if s == "high":
        return "高"
    if s == "mid":
        return "中"
    if s == "low":
        return "低"
    return str(x) if str(x).strip() else "-"


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
    if s.lower() in ("nan", "none", "<na>"):
        return default
    return s


def _select_cols_exist(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _cn_col(col: str) -> str:
    return str(col)


def _df_to_html_table(df: pd.DataFrame) -> str:
    """
    GitHub Markdown 对 <style> 会过滤/转义，导致样式内容被打印出来；
    这里只输出可渲染的 <table>，不输出 <style>。
    """
    if df is None or df.empty:
        return ""
    show = df.copy()
    show.columns = [_cn_col(c) for c in show.columns]
    html = show.to_html(index=False, escape=True, border=0)
    return f"<div>{html}</div>"


# =========================
# 人类操作表：字段生成
# =========================

_OPER_COLS = [
    "操作排名",
    "代码",
    "名称",
    "收盘价",
    "T+2预期收益",
    "预期价格区间",
    "T+2上涨概率",
    "模型置信度",
]


def _expected_range(row: pd.Series) -> str:
    """
    预期价格区间：优先展示 p25~p75，中位 p50。
    若分位列缺失，返回 '-'。
    """
    p25 = _num(row, "close_T2_p25")
    p50 = _num(row, "close_T2_p50")
    p75 = _num(row, "close_T2_p75")

    if np.isfinite(p25) and np.isfinite(p75) and np.isfinite(p50):
        return f"{p25:.2f} ~ {p75:.2f}，中位 {p50:.2f}"
    if np.isfinite(p25) and np.isfinite(p75):
        return f"{p25:.2f} ~ {p75:.2f}"
    if np.isfinite(p50):
        return f"中位 {p50:.2f}"
    return "-"


def _confidence_text(row: pd.Series) -> str:
    label = _fmt_conf_label(_str_val(row, "eret_plus_conf", ""))
    score = _num(row, "eret_plus_conf_score")
    if label != "-" and np.isfinite(score):
        return f"{label}（{score:.3f}）"
    if label != "-":
        return label
    return "-"


def _op_rank(row: pd.Series) -> object:
    for c in ("rank_eret_plus", "rank", "rank_r_p50"):
        if c in row.index and pd.notna(row.get(c)):
            try:
                return int(float(row.get(c)))
            except Exception:
                return row.get(c)
    return "-"


def _format_operation_table(df: pd.DataFrame, verify: bool = False) -> pd.DataFrame:
    """
    预测表与验证表统一使用人类操作表头：
    操作排名｜代码｜名称｜收盘价｜T+2预期收益｜预期价格区间｜T+2上涨概率｜模型置信度
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_OPER_COLS)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "操作排名": _op_rank(row),
            "代码": _str_val(row, "ts_code", "-"),
            "名称": _str_val(row, "name", "-"),
            "收盘价": _fmt_price(_num(row, "close_T"), 2),
            "T+2预期收益": _fmt_pct_ratio(_num(row, "eret_plus_value"), 2),
            "预期价格区间": _expected_range(row),
            "T+2上涨概率": _fmt_prob(_num(row, "p_premium"), 2),
            "模型置信度": _confidence_text(row),
        })

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
    ✅ 新口径：生成 Premium V3 报告（Markdown + 内嵌 HTML table）
    - 不输出 <style>（GitHub 会过滤/转义）。
    - 预测表与验证表统一为人类操作表头。
    - 工程字段不再进入主表，避免实盘阅读负担。
    """
    cfg = PremiumConfig.load()

    trade_date = str(trade_date or "").strip()
    target_date = str(target_date or trade_date).strip()

    parts: List[str] = []
    parts.append("# Premium（溢价预测）V3（E_ret_plus / Close[T+2] 分布预测）")
    parts.append("")
    parts.append("> 注：T 为本次预测的**基准交易日**（使用 Close[T]）；T+2 为**预测到期交易日**（预测 Close[T+2] 的分布）。")
    parts.append("")
    parts.append(f"- 预测日（T）：**{trade_date}**")
    parts.append(f"- 预测到期日（T+2）：**{target_date}**")
    parts.append(f"- 周期：**2 个交易日（T→T+2）**")
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
        parts.append("说明：这属于正常状态（T+2 真值未到）。系统仍会持续输出预测表并落盘缓存。")
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
    parts.append("## 字段说明（V3 人类操作口径）")
    parts.append("")
    parts.append("- 操作排名：优先使用 E_ret_plus 排名。")
    parts.append("- T+2预期收益：EHX 残差增强后的 E_ret_plus。")
    parts.append("- 预期价格区间：使用 T+2 价格分位 p25 ~ p75，并展示 p50 中位价。")
    parts.append("- T+2上涨概率：预测到期日 T+2 收盘上涨概率。")
    parts.append("- 模型置信度：EHX 置信度标签及置信分。")
    parts.append("")
    parts.append("> 注：E_ret原始值、EHX修正值、EHX来源、Raw/Plus误差等工程审计字段仍保留在 CSV 与验证摘要中，主表不再展开展示。")
    parts.append("")

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
    lines.append("- （旧口径报告渲染，保留用于历史链路；新主线已迁移至 V3）")
    lines.append(f"- trade_date：**{trade_date}**")
    lines.append(f"- next_trade_date：**{next_td}**")
    lines.append(f"- 模型版本：**{getattr(cfg, 'model_version', '-') }**")
    lines.append("")

    lines.append("## Top 排序（按 PredEV）")
    lines.append("")
    lines.append(df_show.to_markdown(index=False))
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
