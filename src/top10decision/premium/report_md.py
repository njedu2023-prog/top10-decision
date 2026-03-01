#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Markdown 报告渲染

本文件同时支持两套输出（向后兼容 + 新主线）：

A) ✅ 新口径（Premium 手工交易版 V1，锁死）：
- 报告页面必须包含两张表：预测表 + 验证表（样式来自 PremiumA.html / PremiumB.html）
- 若 T+2 行情未到：pending（不得报错卡死），验证表显示 PENDING 原因
- 验证表顺序必须与预测表 Top30 完全一致（由上游保证）

入口函数：
- render_premium_report_md(...)

B) ♻️ 旧口径（PredEV/TopK/RankIC）：
- 保留 render_premium_md(...)，避免旧链路引用时炸裂
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PremiumConfig


# =========================
# 新口径（V1 两张表模板渲染）
# =========================

def _fmt_prob(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


def _fmt_pct_ratio(x: object, digits: int = 2, with_arrow: bool = True) -> str:
    """
    x 是 ratio（例如 0.0911 表示 +9.11%）
    """
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        if not with_arrow:
            return f"{v * 100:.{digits}f}%"
        sign = "↑" if v >= 0 else "↓"
        return f"{sign} {v:+.{digits}%}"
    except Exception:
        return "-"


def _row_td(v: str, strong: bool = False, color_red_if_up: bool = False) -> str:
    style = "border:1px solid #111; padding:8px;"
    if strong:
        style += " font-weight:700;"
    if color_red_if_up and isinstance(v, str) and v.startswith("↑"):
        style += " color:#d00;"
    return f'<td style="{style}">{v}</td>'


def _build_rows_pred(df_top30: pd.DataFrame) -> str:
    rows = []
    for _, r in df_top30.iterrows():
        rows.append(
            "<tr>"
            + _row_td(str(r.get("rank", "")))
            + _row_td(str(r.get("trade_date", "")))
            + _row_td(str(r.get("target_date", "")))
            + _row_td(str(r.get("ts_code", "")))
            + _row_td(str(r.get("name", "")))
            + _row_td(_fmt_prob(r.get("p_premium", np.nan), 2))
            + _row_td(_fmt_pct_ratio(r.get("e_premium", np.nan), 2, True), strong=True, color_red_if_up=True)
            + _row_td(str(r.get("score_ev", "")))
            + _row_td(str(r.get("risk_flags", "")))
            + _row_td(str(r.get("confidence", "")))
            + _row_td(str(r.get("data_quality", "")))
            + _row_td(str(r.get("dec_rank", "")))
            + _row_td(str(r.get("dec_weight", "")))
            + _row_td(str(r.get("dec_can_buy", "")))
            + _row_td(str(r.get("dec_p_fill", "")))
            + _row_td(str(r.get("dec_reason", "")))
            + "</tr>"
        )
    return "\n".join(rows)


def _build_rows_verify(df_verify: pd.DataFrame) -> str:
    rows = []
    for _, r in df_verify.iterrows():
        pred_v = _fmt_pct_ratio(r.get("e_premium", np.nan), 2, True)
        act_v = _fmt_pct_ratio(r.get("actual_ret", np.nan), 2, True)
        rows.append(
            "<tr>"
            + _row_td(str(r.get("rank", "")))
            + _row_td(str(r.get("trade_date", "")))
            + _row_td(str(r.get("target_date", "")))
            + _row_td(str(r.get("ts_code", "")))
            + _row_td(str(r.get("name", "")))
            + _row_td(pred_v, strong=True, color_red_if_up=True)
            + _row_td(act_v, strong=True, color_red_if_up=True)
            + _row_td(str(r.get("hit_up", "")))
            + "</tr>"
        )
    return "\n".join(rows)


def _read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    ✅ 新口径：生成 Premium V1 报告（Markdown），包含两张表（预测表+验证表）
    模板来源：仓库根目录 PremiumA.html / PremiumB.html
    """
    cfg = PremiumConfig.load()
    repo_root = cfg.repo_root()

    tpl_a = (repo_root / "PremiumA.html").resolve()
    tpl_b = (repo_root / "PremiumB.html").resolve()

    html_a = _read_template(tpl_a)
    html_b = _read_template(tpl_b)

    # 预测表
    title_a = f"{trade_date} → {target_date}　TOP 30 溢价概率研究报告"
    html_a = (
        html_a.replace("{{TITLE}}", title_a)
        .replace("{{ROWS}}", _build_rows_pred(df_top30))
        .replace("{{GEN_TS}}", gen_ts)
        .replace("{{TRADE_DATE}}", trade_date)
        .replace("{{TARGET_DATE}}", target_date)
    )

    # 验证表
    if verify_pending:
        title_b = f"{trade_date} → {target_date}　TOP 30 溢价概率为正预测命中率：PENDING"
        html_b = (
            html_b.replace("{{TITLE}}", title_b)
            .replace("{{ROWS}}", "")
            .replace("{{GEN_TS}}", gen_ts)
            .replace("{{TRADE_DATE}}", trade_date)
            .replace("{{TARGET_DATE}}", target_date)
        )
        pending_line = f"> 验证表状态：**PENDING**（原因：{verify_reason}）\n\n"
        hit_line = ""
    else:
        hit = (df_verify["hit_up"].astype(str) == "是").sum() if "hit_up" in df_verify.columns else 0
        total = int(len(df_verify)) if len(df_verify) else 0
        hit_rate = (hit / total * 100.0) if total > 0 else 0.0
        title_b = f"{trade_date} → {target_date}　TOP 30 溢价概率为正预测命中率：{hit_rate:.2f}%"
        html_b = (
            html_b.replace("{{TITLE}}", title_b)
            .replace("{{ROWS}}", _build_rows_verify(df_verify))
            .replace("{{GEN_TS}}", gen_ts)
            .replace("{{TRADE_DATE}}", trade_date)
            .replace("{{TARGET_DATE}}", target_date)
        )
        pending_line = ""
        hit_line = f"- 命中：{hit}/{total}（{hit_rate:.2f}%）\n\n"

    # 组装 MD
    md = []
    md.append("# Premium（溢价预测）手工交易版 V1\n\n")
    md.append(f"- trade_date（T）：**{trade_date}**\n")
    md.append(f"- target_date（T+2）：**{target_date}**\n")
    md.append("- horizon：**2 个交易日（T→T+2）**\n")
    md.append(f"- 生成时间：{gen_ts}\n\n")

    if pending_line:
        md.append(pending_line)
    if hit_line:
        md.append(hit_line)

    md.append("## 预测表（Top30）\n\n")
    md.append(html_a + "\n\n")
    md.append("## 验证表（Top30）\n\n")
    md.append(html_b + "\n\n")

    md.append("## 全量展开（full）\n\n")
    md.append("（说明：候选=pred_source_latest 全量，不做过滤；排序同 Top30。）\n\n")
    md.append(f"- full 文件：`outputs/premium/premium_full_{trade_date}.csv`\n")
    md.append(f"- top30 文件：`outputs/premium/premium_top30_{trade_date}.csv`\n")
    md.append(f"- verify 文件：`outputs/premium/premium_verify_{trade_date}.csv`\n\n")

    return "".join(md)


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


def _fmt_float(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
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
            f"- RankIC（pred_ev vs real）：{_fmt_float(ric, 4)}",
        ]
    else:
        eval_lines.append("- 真实对照（real_premium_ret）尚未产生：当前为预测版报告（正常）")

    lines = []
    lines.append(f"# Premium 溢价预测排序（{trade_date}）")
    lines.append("")
    lines.append("- （旧口径报告渲染，保留用于历史链路；新主线已迁移至 PremiumA/B 模板）")
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
