#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Markdown 报告渲染

职责：
- 将 Premium 排序表（DataFrame）渲染为 Markdown 文本
- 报告用于人类阅读与 GitHub Pages 展示
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import PremiumConfig


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
    渲染 Premium 报告（Markdown）
    """
    topk = int(cfg.topk)
    df = df_rank.copy()

    # 取 next_trade_date（若存在）
    next_td = "-"
    if "next_trade_date" in df.columns and df["next_trade_date"].notna().any():
        next_td = str(df["next_trade_date"].dropna().iloc[0])

    # TopK 表格
    df_top = df.head(topk).copy()

    # 选择展示列
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

    # 格式化
    if "pred_up_prob" in df_show.columns:
        df_show["pred_up_prob"] = df_show["pred_up_prob"].map(lambda x: _fmt_pct(x, 1))
    if "pred_ret_mean" in df_show.columns:
        df_show["pred_ret_mean"] = df_show["pred_ret_mean"].map(lambda x: _fmt_pct(x, 2))
    if "pred_ev" in df_show.columns:
        df_show["pred_ev"] = df_show["pred_ev"].map(lambda x: _fmt_pct(x, 2))
    if "real_premium_ret" in df_show.columns:
        df_show["real_premium_ret"] = df_show["real_premium_ret"].map(lambda x: _fmt_pct(x, 2))

    # 风险汇总
    risk_summary = []
    if "risk_liquidity" in df.columns:
        n_high = int((df["risk_liquidity"].astype(str) == "HIGH").sum())
        risk_summary.append(f"- 流动性风险 HIGH：{n_high} / {len(df)}")
    if "fill_risk_hint" in df.columns and df["fill_risk_hint"].notna().any():
        risk_summary.append("- 买不到风险：已从上游字段透传（本模块不参与排序）")

    # 若有真实对照，给评估
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

    # 组装 Markdown
    lines = []
    lines.append(f"# Premium 溢价预测排序（{trade_date}）")
    lines.append("")
    lines.append(f"- 预测目标：PremiumRet(2→3) = Close[3]/Close[2] - 1")
    lines.append(f"- trade_date（第2日）：**{trade_date}**")
    lines.append(f"- next_trade_date（第3日）：**{next_td}**")
    lines.append(f"- 模型版本：**{cfg.model_version}**")
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

    lines.append("## 说明")
    lines.append("")
    lines.append("- 本名单给人类决策使用：不考虑买不到（P_fill）与组合权重。")
    lines.append("- PredEV = PredUpProb × PredRetMean，用于排序。")
    lines.append("- 若你在第3日收盘后再看本报告，将自动出现 real_premium_ret 与评估指标。")
    lines.append("")

    return "\n".join(lines)


__all__ = ["render_premium_md"]
