#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Predict（V1：手工交易版，端到端落盘）

锁死契约（来自 Premium.md）：
- 主输入：data/pred/pred_source_latest.csv（a-top10 全量源表，候选=全量，不过滤）
- decision 产物：仅用于字段合并/标签（不得过滤），输出字段需带 dec_ 前缀
- horizon = 2 个交易日（T -> T+2），非交易日顺延
- 行情真值：data/market/daily_YYYYMMDD.csv（由 Market Truth Layer 负责生成/缓存）
- 行情未到：pending（不得报错卡死）
- 验证表顺序必须与预测表 Top30 完全一致（不可重新排序）
- 输出：
  outputs/premium/premium_top30_{T}.csv
  outputs/premium/premium_full_{T}.csv
  outputs/premium/premium_verify_{T}.csv
  docs/reports/premium_{T}.md
  docs/reports/premium_latest.md
  outputs/premium/_last_run.txt（每次覆盖）

说明：
- 本文件不再依赖旧的 LR/LGBM/feature/labels 链路（避免主线被训练模块牵制）
- 预测字段（p_premium/e_premium/score_ev 等）默认从 pred_source_latest 中就地读取；
  若缺失则给出安全兜底（p=0.5, e=0.0, score=p*e）。
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PremiumConfig
from .market_truth import ensure_daily_cached, load_daily


_TD_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class PredictResult:
    ok: bool
    trade_date: str
    target_date: Optional[str]
    pending: bool
    reason: str
    out_top30: Optional[str] = None
    out_full: Optional[str] = None
    out_verify: Optional[str] = None
    report_md: Optional[str] = None


# ========= 通用工具 =========

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_run_id() -> str:
    rid = os.getenv("GITHUB_RUN_ID", "").strip()
    rno = os.getenv("GITHUB_RUN_NUMBER", "").strip()
    if rid:
        return f"gh_{rid}"
    if rno:
        return f"ghno_{rno}"
    return datetime.now(timezone.utc).strftime("local_%Y%m%d%H%M%S")


def _get_commit_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12] if out else "unknown"
    except Exception:
        return "unknown"


def _to_yyyymmdd(x: object) -> str:
    s = str(x).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "")
    return s[:8]


def _read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        return pd.read_csv(path)
    return pd.read_csv(path)


def _ensure_dirs(cfg: PremiumConfig) -> None:
    cfg.out_root().mkdir(parents=True, exist_ok=True)
    cfg.reports_root().mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_last_run(cfg: PremiumConfig, trade_date: str, extra: Dict[str, object]) -> None:
    _ensure_dirs(cfg)
    lines = [
        f"trade_date: {trade_date}",
        f"run_id: {_get_run_id()}",
        f"commit_sha: {_get_commit_sha(cfg.repo_root())}",
        f"created_at_utc: {_utc_now_iso()}",
    ]
    for k, v in extra.items():
        lines.append(f"{k}: {v}")
    _write_text(cfg.out_last_run_path(), "\n".join(lines) + "\n")


# ========= 交易日历推进（按 ensure_daily_cached 探测）=========

def _probe_next_trade_day(cfg: PremiumConfig, start_trade_date: str, max_probe_days: int = 30) -> Optional[str]:
    """
    从 start_trade_date 的次日自然日开始探测，第一天 ensure_daily_cached 成功 -> 视为交易日。
    """
    import datetime as dt

    start_trade_date = _to_yyyymmdd(start_trade_date)
    if not _TD_RE.match(start_trade_date):
        return None

    d0 = dt.datetime.strptime(start_trade_date, "%Y%m%d").date()
    for i in range(1, int(max_probe_days) + 1):
        d = d0 + dt.timedelta(days=i)
        cand = d.strftime("%Y%m%d")
        r = ensure_daily_cached(cfg, cand)
        if r.ok:
            return cand
    return None


def _advance_trade_days(cfg: PremiumConfig, trade_date: str, steps: int) -> Optional[str]:
    """
    按交易日推进 steps 次：T -> T+steps
    """
    cur = _to_yyyymmdd(trade_date)
    for _ in range(int(steps)):
        nxt = _probe_next_trade_day(cfg, cur, max_probe_days=40)
        if not nxt:
            return None
        cur = nxt
    return cur


# ========= pred_source_latest 读取与字段兜底 =========

def _normalize_pred_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols_map:
                return cols_map[n.lower()]
        return None

    c_date = pick("trade_date", "date", "dt", "交易日期", "日期")
    c_code = pick("ts_code", "code", "symbol", "ticker", "股票代码", "代码")
    c_name = pick("name", "stock_name", "股票名称", "名称")

    if c_date:
        df["trade_date"] = df[c_date].astype(str).map(_to_yyyymmdd)
    if c_code:
        df["ts_code"] = df[c_code].astype(str).str.strip()
    if c_name:
        df["name"] = df[c_name].astype(str).str.strip()

    return df


def _infer_trade_date(df: pd.DataFrame) -> str:
    if "trade_date" in df.columns:
        s = df["trade_date"].dropna().astype(str).map(_to_yyyymmdd)
        s = s[s.str.match(r"^\d{8}$", na=False)]
        if not s.empty:
            u = sorted(s.unique().tolist())
            return u[-1]
    # 兜底：全表扫描
    for c in df.columns:
        s = df[c].dropna().astype(str).map(_to_yyyymmdd)
        s = s[s.str.match(r"^\d{8}$", na=False)]
        if not s.empty:
            u = sorted(s.unique().tolist())
            return u[-1]
    return "unknown"


def _pick_pred_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    从 pred_source_latest “就地取材”生成 Premium 必需字段：
    - p_premium：上涨概率（0~1）
    - e_premium：上涨幅度预测（ratio，例如 0.0911 表示 +9.11%）
    - score_ev：综合分值
    - risk_flags/confidence/data_quality：风险/质量提示（可缺省）
    """
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_prob = col("p_premium", "up_prob", "probability", "prob", "p")
    c_ret = col("e_premium", "pred_ret", "pred_return", "ret", "premium_ret", "pred_premium_ret", "pred_ret_mean")
    c_score = col("score_ev", "ev", "final_score", "score", "pred_ev")
    c_conf = col("confidence", "conf")
    c_dq = col("data_quality", "dq")
    c_risk = col("risk_flags", "risk", "warning", "risk_hint", "fill_risk_hint")

    p = pd.to_numeric(df[c_prob], errors="coerce") if c_prob else pd.Series([np.nan] * len(df))
    p = p.fillna(0.5).clip(0.0, 1.0)

    e = pd.to_numeric(df[c_ret], errors="coerce") if c_ret else pd.Series([np.nan] * len(df))
    e = e.fillna(0.0)

    if c_score:
        s = pd.to_numeric(df[c_score], errors="coerce").fillna(p * e)
    else:
        s = (p * e).astype(float)

    conf = pd.to_numeric(df[c_conf], errors="coerce") if c_conf else pd.Series([pd.NA] * len(df))
    dq = pd.to_numeric(df[c_dq], errors="coerce") if c_dq else pd.Series([pd.NA] * len(df))
    risk = df[c_risk].astype(str) if c_risk else pd.Series([""] * len(df))

    return p, e, s, conf, dq, risk


# ========= decision merge（仅标签）=========

def _load_decision_merge(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    repo_root = cfg.repo_root()
    paths = [Path(p).resolve() for p in glob.glob(str((repo_root / cfg.decision_glob).resolve()))]
    if not paths:
        return pd.DataFrame()

    hit = []
    for p in sorted(paths):
        try:
            d = _read_csv_smart(p)
            if d is None or d.empty:
                continue
            d = _normalize_pred_source(d)
            if "trade_date" not in d.columns or "ts_code" not in d.columns:
                continue
            if (d["trade_date"].astype(str) == str(trade_date)).any():
                hit.append(d)
        except Exception:
            continue

    if not hit:
        return pd.DataFrame()

    dec = pd.concat(hit, ignore_index=True)
    dec["trade_date"] = dec["trade_date"].astype(str).map(_to_yyyymmdd)
    dec["ts_code"] = dec["ts_code"].astype(str).str.strip()
    if "name" in dec.columns:
        dec["name"] = dec["name"].astype(str).str.strip()

    cols = {str(c).strip().lower(): c for c in dec.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    out = pd.DataFrame({
        "trade_date": dec["trade_date"].astype(str),
        "ts_code": dec["ts_code"].astype(str),
    })

    if "name" in dec.columns:
        out["name"] = dec["name"]

    # 映射到 dec_ 前缀字段（缺失留空）
    m_rank = pick("dec_rank", "decision_rank", "rank", "决策排名")
    m_w = pick("dec_weight", "weight", "target_weight", "决策权重")
    m_can = pick("dec_can_buy", "can_buy", "可买提示")
    m_pf = pick("dec_p_fill", "p_fill", "P_fill")
    m_reason = pick("dec_reason", "reason", "label", "决策原因", "决策标签")

    out["dec_rank"] = dec[m_rank] if m_rank else pd.NA
    out["dec_weight"] = dec[m_w] if m_w else pd.NA
    out["dec_can_buy"] = dec[m_can] if m_can else pd.NA
    out["dec_p_fill"] = dec[m_pf] if m_pf else pd.NA
    out["dec_reason"] = dec[m_reason] if m_reason else pd.NA

    out = out.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)
    return out


# ========= 报告渲染（使用 PremiumA/B.html 模板）=========

def _fmt_prob(x: object) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        return f"{v * 100:.2f}%"
    except Exception:
        return "-"


def _fmt_pct_ratio(x: object) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "-"
        sign = "↑" if v >= 0 else "↓"
        return f"{sign} {v:+.2%}"
    except Exception:
        return "-"


def _row_td(v: str, strong: bool = False, color_red_if_up: bool = False) -> str:
    style = "border:1px solid #111; padding:8px;"
    if strong:
        style += " font-weight:700;"
    if color_red_if_up and isinstance(v, str) and v.startswith("↑"):
        style += " color:#d00;"
    return f'<td style="{style}">{v}</td>'


def _build_rows_pred(df_top: pd.DataFrame) -> str:
    rows = []
    for _, r in df_top.iterrows():
        rows.append(
            "<tr>"
            + _row_td(str(r.get("rank", "")))
            + _row_td(str(r.get("trade_date", "")))
            + _row_td(str(r.get("target_date", "")))
            + _row_td(str(r.get("ts_code", "")))
            + _row_td(str(r.get("name", "")))
            + _row_td(_fmt_prob(r.get("p_premium", np.nan)))
            + _row_td(_fmt_pct_ratio(r.get("e_premium", np.nan)), strong=True, color_red_if_up=True)
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
        pred_v = _fmt_pct_ratio(r.get("e_premium", np.nan))
        act_v = _fmt_pct_ratio(r.get("actual_ret", np.nan))
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


def _render_md(cfg: PremiumConfig,
               trade_date: str,
               target_date: str,
               df_top: pd.DataFrame,
               df_verify: pd.DataFrame,
               verify_pending: bool,
               verify_reason: str,
               gen_ts: str) -> str:
    repo_root = cfg.repo_root()
    tpl_a = (repo_root / "PremiumA.html").resolve()
    tpl_b = (repo_root / "PremiumB.html").resolve()

    html_a = tpl_a.read_text(encoding="utf-8")
    html_b = tpl_b.read_text(encoding="utf-8")

    title_a = f"{trade_date} → {target_date}　TOP 30 溢价概率研究报告"
    html_a = (
        html_a.replace("{{TITLE}}", title_a)
        .replace("{{ROWS}}", _build_rows_pred(df_top))
        .replace("{{GEN_TS}}", gen_ts)
        .replace("{{TRADE_DATE}}", trade_date)
        .replace("{{TARGET_DATE}}", target_date)
    )

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
        hit = (df_verify["hit_up"].astype(str) == "是").sum()
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

    md = []
    md.append("# Premium（溢价预测）手工交易版 V1\n\n")
    md.append(f"- trade_date（T）：**{trade_date}**\n")
    md.append(f"- target_date（T+2）：**{target_date}**\n")
    md.append(f"- horizon：**2 个交易日（T→T+2）**\n")
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


# ========= 主入口 =========

def predict_latest(cfg: Optional[PremiumConfig] = None) -> PredictResult:
    cfg = cfg or PremiumConfig.load()
    _ensure_dirs(cfg)

    # 1) 读主输入（全量）
    pred_path = cfg.pred_source_latest_path()
    if not pred_path.exists():
        _write_last_run(cfg, "unknown", {"ok": False, "reason": f"pred_source_latest_not_found: {pred_path}"})
        return PredictResult(False, "unknown", None, True, f"未找到主输入：{pred_path}")

    df0 = _read_csv_smart(pred_path)
    if df0 is None or df0.empty:
        _write_last_run(cfg, "unknown", {"ok": False, "reason": "pred_source_latest_empty"})
        return PredictResult(False, "unknown", None, True, "pred_source_latest 为空")

    df0 = _normalize_pred_source(df0)
    trade_date = _infer_trade_date(df0)
    if trade_date == "unknown":
        _write_last_run(cfg, "unknown", {"ok": False, "reason": "cannot_infer_trade_date"})
        return PredictResult(False, "unknown", None, True, "无法从 pred_source_latest 推断 trade_date")

    # 保底 key 列
    if "trade_date" not in df0.columns:
        df0["trade_date"] = trade_date
    else:
        df0["trade_date"] = df0["trade_date"].astype(str).map(_to_yyyymmdd)
    if "ts_code" not in df0.columns:
        _write_last_run(cfg, trade_date, {"ok": False, "reason": "missing_ts_code"})
        return PredictResult(False, trade_date, None, True, "pred_source_latest 缺少 ts_code（或别名 code）")

    df0["ts_code"] = df0["ts_code"].astype(str).str.strip()
    if "name" not in df0.columns:
        df0["name"] = pd.NA

    # 2) 交易日历推进到 T+2
    target_date = _advance_trade_days(cfg, trade_date, cfg.horizon_trade_days)
    pending = False
    if not target_date:
        pending = True
        target_date = ""
        pending_reason = "无法推进到 T+2（可能行情未到/数据源缺失）"
    else:
        pending_reason = "ok"

    # 3) decision merge（仅标签，不过滤）
    dec = _load_decision_merge(cfg, trade_date)
    df = df0.copy()
    if not dec.empty:
        m = df.merge(dec, on=["trade_date", "ts_code"], how="left", suffixes=("", "_dec"))
        # name 补全
        if "name_dec" in m.columns:
            m["name"] = m["name"].where(m["name"].notna() & (m["name"].astype(str).str.strip() != ""), m["name_dec"])
            m = m.drop(columns=["name_dec"])
        df = m
    else:
        for c in ("dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason"):
            df[c] = pd.NA

    # 4) 生成 Premium 字段（就地取材 + 兜底）
    p, e, s, conf, dq, risk = _pick_pred_fields(df)
    df["p_premium"] = p
    df["e_premium"] = e
    df["score_ev"] = s
    df["confidence"] = conf
    df["data_quality"] = dq
    df["risk_flags"] = risk

    df["target_date"] = target_date if target_date else pd.NA

    # 5) 排序：score_ev 降序（默认综合）
    df = df.sort_values(by=["score_ev"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    # 6) top30 + full
    topn = int(cfg.top_n)
    df_top = df.head(topn).copy()
    df_full = df.copy()

    out_cols = [
        "rank", "trade_date", "target_date", "ts_code", "name",
        "p_premium", "e_premium", "score_ev", "risk_flags", "confidence", "data_quality",
        "dec_rank", "dec_weight", "dec_can_buy", "dec_p_fill", "dec_reason",
    ]
    out_top = df_top[out_cols].copy()
    out_full = df_full[out_cols].copy()

    p_top = _write_csv(cfg.out_top30_csv(trade_date), out_top)
    p_full = _write_csv(cfg.out_full_csv(trade_date), out_full)

    # 7) 验证表（依赖 close(T) 与 close(T+2)）
    verify_pending = True
    verify_reason = "pending"
    df_verify = pd.DataFrame(columns=["rank", "trade_date", "target_date", "ts_code", "name", "e_premium", "actual_ret", "hit_up"])

    if target_date:
        r_t = ensure_daily_cached(cfg, trade_date)
        r_t2 = ensure_daily_cached(cfg, target_date)
        if r_t.ok and r_t2.ok:
            d0 = load_daily(cfg, trade_date)[["ts_code", "close"]].rename(columns={"close": "close_T"})
            d2 = load_daily(cfg, target_date)[["ts_code", "close"]].rename(columns={"close": "close_T2"})
            tmp = out_top.merge(d0, on="ts_code", how="left").merge(d2, on="ts_code", how="left")
            tmp["actual_ret"] = tmp["close_T2"] / tmp["close_T"] - 1
            tmp["hit_up"] = tmp["actual_ret"].apply(lambda x: "是" if pd.notna(x) and float(x) > 0 else ("否" if pd.notna(x) else ""))
            df_verify = tmp[["rank", "trade_date", "target_date", "ts_code", "name", "e_premium", "actual_ret", "hit_up"]].copy()
            verify_pending = False
            verify_reason = "ok"
        else:
            verify_pending = True
            verify_reason = f"truth_not_ready: T_ok={r_t.ok} T2_ok={r_t2.ok}"
    else:
        verify_pending = True
        verify_reason = pending_reason

    p_verify = _write_csv(cfg.out_verify_csv(trade_date), df_verify)

    # 8) 渲染报告 md（两张表）
    gen_ts = _utc_now_iso()
    md = _render_md(
        cfg=cfg,
        trade_date=trade_date,
        target_date=target_date if target_date else "",
        df_top=out_top,
        df_verify=df_verify,
        verify_pending=verify_pending,
        verify_reason=verify_reason,
        gen_ts=gen_ts,
    )
    p_md = _write_text(cfg.report_md_path(trade_date), md)
    _write_text(cfg.report_latest_md_path(), md)

    # 9) last_run
    _write_last_run(cfg, trade_date, {
        "ok": True,
        "target_date": target_date,
        "pending": bool(pending or verify_pending),
        "verify_pending": bool(verify_pending),
        "verify_reason": verify_reason,
        "out_top30": str(p_top),
        "out_full": str(p_full),
        "out_verify": str(p_verify),
        "report_md": str(p_md),
    })

    return PredictResult(
        ok=True,
        trade_date=trade_date,
        target_date=(target_date if target_date else None),
        pending=bool(pending or verify_pending),
        reason="pending" if (pending or verify_pending) else "ok",
        out_top30=str(p_top),
        out_full=str(p_full),
        out_verify=str(p_verify),
        report_md=str(p_md),
    )


__all__ = ["PredictResult", "predict_latest"]
