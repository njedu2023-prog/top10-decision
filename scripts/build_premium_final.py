#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Premium Final Buy List artifacts.

This script is a Premium-only post-processing step. It reads existing Premium
outputs and writes additional final execution artifacts without changing the
upstream Premium ranking engine or Decision/a-top10/a-share-top3-data mainlines.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig  # noqa: E402
from top10decision.premium.execution_profit_model import score_execution_candidates  # noqa: E402
from top10decision.premium.final_decision import build_final_decisions, final_display_columns  # noqa: E402


DATE_RE = re.compile(r"(20\d{6})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _inject_execution_link(path: Path, href: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    if 'data-execution-report="1"' in text:
        text = re.sub(
            r'(<a[^>]+data-execution-report="1"[^>]+href=")[^"]+("[^>]*>)',
            rf"\1{href}\2",
            text,
            count=1,
        )
        path.write_text(text, encoding="utf-8")
        return
    marker = '<div class="tabs" role="tablist" aria-label="列表切换">'
    link = (
        f'<a class="tab-btn" data-execution-report="1" href="{_h(href)}" '
        'title="打开净收益执行名单">净收益执行名单</a>'
    )
    if marker in text:
        text = text.replace(marker, marker + link, 1)
        path.write_text(text, encoding="utf-8")


def _date_from_name(path: Path) -> Optional[str]:
    m = DATE_RE.search(path.name)
    return m.group(1) if m else None


def _latest_date(cfg: PremiumConfig) -> str:
    last = cfg.out_last_run_path()
    if last.exists():
        for line in last.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("trade_date:"):
                value = line.split(":", 1)[1].strip()
                if re.fullmatch(r"20\d{6}", value):
                    return value
    dates = []
    for p in cfg.out_root().glob("premium_full_*.csv"):
        d = _date_from_name(p)
        if d:
            dates.append(d)
    if not dates:
        raise FileNotFoundError("no outputs/premium/premium_full_YYYYMMDD.csv artifacts found")
    return sorted(dates)[-1]


def _load_premium_frame(cfg: PremiumConfig, trade_date: str) -> pd.DataFrame:
    for path in (
        cfg.out_full_csv(trade_date),
        cfg.out_top30_csv(trade_date),
        cfg.out_top20_csv(trade_date),
        cfg.out_top10_csv(trade_date),
    ):
        if path.exists():
            df = _read_csv_smart(path)
            if not df.empty:
                return df
    raise FileNotFoundError(f"no Premium CSV found for {trade_date}")


def _load_history(cfg: PremiumConfig) -> pd.DataFrame:
    path = cfg.out_learning_dir() / "limitup_probability_training_samples.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_csv_smart(path)
    except Exception:
        return pd.DataFrame()


def _fmt_pct(value: object) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "-"
        return f"{v * 100:.1f}%"
    except Exception:
        return "-"


def _fmt_num(value: object, digits: int = 2) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def _h(value: object) -> str:
    s = str(value if value is not None else "")
    if s.lower() in {"nan", "none", "<na>", "nat"}:
        s = ""
    return html.escape(s, quote=True)


def _cn_action(value: object) -> str:
    return {
        "BUY_MARKET": "市价竞价",
        "BUY_LIMIT": "限价竞价",
        "SMALL_BUY": "小仓试买",
        "WATCH_ONLY": "只观察不追",
        "REJECT": "禁止买入",
    }.get(str(value), str(value))


def _cn_reason(value: object) -> str:
    mapping = {
        "market_no_trade": "市场模式禁止交易",
        "premium_reject": "Premium动作为放弃",
        "premium_watch_only": "Premium动作为只观察不追",
        "premium_excluded": "Premium专业门槛剔除",
        "decision_veto": "Decision否决",
        "minute_hard_risk": "分钟硬风险",
        "minute_risk_penalty": "分钟风险惩罚高",
        "t1_fail_risk": "T+1失败概率高",
        "t1_drawdown_risk": "T+1大回撤概率高",
        "execution_model_guard": "净收益模型样本外验证未通过，仅观察",
        "low_fill_probability": "竞价可成交概率不足",
        "nonpositive_expected_net_return": "保守净收益不足",
        "profit_edge_not_positive": "盈利优势不足",
        "large_loss_risk": "大亏风险过高",
        "execution_ev_gate": "净收益执行门槛未通过",
        "market_cap_overflow": "超过市场模式允许买入数量，降为观察",
        "pass": "执行条件通过",
    }
    parts = []
    for item in str(value if value is not None else "").split(";"):
        item = item.strip()
        if not item:
            continue
        if item.startswith("premium_gate:"):
            parts.append(f"Premium门槛：{item.split(':', 1)[1]}")
        else:
            parts.append(mapping.get(item, item))
    return "；".join(parts)


def _cn_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "FinalRank": "最终排名",
            "PremiumRank": "Premium排名",
            "Code": "代码",
            "Name": "股票",
            "Sector": "行业",
            "Action": "最终动作",
            "FinalScore": "最终分",
            "PFill": "可成交率",
            "PProfit": "盈利率",
            "PLimit": "涨停率",
            "PBigLoss": "大亏率",
            "NetEV": "净收益EV",
            "ProfitEdge": "盈利优势",
            "ModelState": "模型状态",
            "Position": "建议仓位",
            "MaxBuyPrice": "最高买入价",
            "T-Up": "T日涨停概率",
            "T-Strength": "T日强度",
            "T1-Up": "T+1延续率",
            "Bucket": "候选池",
            "PremiumAction": "Premium原建议",
            "Reason": "原因",
            "T+1 SellPlan": "T+1卖出计划",
        }
    )


def _cn_market(value: object) -> str:
    return {
        "NO_DATA": "无数据",
        "NO_TRADE": "禁止交易",
        "DEFENSIVE": "防守",
        "CAUTION": "谨慎",
        "NORMAL": "正常",
    }.get(str(value), str(value))


def _cn_t1_mode(value: object) -> str:
    return {
        "FROZEN_NEGATIVE_IC": "负IC冻结",
        "LOW_IC": "低IC降权",
        "NORMAL": "正常",
        "PROFIT_ACTIVE": "净收益模型启用",
        "PROFIT_GUARDED": "净收益模型保护",
        "UNKNOWN": "未知",
    }.get(str(value), str(value))


def _table_html(df: pd.DataFrame, empty_text: str) -> str:
    show = final_display_columns(df)
    if show.empty:
        return f'<p class="empty">{_h(empty_text)}</p>'
    show = _cn_columns(show)
    if "最终动作" in show.columns:
        show["最终动作"] = show["最终动作"].map(_cn_action)
    if "原因" in show.columns:
        show["原因"] = show["原因"].map(_cn_reason)
    if "建议仓位" in show.columns:
        show["建议仓位"] = show["建议仓位"].map(_fmt_pct)
    for c in (
        "T日涨停概率", "T+1延续率", "可成交率", "盈利率",
        "涨停率", "大亏率", "净收益EV", "盈利优势",
    ):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(_fmt_pct)
    for c in ("最终分", "T日强度"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: _fmt_num(x, 2))

    head = "".join(f"<th>{_h(c)}</th>" for c in show.columns)
    rows = []
    for _, row in show.iterrows():
        cells = "".join(f"<td>{_h(row.get(c, ''))}</td>" for c in show.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _render_html(
    trade_date: str,
    buy: pd.DataFrame,
    watch: pd.DataFrame,
    reject: pd.DataFrame,
    stats,
    gen_ts: str,
    backtest: Optional[dict] = None,
) -> str:
    buy_count = len(buy)
    strategy = "空仓"
    if buy_count > 0:
        actions = "、".join(sorted(set(buy.get("final_action", pd.Series(dtype=str)).astype(str).map(_cn_action).tolist())))
        strategy = f"{actions} {buy_count}只"
    t1_ic_text = "-" if not np.isfinite(stats.t1_rank_ic) else f"{stats.t1_rank_ic:.4f}"
    fill_auc_text = "-" if not np.isfinite(stats.execution_fill_auc) else f"{stats.execution_fill_auc:.4f}"
    profit_ic_text = "-" if not np.isfinite(stats.execution_profit_rank_ic) else f"{stats.execution_profit_rank_ic:.4f}"
    holdout_net_text = _fmt_pct(stats.execution_holdout_top_net)
    cost_text = "-" if not np.isfinite(stats.execution_cost_bps) else f"{stats.execution_cost_bps:.0f} bp"
    model_state = "启用" if stats.execution_model_mode == "PROFIT_ACTIVE" else "保护模式"
    backtest = backtest or {}
    backtest_fill = f"{int(backtest.get('filled_signals', 0))}/{int(backtest.get('signals', 0))}"
    backtest_win = _fmt_pct(backtest.get("filled_win_rate"))
    backtest_return = _fmt_pct(backtest.get("ten_percent_position_compound_return"))
    backtest_drawdown = _fmt_pct(backtest.get("ten_percent_position_max_drawdown"))
    recent_filled = int(backtest.get("recent_holdout_filled_signals", 0))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Premium 最终竞价买入名单 {trade_date}</title>
  <style>
    :root {{
      --bg:#f5f5f7; --panel:#fff; --ink:#1d1d1f; --muted:#6e6e73;
      --line:#d2d2d7; --accent:#b42318; --good:#0f7b55; --warn:#b26a00;
      --shadow:0 8px 24px rgba(0,0,0,.06);
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; }}
    header {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.92); border-bottom:1px solid var(--line); padding:18px 24px; backdrop-filter:blur(16px); }}
    main {{ max-width:1480px; margin:0 auto; padding:18px 24px 42px; }}
    .top {{ max-width:1480px; margin:0 auto; display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }}
    h1 {{ margin:4px 0 0; font-size:25px; letter-spacing:0; }}
    .kicker {{ color:var(--accent); font-weight:700; font-size:13px; }}
    .sub {{ color:var(--muted); margin:7px 0 0; line-height:1.6; }}
    .pill {{ display:inline-flex; border:1px solid var(--line); background:#fff; border-radius:999px; padding:8px 12px; font-size:13px; white-space:nowrap; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-bottom:16px; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); padding:14px 16px; }}
    .metric span {{ display:block; color:var(--muted); font-size:13px; }}
    .metric strong {{ display:block; margin-top:7px; font-size:22px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-top:14px; overflow:hidden; box-shadow:var(--shadow); }}
    .section-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:16px 18px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0; font-size:18px; }}
    .badge {{ border:1px solid var(--line); border-radius:999px; padding:5px 10px; color:var(--muted); font-size:12px; background:#fff; }}
    .table-wrap {{ overflow:auto; max-height:72vh; }}
    table {{ border-collapse:collapse; min-width:1180px; width:100%; }}
    th, td {{ border-bottom:1px solid #ececf0; padding:10px 12px; text-align:left; white-space:nowrap; font-size:13px; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#f7f7fa; color:#4b5563; z-index:2; }}
    tbody tr:nth-child(even) td {{ background:#fbfbfd; }}
    tbody tr:hover td {{ background:#fff7f5; }}
    th:first-child, td:first-child {{ position:sticky; left:0; z-index:1; background:inherit; }}
    .empty {{ color:var(--muted); padding:16px 18px; margin:0; }}
    .note {{ color:var(--muted); font-size:12px; line-height:1.5; margin:14px 0 0; }}
    @media(max-width:760px) {{ header {{ position:static; }} .top {{ flex-direction:column; }} main, header {{ padding-left:14px; padding-right:14px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <div class="kicker">Premium 最终交易执行层</div>
        <h1>最终竞价买入名单</h1>
        <p class="sub">只基于 Premium 产物生成，不改动 a-top10、Decision、a-share-top3-data 主线。最终买入名单才是可执行名单；观察名单和剔除名单不参与实盘买入。</p>
      </div>
      <div class="pill">D日 {trade_date}</div>
    </div>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><span>今日策略</span><strong>{_h(strategy)}</strong></div>
      <div class="metric"><span>市场模式</span><strong>{_h(_cn_market(stats.market_mode))}</strong></div>
      <div class="metric"><span>允许买入数量</span><strong>{stats.max_trade_count}</strong></div>
      <div class="metric"><span>T+1因子状态</span><strong>{_h(_cn_t1_mode(stats.t1_weight_mode))}</strong></div>
      <div class="metric"><span>T+1 Rank IC</span><strong>{_h(t1_ic_text)}</strong></div>
      <div class="metric"><span>净收益模型</span><strong>{_h(model_state)}</strong></div>
      <div class="metric"><span>可成交模型 AUC</span><strong>{_h(fill_auc_text)}</strong></div>
      <div class="metric"><span>净收益 Rank IC</span><strong>{_h(profit_ic_text)}</strong></div>
      <div class="metric"><span>样本外头部净收益</span><strong>{_h(holdout_net_text)}</strong></div>
      <div class="metric"><span>成本假设</span><strong>{_h(cost_text)}</strong></div>
      <div class="metric"><span>4个月 Walk-forward</span><strong>{_h(backtest_fill)} 成交</strong></div>
      <div class="metric"><span>成交后胜率</span><strong>{_h(backtest_win)}</strong></div>
      <div class="metric"><span>10%仓位累计</span><strong>{_h(backtest_return)}</strong></div>
      <div class="metric"><span>10%仓位 Max Drawdown</span><strong>{_h(backtest_drawdown)}</strong></div>
      <div class="metric"><span>近期独立成交样本</span><strong>{recent_filled} 笔</strong></div>
      <div class="metric"><span>生成时间</span><strong>{_h(gen_ts)}</strong></div>
    </div>
    <section>
      <div class="section-head"><h2>最终买入名单</h2><span class="badge">最多 {stats.max_trade_count} 只，可执行</span></div>
      {_table_html(buy, "今日无可执行买入名单，策略为空仓或只观察。")}
    </section>
    <section>
      <div class="section-head"><h2>观察名单</h2><span class="badge">观察，不追</span></div>
      {_table_html(watch.head(30), "暂无观察名单")}
    </section>
    <section>
      <div class="section-head"><h2>剔除名单</h2><span class="badge">禁止买入</span></div>
      {_table_html(reject.head(50), "暂无剔除名单")}
    </section>
    <p class="note">规则摘要：最终排名以竞价可成交概率、净盈利概率、T日涨停概率和大亏风险共同计算；样本外门槛未通过、保守净收益不足或没有可执行候选时自动空仓。Decision 否决、分钟硬风险和市场禁止交易仍会直接剔除。模型状态：{_h(stats.execution_model_reason)}。历史回测不能保证未来收益，近期独立成交样本仍不足。</p>
  </main>
</body>
</html>
"""


def build(cfg: PremiumConfig, trade_date: str, verbose: bool = False) -> int:
    frame = _load_premium_frame(cfg, trade_date)
    history = _load_history(cfg)
    scored, execution_diagnostics = score_execution_candidates(
        frame,
        out_root=cfg.out_root(),
        market_root=cfg.market_cache_root(),
        trade_date=trade_date,
    )
    buy, watch, reject, stats = build_final_decisions(scored, trade_date=trade_date, history=history)

    out_root = cfg.out_root()
    report_root = cfg.reports_root()
    p_buy = _write_csv(out_root / f"premium_final_buy_{trade_date}.csv", buy)
    p_watch = _write_csv(out_root / f"premium_final_watch_{trade_date}.csv", watch)
    p_reject = _write_csv(out_root / f"premium_final_reject_{trade_date}.csv", reject)
    shutil.copyfile(p_buy, out_root / "premium_final_buy_latest.csv")
    shutil.copyfile(p_watch, out_root / "premium_final_watch_latest.csv")
    shutil.copyfile(p_reject, out_root / "premium_final_reject_latest.csv")
    meta_payload = execution_diagnostics.as_dict()
    _write_json(out_root / "models" / f"execution_profit_model_meta_{trade_date}.json", meta_payload)
    _write_json(out_root / "models" / "execution_profit_model_meta.json", meta_payload)

    backtest_meta = _read_json(out_root / "models" / "execution_profit_backtest_meta.json")
    html_text = _render_html(trade_date, buy, watch, reject, stats, _utc_now_iso(), backtest_meta)
    p_html = _write_text(report_root / f"premium_final_{trade_date}.html", html_text)
    _write_text(report_root / "premium_final_latest.html", html_text)
    _inject_execution_link(report_root / f"premium_{trade_date}.html", f"premium_final_{trade_date}.html")
    _inject_execution_link(report_root / "premium_latest.html", "premium_final_latest.html")

    if verbose:
        print(f"[premium-final] trade_date={trade_date}")
        print(f"[premium-final] buy={len(buy)} watch={len(watch)} reject={len(reject)}")
        print(f"[premium-final] market_mode={stats.market_mode} max_trade_count={stats.max_trade_count}")
        print(f"[premium-final] t1_weight_mode={stats.t1_weight_mode} t1_rank_ic={stats.t1_rank_ic}")
        print(
            f"[premium-final] execution_model={execution_diagnostics.reason} "
            f"fill_auc={execution_diagnostics.fill_auc} "
            f"profit_rank_ic={execution_diagnostics.profit_rank_ic}"
        )
        print(f"[premium-final] out_buy={p_buy}")
        print(f"[premium-final] out_watch={p_watch}")
        print(f"[premium-final] out_reject={p_reject}")
        print(f"[premium-final] report_html={p_html}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Premium final executable buy list.")
    p.add_argument("--trade-date", default="", help="YYYYMMDD; default reads outputs/premium/_last_run.txt")
    p.add_argument("--link-only", action="store_true", help="Only restore links from the main report")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = PremiumConfig.load()
    trade_date = args.trade_date.strip() or _latest_date(cfg)
    if not re.fullmatch(r"20\d{6}", trade_date):
        raise SystemExit(f"bad trade_date: {trade_date}")
    if args.link_only:
        _inject_execution_link(
            cfg.reports_root() / f"premium_{trade_date}.html",
            f"premium_final_{trade_date}.html",
        )
        _inject_execution_link(cfg.reports_root() / "premium_latest.html", "premium_final_latest.html")
        return 0
    return build(cfg, trade_date, verbose=bool(args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
