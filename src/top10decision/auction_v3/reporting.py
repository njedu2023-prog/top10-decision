from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AuctionV3Config


CSS = """
:root{color-scheme:light;--bg:#f4f6f8;--surface:#fff;--ink:#17202a;--muted:#66727f;--line:#dfe4e8;--green:#16794b;--red:#b42318;--amber:#9a6700;--blue:#165d9c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;letter-spacing:0}
.topbar{position:sticky;top:0;z-index:5;background:#17202a;color:#fff;border-bottom:1px solid #000;padding:14px 24px}.topbar h1{margin:0;font-size:20px}.topbar p{margin:4px 0 0;color:#c9d1d9;font-size:13px}
main{max-width:1500px;margin:0 auto;padding:22px}.status{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}.badge{display:inline-flex;align-items:center;border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 10px;font-size:13px}.badge.good{color:var(--green);border-color:#9bd2b9}.badge.warn{color:var(--amber);border-color:#e3c276}.badge.bad{color:var(--red);border-color:#e7a29d}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(140px,1fr));gap:10px;margin:0 0 20px}.metric{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:13px;min-height:82px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{display:block;margin-top:8px;font-size:21px;overflow-wrap:anywhere}
section{background:var(--surface);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 0 20px;padding:18px 0}section h2{font-size:18px;margin:0 18px 14px}.note{margin:0 18px 14px;color:var(--muted);line-height:1.65;font-size:13px}.table-wrap{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;min-width:1050px;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{position:sticky;top:77px;background:#f7f8fa;color:#4b5563;font-weight:650;z-index:2}th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}tr:hover td{background:#fafbfd}.buy{color:var(--green);font-weight:700}.reject{color:var(--red)}.pending{color:var(--amber)}
.chart{margin:0 18px;border:1px solid var(--line);background:#fff;padding:8px;min-height:230px}.chart svg{width:100%;height:210px;display:block}.chart .axis{stroke:#dfe4e8;stroke-width:1}.chart .line{fill:none;stroke:var(--blue);stroke-width:2}.footer{color:var(--muted);font-size:12px;padding:2px 0 30px;line-height:1.7}
@media(max-width:900px){main{padding:12px}.topbar{padding:12px}.metrics{grid-template-columns:repeat(2,minmax(120px,1fr))}th{top:70px}.metric strong{font-size:18px}}
"""


def _esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return html.escape(str(value))


def _num(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _pct(value: Any, digits: int = 2) -> str:
    number = _num(value)
    return "-" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def _price(value: Any) -> str:
    number = _num(value)
    return "-" if not math.isfinite(number) else f"{number:.2f}"


def _float(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return "-" if not math.isfinite(number) else f"{number:.{digits}f}"


def _metric(label: str, value: str) -> str:
    return f'<div class="metric"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>'


def _page(title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{CSS}</style></head>
<body><header class="topbar"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></header><main>{body}
<div class="footer">本页为量化研究与执行审计结果，不构成收益保证。正式状态只由严格样本外回测门槛决定；影子状态不得作为自动实盘指令。</div>
</main></body></html>"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _current_table(frame: pd.DataFrame) -> str:
    headers = ["操作", "代码", "股票", "晋阶", "原排名", "前收", "竞价上限价", "最高竞价涨幅", "预计净收益", "保守收益", "成交概率", "大亏概率", "价格动作"]
    rows: list[str] = []
    for _, row in frame.iterrows():
        action = str(row.get("action", ""))
        css = "buy" if "BUY" in action else "reject" if action == "REJECT" else "pending"
        values = [
            f'<span class="{css}">{_esc(action)}</span>',
            _esc(row.get("ts_code")),
            _esc(row.get("name")),
            _esc(row.get("stage")),
            _float(row.get("source_rank"), 0),
            _price(row.get("d_close")),
            _price(row.get("recommended_max_price")),
            _pct(_num(row.get("max_auction_change_pct")) / 100.0),
            _pct(row.get("predicted_net_return")),
            _pct(row.get("predicted_return_lcb")),
            _pct(row.get("predicted_fill_probability")),
            _pct(row.get("predicted_big_loss_probability")),
            _esc(row.get("price_action")),
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{_esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def current_report(prediction: pd.DataFrame, backtest: dict[str, Any]) -> str:
    signal_date = str(prediction.get("signal_date", pd.Series([""])).iloc[0]) if not prediction.empty else ""
    promoted = bool(int(_num(prediction.get("model_promoted", pd.Series([0])).iloc[0]) or 0)) if not prediction.empty else False
    selected = prediction[pd.to_numeric(prediction.get("selected"), errors="coerce").fillna(0).eq(1)] if not prediction.empty else prediction
    status_class = "good" if promoted else "warn"
    status_text = "正式模型" if promoted else "影子验证"
    body = f'<div class="status"><span class="badge {status_class}">{status_text}</span><span class="badge">信号日 {signal_date}</span><span class="badge">模型 {_esc(prediction.get("model_version", pd.Series(["-"])).iloc[0] if not prediction.empty else "-")}</span></div>'
    body += '<div class="metrics">'
    body += _metric("候选数", str(len(prediction)))
    body += _metric("入选数", str(len(selected)))
    body += _metric("回测交易日", str(backtest.get("oos_dates", 0)))
    body += _metric("回测平均净收益", _pct(backtest.get("mean_trade_net_return")))
    body += _metric("回测胜率", _pct(backtest.get("win_rate")))
    body += _metric("最大回撤", _pct(backtest.get("max_drawdown")))
    body += "</div>"
    failures = backtest.get("promotion_failures", []) or []
    fail_text = "、".join(str(x) for x in failures) if failures else "全部通过"
    body += '<section><h2>当日竞价价格指导</h2><p class="note">最高竞价价是冻结的研究限价。真实开盘价超过上限即视为不成交；模型未晋级时，所有 SHADOW_ONLY 只用于逐笔真价验证，严禁作为实盘买入指令。晋级门槛：' + _esc(fail_text) + "。</p>"
    body += _current_table(prediction.head(20)) + "</section>"
    return _page("Decision Auction V3 竞价隔夜决策", "D日收盘生成，T日竞价限价买入，T+1固定规则退出", body)


def _verification_table(frame: pd.DataFrame) -> str:
    headers = ["信号日", "代码", "股票", "预测操作", "建议上限", "实际买价", "实际卖价", "预计净收益", "实际净收益", "真值来源", "状态", "预测结果"]
    rows: list[str] = []
    for _, row in frame.iterrows():
        status = str(row.get("verification_status", ""))
        trade_success = _num(row.get("trade_success"))
        verdict = "待验证"
        css = "pending"
        if status == "VERIFIED":
            verdict = "成功" if trade_success == 1 else "失败"
            css = "buy" if trade_success == 1 else "reject"
        elif status in ("NO_FILL", "COUNTERFACTUAL_READY"):
            verdict = "未成交"
        values = [
            _esc(row.get("signal_date")),
            _esc(row.get("ts_code")),
            _esc(row.get("name")),
            _esc(row.get("action")),
            _price(row.get("recommended_max_price")),
            _price(row.get("actual_buy_price")),
            _price(row.get("actual_exit_price")),
            _pct(row.get("predicted_net_return")),
            _pct(row.get("actual_net_return")),
            _esc(row.get("truth_source")),
            _esc(status),
            f'<span class="{css}">{_esc(verdict)}</span>',
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{_esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def verification_report(ledger: pd.DataFrame, cumulative: dict[str, Any]) -> str:
    body = '<div class="status"><span class="badge">逐笔冻结预测与真价对照</span><span class="badge">broker_actual 优先，market_proxy 次之</span></div>'
    body += '<div class="metrics">'
    body += _metric("冻结预测", str(cumulative.get("frozen_predictions", 0)))
    body += _metric("已验证交易", str(cumulative.get("verified_trades", 0)))
    body += _metric("成交率", _pct(cumulative.get("fill_rate")))
    body += _metric("价格指导准确率", _pct(cumulative.get("price_guidance_accuracy")))
    body += _metric("实际胜率", _pct(cumulative.get("win_rate")))
    body += _metric("方向准确率", _pct(cumulative.get("direction_accuracy")))
    body += "</div>"
    body += '<section><h2>每笔交易真实价格验证</h2><p class="note">未成交不计作收益为零；一字跌停延迟退出会持续等待首次可成交日。只有实际券商成交文件存在时才标记 broker_actual。</p>'
    latest = ledger.sort_values(["signal_date", "source_rank"], ascending=[False, True]).head(300) if not ledger.empty else ledger
    body += _verification_table(latest) + "</section>"
    return _page("Auction V3 逐笔真价验证", "预测价、实际竞价价、实际退出价和成功判断逐笔可追溯", body)


def _equity_chart(points: Iterable[dict[str, Any]]) -> str:
    data = [p for p in points if _num(p.get("nav")) > 0]
    if len(data) < 2:
        return '<div class="chart"><p class="note">样本外净值数据尚不足。</p></div>'
    values = np.asarray([_num(p["nav"]) for p in data], dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    spread = max(vmax - vmin, 0.01)
    width, height, pad = 1000.0, 200.0, 18.0
    coords = []
    for i, value in enumerate(values):
        x = pad + i * (width - 2 * pad) / (len(values) - 1)
        y = height - pad - (value - vmin) * (height - 2 * pad) / spread
        coords.append(f"{x:.1f},{y:.1f}")
    return f'<div class="chart"><svg viewBox="0 0 1000 200" role="img" aria-label="样本外累计净值"><line class="axis" x1="18" y1="182" x2="982" y2="182"/><polyline class="line" points="{" ".join(coords)}"/></svg></div>'


def dashboard(backtest: dict[str, Any], cumulative: dict[str, Any]) -> str:
    promoted = backtest.get("promoted") is True
    failures = backtest.get("promotion_failures", []) or []
    body = f'<div class="status"><span class="badge {"good" if promoted else "warn"}">{"已通过晋级" if promoted else "未通过晋级"}</span><span class="badge">严格按日期滚动样本外</span><span class="badge">2倍成本压力测试</span></div>'
    body += '<div class="metrics">'
    body += _metric("历史独立日", str(backtest.get("history_dates", 0)))
    body += _metric("样本外交易日", str(backtest.get("oos_dates", 0)))
    body += _metric("累计收益", _pct(backtest.get("cumulative_return")))
    body += _metric("平均日收益", _pct(backtest.get("mean_daily_return"), 3))
    body += _metric("2倍成本日收益", _pct(backtest.get("stress_2x_cost_mean_daily_return"), 3))
    body += _metric("Bootstrap盈利概率", _pct(backtest.get("bootstrap_probability_mean_positive")))
    body += "</div>"
    body += '<section><h2>样本外累计净值</h2>' + _equity_chart(backtest.get("daily_equity", []) or []) + "</section>"
    body += '<section><h2>模型晋级审计</h2><p class="note">当前未通过项目：' + _esc("、".join(str(x) for x in failures) if failures else "无") + "。任何一个硬门槛失败，系统都保持影子状态。</p>"
    checks = backtest.get("promotion_checks", {}) or {}
    rows = "".join(f'<tr><td>{_esc(name)}</td><td><span class="{"buy" if passed else "reject"}">{"通过" if passed else "未通过"}</span></td></tr>' for name, passed in checks.items())
    body += '<div class="table-wrap"><table><thead><tr><th>检查项</th><th>结果</th></tr></thead><tbody>' + rows + "</tbody></table></div></section>"
    body += '<section><h2>上线后累计真价验证</h2><div class="metrics">'
    body += _metric("已验证交易", str(cumulative.get("verified_trades", 0)))
    body += _metric("平均实际净收益", _pct(cumulative.get("mean_actual_net_return")))
    body += _metric("实际胜率", _pct(cumulative.get("win_rate")))
    body += _metric("预测MAE", _pct(cumulative.get("forecast_mae")))
    body += _metric("正确放弃率", _pct(cumulative.get("correct_rejection_rate")))
    body += _metric("机会遗漏率", _pct(cumulative.get("missed_opportunity_rate")))
    body += "</div></section>"
    return _page("Auction V3 竞价回测与累计验证", "模型是否具备实盘资格由样本外收益、压力测试和逐笔真价共同决定", body)


def write_reports(
    config: AuctionV3Config,
    *,
    prediction: pd.DataFrame,
    ledger: pd.DataFrame,
    backtest_trades: pd.DataFrame,
    backtest_metrics: dict[str, Any],
    cumulative_metrics: dict[str, Any],
) -> dict[str, str]:
    del backtest_trades
    signal_date = str(prediction.get("signal_date", pd.Series([""])).iloc[0]) if not prediction.empty else "unknown"
    current = config.report_root / "auction_v3_latest.html"
    current_dated = config.report_root / f"auction_v3_{signal_date}.html"
    verification = config.report_root / "auction_v3_verify_latest.html"
    validation = config.report_root / "auction_v3_validation_dashboard.html"
    current_html = current_report(prediction, backtest_metrics)
    _write(current, current_html)
    _write(current_dated, current_html)
    _write(verification, verification_report(ledger, cumulative_metrics))
    _write(validation, dashboard(backtest_metrics, cumulative_metrics))
    return {"current": str(current), "verification": str(verification), "dashboard": str(validation)}
