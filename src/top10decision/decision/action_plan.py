from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .eligibility import annotate_standard_limit_universe, filter_standard_limit_universe


REPORT_RE = re.compile(r"decision_report_(20\d{6})\.md$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return round(number, 10) if math.isfinite(number) else None


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    return int(number) if number is not None else default


def _text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _report_dates(root: Path) -> list[str]:
    dates: list[str] = []
    for path in (root / "outputs" / "decision").glob("decision_report_*.md"):
        match = REPORT_RE.fullmatch(path.name)
        if match:
            dates.append(match.group(1))
    return sorted(set(dates), reverse=True)


def _candidate_path(root: Path, evaluation: dict[str, Any], signal_date: str) -> Path:
    raw = str((evaluation.get("paths", {}) or {}).get("candidates", "") or "").strip()
    if raw:
        path = Path(raw)
        path = path if path.is_absolute() else root / path
        if path.exists():
            return path
    return root / "data" / "decision" / f"decision_candidates_{signal_date}.csv"


def _industry(row: pd.Series) -> str:
    for column in ("industry", "industry_tag", "行业", "行业板块", "board"):
        if column in row.index:
            value = _text(row.get(column))
            if value:
                return value
    return "未分类"


def _rejection_reason(value: Any) -> str:
    reason = _text(value)
    return {
        "no_safe_price": "没有通过成本、成交、退出和尾部风险约束的安全竞价价格",
        "big_loss_probability_exceeds_cap": "预测大跌概率超过15%硬上限，禁止建议买入",
        "return_lcb_not_positive": "保守收益下界不为正，禁止建议买入",
        "exit_probability_below_floor": "T+1可退出概率不足，禁止建议买入",
        "fill_probability_below_floor": "竞价可成交概率不足，放弃",
        "profit_probability_below_floor": "盈利概率不足，放弃",
        "conservative_edge_below_floor": "扣除尾部风险与不可退出风险后没有正优势",
        "insufficient_independent_history": "独立交易日样本不足，模型尚未达到可用条件",
    }.get(reason, reason or "没有满足风险约束的安全竞价价格")


def _decision_lookup(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "ts_code" not in frame.columns:
        return {}
    return {str(row["ts_code"]): row for _, row in frame.drop_duplicates("ts_code", keep="first").iterrows()}


def _pending_candidates(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    eligible, _ = filter_standard_limit_universe(frame, code_col="ts_code", name_col="name")
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(eligible.head(limit).iterrows(), start=1):
        rows.append(
            {
                "rank": index,
                "action": "PENDING",
                "ts_code": _text(row.get("ts_code")),
                "name": _text(row.get("name")),
                "industry": _industry(row),
                "target_weight": 0.0,
                "decision_p_fill": _number(row.get("p_fill_pred")),
                "decision_e_ret": _number(row.get("e_ret_pred")),
                "decision_ev": _number(row.get("ev_pred")),
                "decision_cost": _number(row.get("cost_est")),
                "decision_risk_penalty": _number(row.get("risk_penalty")),
                "rejection_reason": "等待 Decision 竞价指导模型完成严格样本外定价",
            }
        )
    return rows


def _merge_auction_candidates(
    prediction: pd.DataFrame,
    decision: pd.DataFrame,
    *,
    promoted: bool,
    risk_budget: float,
) -> list[dict[str, Any]]:
    lookup = _decision_lookup(decision)
    prediction = annotate_standard_limit_universe(prediction, code_col="ts_code", name_col="name")
    selected_mask = pd.to_numeric(prediction.get("selected"), errors="coerce").fillna(0).eq(1)
    eligible_mask = pd.to_numeric(prediction["decision_universe_eligible"], errors="coerce").fillna(0).eq(1)
    selected_count = int((selected_mask & eligible_mask).sum())
    per_position_weight = min(0.12, max(0.0, risk_budget) / max(selected_count, 1)) if promoted else 0.0
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(prediction.iterrows(), start=1):
        code = _text(row.get("ts_code"))
        old = lookup.get(code, pd.Series(dtype=object))
        universe_eligible = _integer(row.get("decision_universe_eligible")) == 1
        selected = _integer(row.get("selected")) == 1 and universe_eligible
        action = "BUY" if promoted and selected else "SHADOW_ONLY" if selected else _text(row.get("action")) or "REJECT"
        reason = ""
        if not universe_eligible:
            action = "REJECT"
            reason = _text(row.get("decision_universe_reason")) or "涨跌幅机制不符合不超过10%的交易范围"
        elif not promoted and selected:
            reason = "严格样本外晋级门槛未全部通过，禁止正式买入"
        elif action != "BUY":
            reason = _rejection_reason(row.get("model_reason"))
        rows.append(
            {
                "rank": index,
                "action": action,
                "ts_code": code,
                "name": _text(row.get("name")) or _text(old.get("name")),
                "industry": _text(row.get("industry")) or _industry(old),
                "stage_transition": _text(row.get("stage_transition")) or _text(row.get("stage")),
                "stage_focus": _integer(row.get("stage_focus")),
                "target_weight": per_position_weight if action == "BUY" else 0.0,
                "mechanism_limit_pct": _number(row.get("decision_limit_pct")),
                "recommended_max_price": _number(row.get("recommended_max_price")),
                "max_auction_change_pct": _number(row.get("max_auction_change_pct")),
                "take_profit_pct": _number(row.get("take_profit_pct")),
                "stop_loss_pct": _number(row.get("stop_loss_pct")),
                "take_profit_price": _number(row.get("take_profit_price")),
                "stop_loss_price": _number(row.get("stop_loss_price")),
                "latest_exit_time": _text(row.get("latest_exit_time")),
                "exit_policy_version": _text(row.get("exit_policy_version")),
                "predicted_fill_probability": _number(row.get("predicted_fill_probability")),
                "predicted_exit_probability": _number(row.get("predicted_exit_probability")),
                "predicted_profit_probability": _number(row.get("predicted_profit_probability")),
                "predicted_big_loss_probability": _number(row.get("predicted_big_loss_probability")),
                "predicted_continuation_limit_up_probability": _number(
                    row.get("predicted_continuation_limit_up_probability")
                ),
                "predicted_net_return": _number(row.get("predicted_net_return")),
                "predicted_return_lcb": _number(row.get("predicted_return_lcb")),
                "predicted_return_ucb": _number(row.get("predicted_return_ucb")),
                "conservative_ev": _number(row.get("conservative_ev")),
                "decision_p_fill": _number(old.get("p_fill_pred")),
                "decision_e_ret": _number(old.get("e_ret_pred")),
                "decision_ev": _number(old.get("ev_pred")),
                "decision_cost": _number(old.get("cost_est")),
                "decision_risk_penalty": _number(old.get("risk_penalty")),
                "entry_rule": _text(row.get("entry_rule")),
                "exit_rule": _text(row.get("exit_rule")),
                "order_type": _text(row.get("order_type")),
                "market_order_allowed": _integer(row.get("market_order_allowed")),
                "risk_gate_pass": _integer(row.get("risk_gate_pass")),
                "rejection_reason": reason,
            }
        )
    return rows


def _stage_watchlist(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    focused = [
        dict(row)
        for row in rows
        if _text(row.get("stage_transition")) in {"2→3", "3→4"}
    ]

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        continuation = _number(row.get("predicted_continuation_limit_up_probability"))
        big_loss = _number(row.get("predicted_big_loss_probability"))
        conservative = _number(row.get("conservative_ev"))
        return (
            -(continuation if continuation is not None else -1.0),
            big_loss if big_loss is not None else 2.0,
            -(conservative if conservative is not None else -1.0),
            _integer(row.get("rank"), 9999),
        )

    focused.sort(key=sort_key)
    for rank, row in enumerate(focused[:limit], start=1):
        row["stage_watch_rank"] = rank
        row["watch_label"] = "正式买入" if row.get("action") == "BUY" else "仅观察"
    return focused[:limit]


def build_action_plan(root: Path, report_date: str = "") -> dict[str, Any]:
    root = root.resolve()
    dates = _report_dates(root)
    chosen_date = _date(report_date) or (dates[0] if dates else "")
    if not chosen_date:
        raise RuntimeError("No decision_report_YYYYMMDD.md exists")

    evaluation = _read_json(root / "outputs" / "decision" / f"eval_{chosen_date}.json")
    signal_date = _date(evaluation.get("signal_date"))
    exec_date = _date(evaluation.get("exec_date")) or chosen_date
    exit_date = _date(evaluation.get("exit_date"))
    risk_budget = _number(evaluation.get("risk_budget")) or 0.0
    candidates = _read_csv(_candidate_path(root, evaluation, signal_date))

    prediction = _read_csv(root / "outputs" / "auction_v3" / "predictions" / "pred_latest.csv")
    backtest = _read_json(root / "outputs" / "auction_v3" / "metrics" / "backtest_latest.json")
    model_meta = _read_json(root / "outputs" / "auction_v3" / "models" / "model_meta_latest.json")
    pred_signal = _date(prediction.get("signal_date", pd.Series([""])).iloc[0]) if not prediction.empty else ""
    pred_buy = _date(prediction.get("expected_buy_date", pd.Series([""])).iloc[0]) if not prediction.empty else ""
    pred_exit = _date(prediction.get("expected_exit_date", pd.Series([""])).iloc[0]) if not prediction.empty else ""
    prediction_matches = bool(signal_date and pred_signal == signal_date and pred_buy == exec_date and pred_exit == exit_date)
    pred_version = _text(prediction.get("model_version", pd.Series([""])).iloc[0]) if not prediction.empty else ""
    backtest_version = _text(backtest.get("model_version"))
    meta_version = _text(model_meta.get("model_version"))
    artifact_versions_match = bool(pred_version and pred_version == backtest_version == meta_version)
    prediction_ready = _integer(prediction.get("model_ready", pd.Series([0])).iloc[0]) == 1 if not prediction.empty else False
    prediction_promoted = _integer(prediction.get("model_promoted", pd.Series([0])).iloc[0]) == 1 if not prediction.empty else False
    promoted = bool(
        backtest.get("promoted") is True
        and model_meta.get("promoted") is True
        and model_meta.get("ready") is True
        and prediction_ready
        and prediction_promoted
        and prediction_matches
        and artifact_versions_match
    )

    if evaluation.get("stop_trading") is True:
        status_code = "NO_TRADE_GUARDRAIL"
        status_label = "停手：风控阻止交易"
        action_rows = _pending_candidates(candidates)
        for row in action_rows:
            row["action"] = "REJECT"
            row["rejection_reason"] = _text(evaluation.get("reason")) or "Decision guardrail stopped trading"
    elif not prediction_matches:
        status_code = "PENDING_AUCTION_MODEL"
        status_label = "等待竞价执行模型完成"
        action_rows = _pending_candidates(candidates)
    else:
        action_rows = _merge_auction_candidates(prediction, candidates, promoted=promoted, risk_budget=risk_budget)
        formal_count = sum(row["action"] == "BUY" for row in action_rows)
        if not promoted:
            status_code = "NO_TRADE_MODEL_NOT_PROMOTED"
            status_label = "不交易：样本外晋级未通过"
        elif formal_count == 0:
            status_code = "NO_TRADE_NO_POSITIVE_EDGE"
            status_label = "不交易：没有通过全部约束的正收益机会"
        else:
            status_code = "ACTIONABLE_BUY"
            status_label = "人工参考：按竞价上限自行挂单"

    formal_count = sum(row["action"] == "BUY" for row in action_rows)
    shadow_count = sum(row["action"] == "SHADOW_ONLY" for row in action_rows)
    stage_watchlist = _stage_watchlist(action_rows)
    plan = {
        "schema_version": "decision_action_plan_v4_full_limit_pool_stage_watch",
        "generated_at_utc": _utc_now(),
        "report_date": chosen_date,
        "report_file": f"decision_report_{chosen_date}.md",
        "signal_date": signal_date,
        "exec_date": exec_date,
        "exit_date": exit_date,
        "status_code": status_code,
        "status_label": status_label,
        "formal_buy_count": formal_count,
        "shadow_count": shadow_count,
        "stage_watch_count": len(stage_watchlist),
        "risk_budget": risk_budget,
        "guidance_only": True,
        "broker_connected": False,
        "order_execution": "manual_only",
        "model": {
            "version": _text(model_meta.get("model_version")) or _text(backtest.get("model_version")),
            "ready": model_meta.get("ready") is True,
            "promoted": promoted,
            "prediction_matches_report": prediction_matches,
            "artifact_versions_match": artifact_versions_match,
            "promotion_failures": list(backtest.get("promotion_failures", []) or []),
            "return_model": _text((model_meta.get("return_selection", {}) or {}).get("selected")),
            "profit_model": _text(
                ((model_meta.get("classifier_selection", {}) or {}).get("profit", {}) or {}).get("selected")
            ),
            "big_loss_model": _text(
                ((model_meta.get("classifier_selection", {}) or {}).get("big_loss", {}) or {}).get("selected")
            ),
            "continuation_model": _text(
                ((model_meta.get("classifier_selection", {}) or {}).get("continuation_limit_up", {}) or {}).get("selected")
            ),
        },
        "backtest": {
            key: backtest.get(key)
            for key in (
                "history_dates",
                "oos_dates",
                "signals",
                "signal_dates",
                "signal_date_ratio",
                "max_no_signal_streak",
                "filled_trades",
                "mean_trade_net_return",
                "win_rate",
                "realized_big_loss_rate",
                "tail_10pct_mean_return",
                "worst_trade_net_return",
                "stage_focus_signals",
                "stage_focus_filled_trades",
                "stage_focus_continuation_hit_rate",
                "cumulative_return",
                "max_drawdown",
                "sharpe",
                "stress_2x_cost_mean_daily_return",
                "bootstrap_probability_mean_positive",
                "exit_on_time_rate",
            )
        },
        "universe_eligibility": model_meta.get("universe_eligibility") or evaluation.get("universe_eligibility") or {},
        "execution_contract": {
            "objective": "D日冻结信号，指导人工在T日9:25前参与开盘集合竞价，T+1按预声明择机规则卖出，最大化扣除费用和不可成交风险后的样本外收益",
            "calendar": "严格使用上交所A股交易日历，禁止工作日或raw目录推断",
            "candidate_pool": "以D日limit_list_d确认涨停清单为权威全集，不扩展到全市场、不受旧Top50截断；2进3、3进4作为重点晋级目标",
            "eligible_universe": "D日已涨停且价格涨跌幅限制机制不超过10%的A股",
            "entry": "系统不下单；T日9:25前仅允许人工限价挂单，禁止无上限市价单，高于冻结上限或未成交均放弃",
            "exit": "T+1按实际成交价计算3%止盈、2.5%止损，首次触发即人工退出；均未触发则14:50退出；一字跌停顺延",
            "return_target": "T日开盘集合竞价代理成交价到T+1止盈/止损/14:50时间退出价的保守可执行收益",
            "validation": "公开行情只生成模拟验证；人工实际成交需手工回填，二者分开累计",
            "risk_veto": "预测大跌概率超过15%、保守收益下界不为正、竞价成交或T+1退出概率不足，任一项触发即否决",
            "guidance_only": True,
            "broker_connected": False,
            "no_trade_is_valid": True,
            "profit_not_guaranteed": True,
        },
        "stage_watchlist": stage_watchlist,
        "candidates": action_rows,
    }
    return _json_safe(plan)


def build_report_index(root: Path, latest_report_date: str = "") -> dict[str, Any]:
    dates = _report_dates(root)
    latest = _date(latest_report_date) or (dates[0] if dates else "")
    reports = [
        {
            "report_date": date,
            "report_file": f"decision_report_{date}.md",
            "report_url": f"outputs/decision/decision_report_{date}.md",
            "eval_url": f"outputs/decision/eval_{date}.json",
            "action_url": f"outputs/decision/action_plan_{date}.json",
        }
        for date in dates
    ]
    return {
        "schema_version": "decision_report_index_v1",
        "generated_at_utc": _utc_now(),
        "latest_report_date": latest,
        "latest_report_file": f"decision_report_{latest}.md" if latest else "",
        "latest_action_url": "outputs/decision/action_plan_latest.json" if latest else "",
        "reports": reports,
    }


def publish_action_plan(root: Path, report_date: str = "") -> tuple[Path, Path, Path, dict[str, Any]]:
    root = root.resolve()
    plan = build_action_plan(root, report_date)
    report_date = str(plan["report_date"])
    output = root / "outputs" / "decision"
    dated_path = output / f"action_plan_{report_date}.json"
    latest_path = output / "action_plan_latest.json"
    index_path = output / "report_index.json"
    _write_json(dated_path, plan)
    _write_json(latest_path, plan)
    _write_json(index_path, build_report_index(root, report_date))
    return dated_path, latest_path, index_path, plan


__all__ = ["build_action_plan", "build_report_index", "publish_action_plan"]
