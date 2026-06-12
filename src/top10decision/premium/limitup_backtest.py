# -*- coding: utf-8 -*-
"""
Premium 子系统 — LimitUp Backtest（V3.6：涨停接力离线回测）

用途：
    回测当前 Premium V3.5 / V3.6 的“涨停接力评分”是否真的有预测力。

输入：
    limitup_labels.py 生成的历史标签文件，且最好包含：
        涨停接力评分 / limitup_continuation_score
        T日建议买入方式 / t_buy_method
        t_limitup_hit
        t_touch_limitup
        t1_up_hit
        t1_high_profit_hit
        t1_close_ret
        t1_high_ret
        label_valid

固定输出：
    Top5 T日涨停率
    Top10 T日涨停率
    Top5 T日触板率
    Top10 T日触板率
    Top5 T+1上涨率
    Top10 T+1上涨率
    Top5 平均T+1最高收益
    Top10 平均T+1最高收益
    最大单票亏损
    市价竞价建议命中率
    限价竞价建议命中率

设计原则：
    - 只做离线校验，不修改 predict.py / report_md.py / train.py / workflow。
    - 按 D 日分组做 TopN，避免全样本混排造成日期偏差。
    - 如果 Top5 明显优于 Top10 / 全样本，才说明当前评分有预测力。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DATE_COLS = ["d_trade_date", "trade_date", "date", "交易日期", "日期"]
SCORE_COLS = ["涨停接力评分", "limitup_continuation_score", "limitup_model_score"]
BUY_METHOD_COLS = ["T日建议买入方式", "t_buy_method", "buy_method", "建议买入方式"]


def _first_existing(df: pd.DataFrame, names: Sequence[str], required: bool = True) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"缺少必要字段，候选字段={list(names)}，当前字段={list(df.columns)}")
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"暂不支持的输入格式: {suffix}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"暂不支持的输出格式: {suffix}")


def _norm_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    dt = pd.to_datetime(raw, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y%m%d", errors="coerce")
    return dt.dt.strftime("%Y%m%d")


def _mean_pct(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def _min_pct(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.min())


def _hit_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def _topn_by_day(df: pd.DataFrame, date_col: str, score_col: str, n: int) -> pd.DataFrame:
    x = df.copy()
    x["_score_num"] = pd.to_numeric(x[score_col], errors="coerce")
    x = x[x["_score_num"].notna()].copy()
    if x.empty:
        return x
    return (
        x.sort_values([date_col, "_score_num"], ascending=[True, False])
         .groupby(date_col, group_keys=False)
         .head(n)
         .drop(columns=["_score_num"], errors="ignore")
    )


def _buy_method_hit_rate(df: pd.DataFrame, method_col: Optional[str], keyword: str) -> float:
    if method_col is None or method_col not in df.columns or "t1_high_profit_hit" not in df.columns:
        return np.nan
    sub = df[df[method_col].astype(str).str.contains(keyword, na=False)].copy()
    if sub.empty:
        return np.nan
    return _hit_rate(sub, "t1_high_profit_hit")


def backtest_limitup_score(
    df: pd.DataFrame,
    top_ns: Sequence[int] = (5, 10),
    score_col: Optional[str] = None,
    date_col: Optional[str] = None,
    min_valid_per_day: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
        summary_df：聚合回测指标
        ranked_df：带 daily_rank 的逐票明细
    """
    data = df.copy()

    date_col = date_col or _first_existing(data, DATE_COLS)
    score_col = score_col or _first_existing(data, SCORE_COLS)
    buy_method_col = _first_existing(data, BUY_METHOD_COLS, required=False)

    if "label_valid" in data.columns:
        data = data[pd.to_numeric(data["label_valid"], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif "label_matured" in data.columns:
        data = data[pd.to_numeric(data["label_matured"], errors="coerce").fillna(0).astype(int) == 1].copy()

    if data.empty:
        raise ValueError("没有可回测的有效标签样本：请先检查 label_valid / label_matured 和行情标签生成结果")

    data[date_col] = _norm_date(data[date_col])
    data["_score_num"] = pd.to_numeric(data[score_col], errors="coerce")
    data = data[data["_score_num"].notna()].copy()
    if data.empty:
        raise ValueError(f"评分字段 {score_col} 全为空，无法回测")

    valid_counts = data.groupby(date_col).size()
    keep_dates = valid_counts[valid_counts >= int(min_valid_per_day)].index
    data = data[data[date_col].isin(keep_dates)].copy()

    data = data.sort_values([date_col, "_score_num"], ascending=[True, False])
    data["daily_rank"] = data.groupby(date_col).cumcount() + 1

    rows = []
    all_row = {
        "bucket": "ALL",
        "days": int(data[date_col].nunique()),
        "samples": int(len(data)),
        "T日涨停率": _hit_rate(data, "t_limitup_hit"),
        "T日触板率": _hit_rate(data, "t_touch_limitup"),
        "T+1上涨率": _hit_rate(data, "t1_up_hit"),
        "T+1高点兑现率": _hit_rate(data, "t1_high_profit_hit"),
        "平均T+1收盘收益": _mean_pct(data["t1_close_ret"]) if "t1_close_ret" in data.columns else np.nan,
        "平均T+1最高收益": _mean_pct(data["t1_high_ret"]) if "t1_high_ret" in data.columns else np.nan,
        "最大单票亏损": _min_pct(data["t1_close_ret"]) if "t1_close_ret" in data.columns else np.nan,
        "市价竞价建议命中率": _buy_method_hit_rate(data, buy_method_col, "市价"),
        "限价竞价建议命中率": _buy_method_hit_rate(data, buy_method_col, "限价"),
        "score_col": score_col,
        "date_col": date_col,
    }
    rows.append(all_row)

    for n in top_ns:
        sub = data[data["daily_rank"] <= int(n)].copy()
        rows.append({
            "bucket": f"Top{n}",
            "days": int(sub[date_col].nunique()),
            "samples": int(len(sub)),
            "T日涨停率": _hit_rate(sub, "t_limitup_hit"),
            "T日触板率": _hit_rate(sub, "t_touch_limitup"),
            "T+1上涨率": _hit_rate(sub, "t1_up_hit"),
            "T+1高点兑现率": _hit_rate(sub, "t1_high_profit_hit"),
            "平均T+1收盘收益": _mean_pct(sub["t1_close_ret"]) if "t1_close_ret" in sub.columns else np.nan,
            "平均T+1最高收益": _mean_pct(sub["t1_high_ret"]) if "t1_high_ret" in sub.columns else np.nan,
            "最大单票亏损": _min_pct(sub["t1_close_ret"]) if "t1_close_ret" in sub.columns else np.nan,
            "市价竞价建议命中率": _buy_method_hit_rate(sub, buy_method_col, "市价"),
            "限价竞价建议命中率": _buy_method_hit_rate(sub, buy_method_col, "限价"),
            "score_col": score_col,
            "date_col": date_col,
        })

    summary = pd.DataFrame(rows)

    # 增加 TopN 相对全样本提升，方便判断有没有预测力
    base = summary[summary["bucket"] == "ALL"].iloc[0]
    for col in ["T日涨停率", "T日触板率", "T+1上涨率", "T+1高点兑现率", "平均T+1最高收益"]:
        summary[f"{col}_相对ALL提升"] = summary[col] - base[col]

    ranked = data.drop(columns=["_score_num"], errors="ignore")
    return summary, ranked


def backtest_from_files(
    input_path: Path,
    output_path: Path,
    detail_output_path: Optional[Path] = None,
    score_col: Optional[str] = None,
    date_col: Optional[str] = None,
    top_ns: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    df = _read_table(input_path)
    summary, detail = backtest_limitup_score(df, top_ns=top_ns, score_col=score_col, date_col=date_col)
    _write_table(summary, output_path)
    if detail_output_path:
        _write_table(detail, detail_output_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Premium 涨停接力评分 TopN 离线回测")
    p.add_argument("--input", required=True, help="limitup_labels.py 输出的标签文件")
    p.add_argument("--output", required=True, help="回测汇总输出 csv/xlsx/parquet")
    p.add_argument("--detail-output", default=None, help="可选：逐票排名明细输出")
    p.add_argument("--score-col", default=None, help="评分字段；默认自动识别 涨停接力评分 / limitup_continuation_score / limitup_model_score")
    p.add_argument("--date-col", default=None, help="日期字段；默认自动识别 d_trade_date / trade_date")
    p.add_argument("--top-n", default="5,10", help="逗号分隔 TopN，默认 5,10")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    top_ns = [int(x.strip()) for x in args.top_n.split(",") if x.strip()]
    summary = backtest_from_files(
        input_path=Path(args.input),
        output_path=Path(args.output),
        detail_output_path=Path(args.detail_output) if args.detail_output else None,
        score_col=args.score_col,
        date_col=args.date_col,
        top_ns=top_ns,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
