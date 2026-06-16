# -*- coding: utf-8 -*-
"""
Premium 子系统 — LimitUp Labels（V3.6：A股交易日历严格对齐版）

用途：
    为 Premium V3.5 / 涨停接力实盘评分排序阶段构建历史真实标签。

核心修正：
    1. 严格按“全市场 A 股交易日历”对齐 D → T → T+1。
    2. 不再简单按单票 shift(-1/-2) 推断下一条记录，避免停牌/缺失行情导致日期错位。
    3. 支持 calendar 文件；未提供时，使用行情数据中的全市场 trade_date 唯一值作为交易日历。
    4. 支持 --as-of latest / YYYYMMDD，把标签日期跑到当前行情可验证的最新成熟样本。

真实时间轴：
    D   = 分析基准日，使用 D 日收盘后可见信息
    T   = D 后第 1 个 A 股交易日，集合竞价/开盘买入验证日
    T+1 = D 后第 2 个 A 股交易日，卖出/验证日

核心输出标签：
    t_limitup_hit        # T 日是否收盘涨停
    t_touch_limitup      # T 日是否盘中触及涨停
    t1_up_hit            # T+1 收盘是否上涨
    t1_high_profit_hit   # T+1 盘中是否给过可兑现收益
    t1_close_ret         # T+1 收盘收益
    t1_high_ret          # T+1 最高价收益

设计原则：
    - 不依赖 Decision 主线。
    - 不修改 train.py / run_premium.yml。
    - 本文件只做标签构建，不做预测、不做排序。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CODE_COLS = ["ts_code", "code", "symbol", "证券代码", "股票代码"]
DATE_COLS = ["trade_date", "date", "交易日期", "日期"]
OPEN_COLS = ["open", "开盘价"]
HIGH_COLS = ["high", "最高价"]
LOW_COLS = ["low", "最低价"]
CLOSE_COLS = ["close", "收盘价", "D日收盘价"]
PRE_CLOSE_COLS = ["pre_close", "preclose", "prev_close", "昨收", "前收盘"]
LIMIT_UP_COLS = ["limit_up", "up_limit", "涨停价", "涨停板价"]
PCT_CHG_COLS = ["pct_chg", "pct_change", "涨跌幅"]
AMOUNT_COLS = ["amount", "成交额"]
VOLUME_COLS = ["vol", "volume", "成交量"]


def _first_existing(df: pd.DataFrame, names: Sequence[str], required: bool = True) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"缺少必要字段，候选字段={list(names)}，当前字段={list(df.columns)}")
    return None


def _norm_ts_code(x: object) -> str:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if "." in s:
        left, right = s.split(".", 1)
        return f"{left.zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
        if digits.startswith(("60", "68", "90")):
            return f"{digits}.SH"
        if digits.startswith(("00", "30", "20")):
            return f"{digits}.SZ"
        if digits.startswith(("43", "83", "87", "88", "92")):
            return f"{digits}.BJ"
        return digits
    return s


def _norm_date_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    dt = pd.to_datetime(raw, errors="coerce")
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y%m%d", errors="coerce")
    return dt.dt.strftime("%Y%m%d")


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


def _calendar_next_map(calendar_dates: Sequence[str]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    dates = list(sorted(set(str(x) for x in calendar_dates if str(x) and str(x) != "NaT")))
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for i, d in enumerate(dates):
        t = dates[i + 1] if i + 1 < len(dates) else None
        t1 = dates[i + 2] if i + 2 < len(dates) else None
        out[d] = (t, t1)
    return out


def _limit_rate_for_code(ts_code: str) -> float:
    code = _norm_ts_code(ts_code)
    raw = code.split(".")[0]
    suffix = code.split(".")[-1] if "." in code else ""
    if suffix == "BJ" or raw.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if raw.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def build_limitup_labels(
    market_df: pd.DataFrame,
    sample_df: Optional[pd.DataFrame] = None,
    calendar_df: Optional[pd.DataFrame] = None,
    as_of: str = "latest",
    high_profit_threshold: float = 0.02,
    limit_tolerance: float = 0.0015,
) -> pd.DataFrame:
    """
    构建 D → T → T+1 标签。

    关键约束：
        - T / T+1 来自全市场 A 股交易日历。
        - 如果个股在 T 或 T+1 缺行情，不向后漂移，直接标记 label_valid=0。
        - 这样可以避免停牌、缺失行情造成的假标签。
    """
    m = market_df.copy()
    m_code = _first_existing(m, CODE_COLS)
    m_date = _first_existing(m, DATE_COLS)
    m_open = _first_existing(m, OPEN_COLS, required=False)
    m_high = _first_existing(m, HIGH_COLS)
    m_low = _first_existing(m, LOW_COLS, required=False)
    m_close = _first_existing(m, CLOSE_COLS)
    m_pre_close = _first_existing(m, PRE_CLOSE_COLS, required=False)
    m_limit_up = _first_existing(m, LIMIT_UP_COLS, required=False)
    m_pct_chg = _first_existing(m, PCT_CHG_COLS, required=False)
    m_amount = _first_existing(m, AMOUNT_COLS, required=False)
    m_volume = _first_existing(m, VOLUME_COLS, required=False)

    m["_ts_code_norm"] = m[m_code].map(_norm_ts_code)
    m["_trade_date_norm"] = _norm_date_series(m[m_date])
    for c in [m_open, m_high, m_low, m_close, m_pre_close, m_limit_up, m_pct_chg, m_amount, m_volume]:
        if c is not None:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    if calendar_df is not None:
        cdate = _first_existing(calendar_df, DATE_COLS)
        calendar_dates = _norm_date_series(calendar_df[cdate]).dropna().unique().tolist()
        calendar_source = "calendar_df"
    else:
        calendar_dates = sorted(m["_trade_date_norm"].dropna().unique().tolist())
        calendar_source = "market_trade_date_unique"

    next_map = _calendar_next_map(calendar_dates)
    as_of_date = max(calendar_dates) if as_of == "latest" else _norm_date_series(pd.Series([as_of])).iloc[0]

    if sample_df is None:
        s = m.copy()
    else:
        s = sample_df.copy()
        s_code_raw = _first_existing(s, CODE_COLS)
        s_date_raw = _first_existing(s, DATE_COLS)
        s["_ts_code_norm"] = s[s_code_raw].map(_norm_ts_code)
        s["_trade_date_norm"] = _norm_date_series(s[s_date_raw])

    s["d_trade_date"] = s["_trade_date_norm"]
    s["t_trade_date"] = s["d_trade_date"].map(lambda d: next_map.get(d, (None, None))[0])
    s["t1_trade_date"] = s["d_trade_date"].map(lambda d: next_map.get(d, (None, None))[1])
    s["label_matured"] = (
        s["t_trade_date"].notna()
        & s["t1_trade_date"].notna()
        & (s["t1_trade_date"].astype(str) <= str(as_of_date))
    ).astype(int)

    base_cols = [
        "_ts_code_norm",
        "_trade_date_norm",
        m_open,
        m_high,
        m_low,
        m_close,
        m_pre_close,
        m_limit_up,
        m_pct_chg,
        m_amount,
        m_volume,
    ]
    base_cols = [c for c in base_cols if c is not None]
    q = m[base_cols].copy()

    right_d = q.copy()
    rename_d = {
        "_ts_code_norm": "_join_code",
        "_trade_date_norm": "_join_date",
        m_high: "d_high",
        m_close: "d_close",
    }
    if m_open is not None:
        rename_d[m_open] = "d_open"
    if m_low is not None:
        rename_d[m_low] = "d_low"
    if m_pre_close is not None:
        rename_d[m_pre_close] = "d_pre_close"
    if m_pct_chg is not None:
        rename_d[m_pct_chg] = "d_pct_chg"
    if m_amount is not None:
        rename_d[m_amount] = "d_amount"
    if m_volume is not None:
        rename_d[m_volume] = "d_volume"
    right_d = right_d.rename(columns=rename_d)

    out = s.copy()
    out["_join_code"] = out["_ts_code_norm"]
    out["_join_date"] = out["d_trade_date"]
    out = out.merge(right_d, on=["_join_code", "_join_date"], how="left").drop(columns=["_join_code", "_join_date"])

    right_t = q.copy()
    rename_t = {
        "_ts_code_norm": "_join_code",
        "_trade_date_norm": "_join_date",
        m_high: "t_high",
        m_close: "t_close",
    }
    if m_open is not None:
        rename_t[m_open] = "t_open"
    if m_low is not None:
        rename_t[m_low] = "t_low"
    if m_pre_close is not None:
        rename_t[m_pre_close] = "t_pre_close"
    if m_limit_up is not None:
        rename_t[m_limit_up] = "t_limit_up"
    right_t = right_t.rename(columns=rename_t)

    out["_join_code"] = out["_ts_code_norm"]
    out["_join_date"] = out["t_trade_date"]
    out = out.merge(right_t, on=["_join_code", "_join_date"], how="left").drop(columns=["_join_code", "_join_date"])

    right_t1 = q.copy()
    rename_t1 = {
        "_ts_code_norm": "_join_code",
        "_trade_date_norm": "_join_date",
        m_high: "t1_high",
        m_close: "t1_close",
    }
    if m_open is not None:
        rename_t1[m_open] = "t1_open"
    if m_low is not None:
        rename_t1[m_low] = "t1_low"
    if m_pre_close is not None:
        rename_t1[m_pre_close] = "t1_pre_close"
    if m_limit_up is not None:
        rename_t1[m_limit_up] = "t1_limit_up"
    right_t1 = right_t1.rename(columns=rename_t1)

    out["_join_code"] = out["_ts_code_norm"]
    out["_join_date"] = out["t1_trade_date"]
    out = out.merge(right_t1, on=["_join_code", "_join_date"], how="left").drop(columns=["_join_code", "_join_date"])

    if "d_close" not in out.columns:
        out["d_close"] = pd.to_numeric(out[m_close], errors="coerce") if m_close in out.columns else np.nan

    if "t_limit_up" in out.columns:
        out["_t_limit_price"] = pd.to_numeric(out["t_limit_up"], errors="coerce")
    else:
        pre = pd.to_numeric(out.get("t_pre_close", np.nan), errors="coerce")
        rates = out["_ts_code_norm"].map(_limit_rate_for_code).astype(float)
        out["_t_limit_price"] = (pre * (1.0 + rates)).round(2)

    out["t_limit_price"] = out["_t_limit_price"]
    t_open = pd.to_numeric(out.get("t_open", np.nan), errors="coerce")
    t_high = pd.to_numeric(out.get("t_high", np.nan), errors="coerce")
    t_low = pd.to_numeric(out.get("t_low", np.nan), errors="coerce")
    t_close = pd.to_numeric(out.get("t_close", np.nan), errors="coerce")
    t_limit = pd.to_numeric(out["_t_limit_price"], errors="coerce")
    t1_open = pd.to_numeric(out.get("t1_open", np.nan), errors="coerce")
    t1_low = pd.to_numeric(out.get("t1_low", np.nan), errors="coerce")
    t1_close = pd.to_numeric(out.get("t1_close", np.nan), errors="coerce")
    t1_high = pd.to_numeric(out.get("t1_high", np.nan), errors="coerce")
    d_close = pd.to_numeric(out["d_close"], errors="coerce")
    buy_base = t_close.where(t_close.notna(), d_close)

    out["t_touch_limitup"] = ((t_high >= t_limit * (1.0 - limit_tolerance)) & t_limit.notna()).astype(float)
    out["t_open_up_hit"] = (t_open > d_close).astype(float)
    out["t_up_hit"] = (t_close > d_close).astype(float)
    out["t_limitup_hit"] = ((t_close >= t_limit * (1.0 - limit_tolerance)) & t_limit.notna()).astype(float)
    out["t_open_ret"] = np.where(d_close > 0, t_open / d_close - 1.0, np.nan)
    out["t_intraday_ret"] = np.where(d_close > 0, t_high / d_close - 1.0, np.nan)
    out["t_close_ret"] = np.where(d_close > 0, t_close / d_close - 1.0, np.nan)
    out["t_high_profit_hit"] = (out["t_intraday_ret"] >= float(high_profit_threshold)).astype(float)
    out["t_low_ret"] = np.where(d_close > 0, t_low / d_close - 1.0, np.nan)
    out["t1_open_ret"] = np.where(buy_base > 0, t1_open / buy_base - 1.0, np.nan)
    out["t1_close_ret"] = np.where(buy_base > 0, t1_close / buy_base - 1.0, np.nan)
    out["t1_high_ret"] = np.where(buy_base > 0, t1_high / buy_base - 1.0, np.nan)
    out["t1_low_ret"] = np.where(buy_base > 0, t1_low / buy_base - 1.0, np.nan)
    out["t1_up_hit"] = (out["t1_close_ret"] > 0).astype(float)
    out["t1_high_profit_hit"] = (out["t1_high_ret"] >= float(high_profit_threshold)).astype(float)
    out["t1_accept_hit"] = ((pd.to_numeric(out["t1_high_ret"], errors="coerce") >= 0.015) & (pd.to_numeric(out["t1_close_ret"], errors="coerce") >= -0.015)).astype(float)
    out["t1_fail_hit"] = ((pd.to_numeric(out["t1_high_ret"], errors="coerce") < 0.008) | (pd.to_numeric(out["t1_close_ret"], errors="coerce") <= -0.025)).astype(float)
    out["t1_big_drawdown_hit"] = (pd.to_numeric(out["t1_low_ret"], errors="coerce") <= -0.04).astype(float)
    out["t1_limitdown_risk_hit"] = (pd.to_numeric(out["t1_low_ret"], errors="coerce") <= -0.08).astype(float)

    valid_t = t_close.notna() & t_high.notna()
    valid_t1 = t1_close.notna() & t1_high.notna()
    valid_all = out["label_matured"].eq(1) & valid_t & valid_t1

    label_cols = [
        "t_open_up_hit",
        "t_up_hit",
        "t_high_profit_hit",
        "t_limitup_hit",
        "t_touch_limitup",
        "t_open_ret",
        "t_intraday_ret",
        "t_close_ret",
        "t_low_ret",
        "t1_open_ret",
        "t1_up_hit",
        "t1_high_profit_hit",
        "t1_accept_hit",
        "t1_fail_hit",
        "t1_big_drawdown_hit",
        "t1_limitdown_risk_hit",
        "t1_close_ret",
        "t1_high_ret",
        "t1_low_ret",
    ]
    for c in label_cols:
        out.loc[~valid_all, c] = np.nan

    out["label_valid"] = valid_all.astype(int)
    out["calendar_status"] = np.select(
        [
            out["t_trade_date"].isna(),
            out["t1_trade_date"].isna(),
            out["label_matured"].ne(1),
            valid_all,
        ],
        ["missing_t_trade_date", "missing_t1_trade_date", "not_matured", "ok"],
        default="missing_price",
    )
    out["calendar_reason"] = np.where(
        valid_all,
        "strict_a_share_calendar_ok",
        "strict_a_share_calendar_or_price_not_ready",
    )
    out["calendar_source"] = calendar_source
    out["label_as_of"] = as_of_date
    return out


def build_limitup_labels_from_files(
    market_path: Path,
    output_path: Path,
    sample_path: Optional[Path] = None,
    calendar_path: Optional[Path] = None,
    as_of: str = "latest",
    high_profit_threshold: float = 0.02,
) -> pd.DataFrame:
    market_df = _read_table(market_path)
    sample_df = _read_table(sample_path) if sample_path else None
    calendar_df = _read_table(calendar_path) if calendar_path else None
    out = build_limitup_labels(
        market_df=market_df,
        sample_df=sample_df,
        calendar_df=calendar_df,
        as_of=as_of,
        high_profit_threshold=high_profit_threshold,
    )
    _write_table(out, output_path)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 Premium 涨停接力 D→T→T+1 历史真实标签")
    p.add_argument("--market", required=True, help="全市场历史行情 csv/xlsx/parquet")
    p.add_argument("--output", required=True, help="标签输出文件 csv/xlsx/parquet")
    p.add_argument("--sample", default=None, help="可选：待打标签样本文件；不填则 market 每行都视为 D 日样本")
    p.add_argument("--calendar", default=None, help="可选：A股交易日历文件；不填则用 market.trade_date 唯一值")
    p.add_argument("--as-of", default="latest", help="latest 或 YYYYMMDD；只保留 T+1 已成熟样本")
    p.add_argument("--high-profit-threshold", type=float, default=0.02, help="T+1 盘中可兑现收益阈值，默认 2%")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = build_limitup_labels_from_files(
        market_path=Path(args.market),
        output_path=Path(args.output),
        sample_path=Path(args.sample) if args.sample else None,
        calendar_path=Path(args.calendar) if args.calendar else None,
        as_of=args.as_of,
        high_profit_threshold=args.high_profit_threshold,
    )
    total = len(out)
    valid = int(pd.to_numeric(out["label_valid"], errors="coerce").fillna(0).sum())
    if total:
        print(f"[limitup_labels] output={args.output}")
        print(f"[limitup_labels] rows={total}, valid_labels={valid}, valid_ratio={valid / total:.2%}")
    else:
        print("[limitup_labels] rows=0")


if __name__ == "__main__":
    main()
