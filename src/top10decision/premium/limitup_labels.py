# -*- coding: utf-8 -*-
"""
Premium 子系统 — LimitUp Labels（V3.5 离线真实标签构建）

用途：
    把 Premium V3.5 当前“涨停接力评分层”升级为可验证闭环前，
    先根据历史行情生成真实标签，用于后续 limitup_backtest.py 回测。

时间轴：
    D   = 分析基准日，使用 D 日收盘后可见信息
    T   = 下一交易日，集合竞价/开盘买入验证日
    T+1 = 买入后的下一个交易日，卖出/验证日

核心输出标签：
    t_limitup_hit        # T 日是否收盘涨停
    t_touch_limitup      # T 日是否盘中触及涨停
    t1_up_hit            # T+1 收盘是否上涨
    t1_high_profit_hit   # T+1 盘中是否给过可兑现收益
    t1_close_ret         # T+1 收盘收益
    t1_high_ret          # T+1 最高价收益

设计原则：
    1. 不依赖 Decision 主线。
    2. 不修改 predict.py / train.py / workflow。
    3. 对字段名做兼容，适配常见行情 CSV。
    4. 默认按 ts_code 分组、trade_date 升序，用 shift(-1/-2) 对齐 T / T+1。
    5. 若行情中已有 limit_up / up_limit 字段，优先使用真实涨停价；否则按代码市场推断涨跌停比例。

注意：
    本文件只负责“历史真实标签构建”，不负责预测、不负责排序、不负责生成报告。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LimitupLabelConfig:
    """涨停接力标签构建参数。"""

    code_col: str = "ts_code"
    date_col: str = "trade_date"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    pre_close_col: str = "pre_close"
    limit_up_col: Optional[str] = None
    buy_price_col: Optional[str] = None
    high_profit_threshold: float = 0.03
    limit_tolerance: float = 0.0005
    keep_unmatured: bool = False
    infer_market_limit: bool = True
    default_limit_pct: float = 0.10


_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "ts_code": ("ts_code", "code", "symbol", "证券代码", "股票代码", "代码"),
    "trade_date": ("trade_date", "date", "datetime", "交易日期", "日期"),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价"),
    "pre_close": ("pre_close", "preclose", "prev_close", "昨收", "昨收价", "前收盘"),
    "limit_up": ("limit_up", "up_limit", "涨停价", "涨停", "upper_limit"),
}


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_to_real:
            return lower_to_real[key]
    return None


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把常见中英文字段名统一为标准行情字段名。"""
    out = df.copy()
    rename: Dict[str, str] = {}
    for std_col, aliases in _ALIAS_MAP.items():
        found = _first_existing_col(out, aliases)
        if found is not None and found != std_col and std_col not in out.columns:
            rename[found] = std_col
    if rename:
        out = out.rename(columns=rename)
    return out


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_code_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def infer_limit_pct_by_code(code: object, default_limit_pct: float = 0.10) -> float:
    """
    按 A 股代码粗略推断涨跌停比例。

    说明：
        - 主板多数股票默认 10%。
        - 创业板 / 科创板多数股票 20%。
        - 北交所多数股票 30%。
        - ST 5% 需要额外名称或专门字段，本函数不强行猜 ST。
    """
    c = str(code).strip().upper()
    pure = c.split(".")[0]

    if pure.startswith(("300", "301", "688", "689")):
        return 0.20
    if pure.startswith(("8", "4", "920")):
        return 0.30
    return float(default_limit_pct)


def calc_limit_up_price(
    pre_close: pd.Series,
    code: Optional[pd.Series] = None,
    default_limit_pct: float = 0.10,
    infer_market_limit: bool = True,
) -> pd.Series:
    """根据昨收和代码推算涨停价，按 A 股价格习惯保留 2 位小数。"""
    pre = _to_numeric(pre_close)
    if infer_market_limit and code is not None:
        pct = code.map(lambda x: infer_limit_pct_by_code(x, default_limit_pct))
        pct = _to_numeric(pct).fillna(default_limit_pct)
    else:
        pct = pd.Series(default_limit_pct, index=pre.index, dtype="float64")
    return (pre * (1.0 + pct)).round(2)


def _ensure_required_columns(df: pd.DataFrame, cfg: LimitupLabelConfig) -> None:
    required = [cfg.code_col, cfg.date_col, cfg.open_col, cfg.high_col, cfg.close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行情字段: {missing}")


def build_limitup_labels(
    df: pd.DataFrame,
    config: Optional[LimitupLabelConfig] = None,
) -> pd.DataFrame:
    """
    根据历史行情构建 D→T→T+1 的真实标签。

    输入：
        df: 至少包含 ts_code / trade_date / open / high / close。
            最好包含 pre_close 或 limit_up。

    输出：
        每一行仍代表 D 日，但新增 T / T+1 对齐字段和真实标签。
    """
    cfg = config or LimitupLabelConfig()
    data = normalize_ohlcv_columns(df)
    _ensure_required_columns(data, cfg)

    data = data.copy()
    data[cfg.code_col] = _normalize_code_series(data[cfg.code_col])

    # 日期字段保留原始展示值，同时增加排序键，兼容 20260609 / 2026-06-09。
    data["_label_sort_date"] = pd.to_datetime(data[cfg.date_col].astype(str), errors="coerce")
    numeric_date_mask = data["_label_sort_date"].isna()
    if numeric_date_mask.any():
        data.loc[numeric_date_mask, "_label_sort_date"] = pd.to_datetime(
            data.loc[numeric_date_mask, cfg.date_col].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

    for col in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.pre_close_col]:
        if col in data.columns:
            data[col] = _to_numeric(data[col])

    data = data.sort_values([cfg.code_col, "_label_sort_date", cfg.date_col]).reset_index(drop=True)
    g = data.groupby(cfg.code_col, group_keys=False)

    # D 日基础字段
    data["d_trade_date"] = data[cfg.date_col]
    data["d_close"] = data[cfg.close_col]

    # T 日字段：下一交易日
    data["t_trade_date"] = g[cfg.date_col].shift(-1)
    data["t_open"] = g[cfg.open_col].shift(-1)
    data["t_high"] = g[cfg.high_col].shift(-1)
    data["t_close"] = g[cfg.close_col].shift(-1)

    if cfg.pre_close_col in data.columns:
        data["t_pre_close"] = g[cfg.pre_close_col].shift(-1)
    else:
        # 无 pre_close 时，用 D 日收盘近似 T 日昨收。
        data["t_pre_close"] = data[cfg.close_col]

    # T+1 字段：买入后的下一交易日
    data["t1_trade_date"] = g[cfg.date_col].shift(-2)
    data["t1_open"] = g[cfg.open_col].shift(-2)
    data["t1_high"] = g[cfg.high_col].shift(-2)
    data["t1_close"] = g[cfg.close_col].shift(-2)

    if cfg.limit_up_col and cfg.limit_up_col in data.columns:
        data["t_limit_up_price"] = g[cfg.limit_up_col].shift(-1)
    elif "limit_up" in data.columns:
        data["t_limit_up_price"] = g["limit_up"].shift(-1)
    else:
        data["t_limit_up_price"] = calc_limit_up_price(
            data["t_pre_close"],
            code=data[cfg.code_col],
            default_limit_pct=cfg.default_limit_pct,
            infer_market_limit=cfg.infer_market_limit,
        )

    data["t_limit_up_price"] = _to_numeric(data["t_limit_up_price"])

    # 买入价：默认 T 日开盘价，近似集合竞价买入价。
    # 若后续有真实 auction_price / match_price，可通过 buy_price_col 指定。
    if cfg.buy_price_col and cfg.buy_price_col in data.columns:
        data["t_buy_price"] = g[cfg.buy_price_col].shift(-1)
    else:
        data["t_buy_price"] = data["t_open"]
    data["t_buy_price"] = _to_numeric(data["t_buy_price"])

    # 标签：T 日收盘涨停、T 日盘中触板。
    tolerance = float(cfg.limit_tolerance)
    hit_line = data["t_limit_up_price"] * (1.0 - tolerance)
    data["t_limitup_hit"] = (data["t_close"] >= hit_line).astype("Int64")
    data["t_touch_limitup"] = (data["t_high"] >= hit_line).astype("Int64")

    # 标签：T+1 收盘收益 / 最高收益 / 是否上涨 / 是否给过可兑现收益。
    buy = data["t_buy_price"].replace(0, np.nan)
    data["t1_close_ret"] = data["t1_close"] / buy - 1.0
    data["t1_high_ret"] = data["t1_high"] / buy - 1.0
    data["t1_up_hit"] = (data["t1_close_ret"] > 0).astype("Int64")
    data["t1_high_profit_hit"] = (data["t1_high_ret"] >= float(cfg.high_profit_threshold)).astype("Int64")

    # 未成熟样本：没有 T 或 T+1 时，不应参与回测。
    mature_mask = data["t_trade_date"].notna() & data["t1_trade_date"].notna() & data["t_buy_price"].notna()
    data["label_matured"] = mature_mask.astype("Int64")

    label_cols = [
        "t_limitup_hit",
        "t_touch_limitup",
        "t1_up_hit",
        "t1_high_profit_hit",
    ]
    if not cfg.keep_unmatured:
        data = data.loc[mature_mask].copy()
    else:
        for col in label_cols:
            data.loc[~mature_mask, col] = pd.NA
        data.loc[~mature_mask, ["t1_close_ret", "t1_high_ret"]] = np.nan

    data = data.drop(columns=["_label_sort_date"], errors="ignore")
    return data


def build_label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """按全样本汇总标签表现，方便快速验收。"""
    if labels.empty:
        return pd.DataFrame(
            [{
                "样本数": 0,
                "T日收盘涨停率": np.nan,
                "T日触板率": np.nan,
                "T+1上涨率": np.nan,
                "T+1高点收益命中率": np.nan,
                "T+1平均收盘收益": np.nan,
                "T+1平均最高收益": np.nan,
            }]
        )

    return pd.DataFrame(
        [{
            "样本数": int(len(labels)),
            "T日收盘涨停率": float(labels["t_limitup_hit"].mean()),
            "T日触板率": float(labels["t_touch_limitup"].mean()),
            "T+1上涨率": float(labels["t1_up_hit"].mean()),
            "T+1高点收益命中率": float(labels["t1_high_profit_hit"].mean()),
            "T+1平均收盘收益": float(labels["t1_close_ret"].mean()),
            "T+1平均最高收益": float(labels["t1_high_ret"].mean()),
        }]
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Premium V3.5 涨停接力历史真实标签")
    parser.add_argument("--input", required=True, help="历史行情文件路径，支持 csv/parquet/xlsx")
    parser.add_argument("--output", required=True, help="标签输出文件路径，支持 csv/parquet/xlsx")
    parser.add_argument("--summary-output", default=None, help="可选：汇总验收表输出路径")
    parser.add_argument("--high-profit-threshold", type=float, default=0.03, help="T+1 盘中可兑现收益阈值，默认 3%%")
    parser.add_argument("--keep-unmatured", action="store_true", help="保留没有完整 T/T+1 的未成熟样本")
    parser.add_argument("--limit-up-col", default=None, help="行情中真实涨停价字段名，例如 limit_up/up_limit")
    parser.add_argument("--buy-price-col", default=None, help="真实买入价字段名；不填则默认使用 T 日 open")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = _read_table(input_path)
    cfg = LimitupLabelConfig(
        limit_up_col=args.limit_up_col,
        buy_price_col=args.buy_price_col,
        high_profit_threshold=args.high_profit_threshold,
        keep_unmatured=args.keep_unmatured,
    )
    labels = build_limitup_labels(df, cfg)
    _write_table(labels, output_path)

    summary = build_label_summary(labels)
    if args.summary_output:
        _write_table(summary, Path(args.summary_output))
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
