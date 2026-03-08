#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_fill_truth.py

用途：
- 基于 T 日 pred_source 候选池（不是全市场 features_limit）
- 使用 features_limit / raw(T+1) 仅为候选池股票补字段与打标签
- 生成 P_fill 学习标签文件：
    data/market/fill_truth_{trade_date}.csv

当前阶段只做 P_fill 标签层：
- y_fill
- fill_label_quality
- entry_price_proxy_t1
- entry_price_proxy_mode

并补充训练窗口 / 成熟度字段：
- trade_date
- exec_date
- target_date
- sample_maturity
- label_ready_fill
- label_ready_ret

不负责：
- 训练样本拼装
- 模型训练
- run_v2.py 接入

主口径（与契约一致）：
- P_fill 只回答：T+1 是否可买
- 计算对象必须严格限制在 pred_source 候选池名单内
- features_limit / raw 只负责给候选池股票补字段，不得反向扩池
- EntryPriceProxy_T+1 不是 PredOpen_T+1，也不是理想成交价
- 在分钟级真值缺失时，首版优先把 y_fill 做稳

本文件已升级：
- 不再负责内部猜 exec_date / target_date
- 默认从 data/market/sample_maturity_latest.csv 读取正式成熟度解析结果
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# =========================
# 基础工具
# =========================

def norm_ymd(x: object) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return s


def to_float(x: object) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return hit
    return None


def ensure_ts_code(df: pd.DataFrame) -> pd.DataFrame:
    hit = first_existing(df, ["ts_code", "code", "symbol", "证券代码"])
    if hit is None:
        raise ValueError("缺少 ts_code/code/symbol 列，无法构建 fill_truth")
    out = df.copy()
    if hit != "ts_code":
        out = out.rename(columns={hit: "ts_code"})
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    return out


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "1": True,
                "true": True,
                "yes": True,
                "y": True,
                "停牌": True,
                "suspend": True,
                "suspended": True,
                "0": False,
                "false": False,
                "no": False,
                "n": False,
                "交易": False,
                "normal": False,
            }
        )
    )


# =========================
# 路径
# =========================

@dataclass
class Paths:
    project_root: Path
    market_dir: Path
    raw_root: Path
    pred_dir: Path
    pred_latest: Path
    pred_archive: Path
    features_limit: Path
    maturity_csv: Path
    out_csv: Path
    out_meta: Path


def detect_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_paths(
    trade_date: str,
    maturity_csv: str = "",
    project_root: Optional[Path] = None,
) -> Paths:
    root = project_root or detect_project_root()
    market_dir = root / "data" / "market"
    raw_root = market_dir / "raw"
    pred_dir = root / "data" / "pred"
    pred_latest = pred_dir / "pred_source_latest.csv"
    pred_archive = pred_dir / "archive" / f"pred_source_{trade_date}.csv"
    features_limit = market_dir / f"features_limit_{trade_date}.csv"
    out_csv = market_dir / f"fill_truth_{trade_date}.csv"
    out_meta = market_dir / f"fill_truth_{trade_date}.meta.json"
    maturity_csv_path = Path(maturity_csv) if maturity_csv else (market_dir / "sample_maturity_latest.csv")

    return Paths(
        project_root=root,
        market_dir=market_dir,
        raw_root=raw_root,
        pred_dir=pred_dir,
        pred_latest=pred_latest,
        pred_archive=pred_archive,
        features_limit=features_limit,
        maturity_csv=maturity_csv_path,
        out_csv=out_csv,
        out_meta=out_meta,
    )


def snapshot_dir(raw_root: Path, ymd: str) -> Path:
    return raw_root / ymd[:4] / ymd


# =========================
# 成熟度解析结果读取
# =========================

@dataclass
class MaturityInfo:
    trade_date: str
    exec_date: str
    target_date: str
    sample_maturity: str
    label_ready_fill: int
    label_ready_ret: int
    fully_ready: int


def _parse_ready_flag(x: object) -> int:
    s = str(x or "").strip()
    return 1 if s in {"1", "true", "True", "TRUE"} else 0


def load_maturity_info(
    maturity_csv: Path,
    trade_date: str,
    exec_date_override: str = "",
    target_date_override: str = "",
) -> MaturityInfo:
    if not maturity_csv.exists():
        raise FileNotFoundError(
            f"缺少样本成熟度解析结果：{maturity_csv}。"
            f"请先运行 scripts/resolve_sample_maturity.py。"
        )

    df = safe_read_csv(maturity_csv)
    if df.empty:
        raise ValueError(f"样本成熟度文件为空：{maturity_csv}")

    if "trade_date" not in df.columns:
        raise ValueError(f"样本成熟度文件缺少 trade_date 列：{maturity_csv}")

    out = df.copy()
    out["trade_date"] = out["trade_date"].map(norm_ymd)
    hit = out[out["trade_date"] == trade_date].copy()
    if hit.empty:
        raise ValueError(
            f"样本成熟度文件中找不到 trade_date={trade_date}：{maturity_csv}"
        )

    row = hit.iloc[-1]

    exec_date = norm_ymd(exec_date_override) or norm_ymd(row.get("exec_date"))
    target_date = norm_ymd(target_date_override) or norm_ymd(row.get("target_date"))
    sample_maturity = str(row.get("sample_maturity") or "").strip()
    pfill_ready = _parse_ready_flag(row.get("PFILL_READY"))
    eret_ready = _parse_ready_flag(row.get("ERET_READY"))
    fully_ready = _parse_ready_flag(row.get("FULLY_READY"))

    if not exec_date:
        raise ValueError(
            f"trade_date={trade_date} 在成熟度文件中 exec_date 为空，"
            f"说明该样本尚未具备 P_fill 真值构建条件：{maturity_csv}"
        )

    if not sample_maturity:
        if pfill_ready and eret_ready:
            sample_maturity = "FULLY_READY"
        elif pfill_ready:
            sample_maturity = "PFILL_READY"
        else:
            sample_maturity = "UNREADY"

    return MaturityInfo(
        trade_date=trade_date,
        exec_date=exec_date,
        target_date=target_date,
        sample_maturity=sample_maturity,
        label_ready_fill=pfill_ready,
        label_ready_ret=eret_ready,
        fully_ready=fully_ready,
    )


# =========================
# 候选池读取（严格以 pred_source 为准）
# =========================

def load_candidate_pool(paths: Paths, trade_date: str) -> Tuple[pd.DataFrame, str]:
    src_path = paths.pred_archive if paths.pred_archive.exists() else paths.pred_latest
    if not src_path.exists():
        raise FileNotFoundError(
            f"找不到 pred_source 候选池：{paths.pred_archive} / {paths.pred_latest}"
        )

    pred = safe_read_csv(src_path)
    if pred.empty:
        raise ValueError(f"pred_source 候选池为空：{src_path}")

    pred = ensure_ts_code(pred)
    if "trade_date" in pred.columns:
        pred["trade_date"] = pred["trade_date"].map(norm_ymd)
        filtered = pred[pred["trade_date"] == trade_date].copy()
        if not filtered.empty:
            pred = filtered
        else:
            pred = pred.copy()
            pred["trade_date"] = trade_date
    else:
        pred["trade_date"] = trade_date

    if pred.empty:
        raise ValueError(f"pred_source 在 trade_date={trade_date} 下无候选样本：{src_path}")

    pred = pred.drop_duplicates(subset=["trade_date", "ts_code"]).copy()
    return pred, str(src_path)


# =========================
# 读取 T+1 快照
# =========================

@dataclass
class T1Truth:
    daily: pd.DataFrame
    stk_limit: pd.DataFrame
    limit_list_d: pd.DataFrame
    limit_break_d: pd.DataFrame
    suspend_d: pd.DataFrame


def load_t1_truth(raw_root: Path, exec_date: str) -> T1Truth:
    snap = snapshot_dir(raw_root, exec_date)
    return T1Truth(
        daily=safe_read_csv(snap / "daily.csv"),
        stk_limit=safe_read_csv(snap / "stk_limit.csv"),
        limit_list_d=safe_read_csv(snap / "limit_list_d.csv"),
        limit_break_d=safe_read_csv(snap / "limit_break_d.csv"),
        suspend_d=safe_read_csv(snap / "suspend_d.csv"),
    )


def prep_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code"])
    out = ensure_ts_code(df)
    rename_map: Dict[str, str] = {}

    for srcs, target in [
        (["open", "开盘价"], "open_t1"),
        (["high", "最高价"], "high_t1"),
        (["low", "最低价"], "low_t1"),
        (["close", "收盘价"], "close_t1"),
        (["vol", "volume", "成交量"], "vol_t1"),
        (["amount", "成交额"], "amount_t1"),
        (["pct_chg", "涨跌幅"], "pct_chg_t1"),
        (["trade_date", "日期"], "exec_date"),
    ]:
        hit = first_existing(out, srcs)
        if hit is not None:
            rename_map[hit] = target

    out = out.rename(columns=rename_map)
    keep = ["ts_code"] + [c for c in rename_map.values() if c in out.columns]
    return out[keep].copy()


def prep_stk_limit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code"])
    out = ensure_ts_code(df)
    rename_map: Dict[str, str] = {}

    for srcs, target in [
        (["up_limit", "涨停价"], "up_limit_t1"),
        (["down_limit", "跌停价"], "down_limit_t1"),
    ]:
        hit = first_existing(out, srcs)
        if hit is not None:
            rename_map[hit] = target

    out = out.rename(columns=rename_map)
    keep = ["ts_code"] + [c for c in rename_map.values() if c in out.columns]
    return out[keep].copy()


def prep_limit_list(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code"])
    out = ensure_ts_code(df)
    rename_map: Dict[str, str] = {}

    for srcs, target in [
        (["limit_type"], "limit_type_t1"),
        (["open_times"], "open_times_t1"),
        (["first_seal_time"], "first_seal_time_t1"),
        (["last_seal_time"], "last_seal_time_t1"),
        (["seal_amount"], "seal_amount_t1"),
    ]:
        hit = first_existing(out, srcs)
        if hit is not None:
            rename_map[hit] = target

    out = out.rename(columns=rename_map)
    keep = ["ts_code"] + [c for c in rename_map.values() if c in out.columns]
    return out[keep].copy()


def prep_limit_break(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code"])
    out = ensure_ts_code(df)
    rename_map: Dict[str, str] = {}

    for srcs, target in [
        (["open_times"], "break_open_times_t1"),
        (["limit_times"], "limit_times_t1"),
        (["latest", "last_price"], "break_latest_t1"),
    ]:
        hit = first_existing(out, srcs)
        if hit is not None:
            rename_map[hit] = target

    out = out.rename(columns=rename_map)
    keep = ["ts_code"] + [c for c in rename_map.values() if c in out.columns]
    return out[keep].copy()


def prep_suspend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "is_suspended_t1"])
    out = ensure_ts_code(df)
    status_col = first_existing(out, ["suspend_type", "suspend_reason", "is_suspended", "status"])
    if status_col is None:
        out["is_suspended_t1"] = True
        return out[["ts_code", "is_suspended_t1"]].copy()

    flag = normalize_bool_series(out[status_col])
    out["is_suspended_t1"] = flag.fillna(True)
    return out[["ts_code", "is_suspended_t1"]].copy()


def merge_t1_truth(truth: T1Truth) -> pd.DataFrame:
    dfs = [
        prep_daily(truth.daily),
        prep_stk_limit(truth.stk_limit),
        prep_limit_list(truth.limit_list_d),
        prep_limit_break(truth.limit_break_d),
        prep_suspend(truth.suspend_d),
    ]
    out: Optional[pd.DataFrame] = None
    for df in dfs:
        if df.empty:
            continue
        out = df if out is None else out.merge(df, on="ts_code", how="outer")
    return out if out is not None else pd.DataFrame(columns=["ts_code"])


# =========================
# 标签逻辑
# =========================

def is_limit_up_dead(row: pd.Series) -> bool:
    open_t1 = to_float(row.get("open_t1"))
    up_limit_t1 = to_float(row.get("up_limit_t1"))
    open_times_t1 = to_float(row.get("open_times_t1"))
    break_open_times_t1 = to_float(row.get("break_open_times_t1"))

    has_open_eq_limit = pd.notna(open_t1) and pd.notna(up_limit_t1) and abs(open_t1 - up_limit_t1) < 1e-8
    no_break_info = (
        (pd.isna(open_times_t1) or open_times_t1 <= 0)
        and (pd.isna(break_open_times_t1) or break_open_times_t1 <= 0)
    )
    return bool(has_open_eq_limit and no_break_info)


def infer_fill_label(row: pd.Series) -> Tuple[int, str, Optional[float], str]:
    is_suspended = row.get("is_suspended_t1")
    if pd.notna(is_suspended) and bool(is_suspended):
        return 0, "strong_suspend", None, ""

    open_t1 = to_float(row.get("open_t1"))
    up_limit_t1 = to_float(row.get("up_limit_t1"))
    open_times_t1 = to_float(row.get("open_times_t1"))
    break_open_times_t1 = to_float(row.get("break_open_times_t1"))

    if pd.notna(open_t1):
        if pd.isna(up_limit_t1) or abs(open_t1 - up_limit_t1) >= 1e-8:
            return 1, "strong_open_tradable", float(open_t1), "open_t1"

    has_break = (
        (pd.notna(open_times_t1) and open_times_t1 > 0)
        or (pd.notna(break_open_times_t1) and break_open_times_t1 > 0)
    )
    if has_break:
        if pd.notna(open_t1):
            return 1, "weak_intraday_break_without_minbar", float(open_t1), "weak_daily_proxy"
        return 1, "weak_intraday_break_without_price", None, ""

    if is_limit_up_dead(row):
        return 0, "strong_dead_limit_up", None, ""

    if pd.isna(open_t1) and pd.isna(up_limit_t1):
        return 0, "weak_missing_truth", None, ""

    return 0, "soft_not_tradable", None, ""


# =========================
# 主流程
# =========================

def build_fill_truth(
    trade_date: str,
    exec_date: str,
    target_date: str,
    sample_maturity: str,
    label_ready_fill: int,
    label_ready_ret: int,
    paths: Paths,
) -> Tuple[pd.DataFrame, str]:
    pred, pred_source_path = load_candidate_pool(paths, trade_date)

    feat = safe_read_csv(paths.features_limit)
    if feat.empty:
        raise FileNotFoundError(f"找不到或读不到 features_limit 文件：{paths.features_limit}")
    feat = ensure_ts_code(feat)
    if "trade_date" in feat.columns:
        feat["trade_date"] = feat["trade_date"].map(norm_ymd)
        feat = feat[feat["trade_date"] == trade_date].copy()
    else:
        feat["trade_date"] = trade_date

    truth = load_t1_truth(paths.raw_root, exec_date)
    t1 = merge_t1_truth(truth)

    pred_base_cols = ["trade_date", "ts_code"]
    for c in [
        "name",
        "rank",
        "prob",
        "StrengthScore",
        "ThemeBoost",
        "board",
        "run_id",
        "run_attempt",
        "commit_sha",
        "generated_at_utc",
    ]:
        if c in pred.columns:
            pred_base_cols.append(c)

    out = pred[pred_base_cols].drop_duplicates(subset=["trade_date", "ts_code"]).copy()

    # 用 features_limit 仅补充候选池股票的字段，不得扩池
    feat = feat.drop_duplicates(subset=["trade_date", "ts_code"]).copy()
    feat_cols_to_add = [
        c for c in feat.columns
        if c not in out.columns and c not in {"trade_date", "ts_code"}
    ]
    if feat_cols_to_add:
        out = out.merge(
            feat[["trade_date", "ts_code"] + feat_cols_to_add],
            on=["trade_date", "ts_code"],
            how="left",
        )

    # T+1 truth 同样只给候选池补字段，不扩池
    out = out.merge(t1, on="ts_code", how="left")
    out["exec_date"] = exec_date
    out["target_date"] = target_date

    labels = out.apply(infer_fill_label, axis=1, result_type="expand")
    labels.columns = [
        "y_fill",
        "fill_label_quality",
        "entry_price_proxy_t1",
        "entry_price_proxy_mode",
    ]
    out = pd.concat([out, labels], axis=1)

    out["sample_maturity"] = sample_maturity
    out["label_ready_fill"] = int(label_ready_fill)
    out["label_ready_ret"] = int(label_ready_ret)

    out["label_version"] = "pfill_truth_v4_maturity_resolved"
    out["buy_window_start"] = "09:30:00"
    out["buy_window_end"] = "10:30:00"

    front = [
        "trade_date",
        "exec_date",
        "target_date",
        "ts_code",
        "name",
        "rank",
        "prob",
        "StrengthScore",
        "ThemeBoost",
        "board",
        "sample_maturity",
        "label_ready_fill",
        "label_ready_ret",
        "y_fill",
        "fill_label_quality",
        "entry_price_proxy_t1",
        "entry_price_proxy_mode",
        "label_version",
        "buy_window_start",
        "buy_window_end",
    ]
    remain = [c for c in out.columns if c not in front]
    out = out[[c for c in front if c in out.columns] + remain].copy()

    out["y_fill"] = out["y_fill"].fillna(0).astype(int)
    out["label_ready_fill"] = out["label_ready_fill"].fillna(0).astype(int)
    out["label_ready_ret"] = out["label_ready_ret"].fillna(0).astype(int)

    return out, pred_source_path


def write_meta(
    df: pd.DataFrame,
    trade_date: str,
    exec_date: str,
    target_date: str,
    paths: Paths,
    pred_source_path: str,
) -> None:
    payload = {
        "trade_date": trade_date,
        "exec_date": exec_date,
        "target_date": target_date,
        "rows": int(len(df)),
        "sample_maturity": (
            df["sample_maturity"].astype(str).value_counts(dropna=False).to_dict()
            if "sample_maturity" in df.columns
            else {}
        ),
        "label_ready_fill": int(df["label_ready_fill"].fillna(0).sum()) if "label_ready_fill" in df.columns else 0,
        "label_ready_ret": int(df["label_ready_ret"].fillna(0).sum()) if "label_ready_ret" in df.columns else 0,
        "y_fill_1": int((df["y_fill"] == 1).sum()) if "y_fill" in df.columns else 0,
        "y_fill_0": int((df["y_fill"] == 0).sum()) if "y_fill" in df.columns else 0,
        "quality_counts": (
            df["fill_label_quality"].astype(str).value_counts(dropna=False).to_dict()
            if "fill_label_quality" in df.columns
            else {}
        ),
        "source": {
            "pred_source": pred_source_path,
            "features_limit": str(paths.features_limit),
            "raw_root": str(paths.raw_root),
            "sample_maturity_csv": str(paths.maturity_csv),
        },
        "output": str(paths.out_csv),
    }
    paths.out_meta.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="构建 P_fill 标签层 fill_truth_{trade_date}.csv")
    ap.add_argument("--trade-date", required=True, help="T 日，格式 YYYYMMDD")
    ap.add_argument(
        "--exec-date",
        default="",
        help="T+1 执行日，格式 YYYYMMDD；默认从 sample_maturity_latest.csv 读取",
    )
    ap.add_argument(
        "--target-date",
        default="",
        help="T+2 目标日，格式 YYYYMMDD；默认从 sample_maturity_latest.csv 读取",
    )
    ap.add_argument(
        "--maturity-csv",
        default="",
        help="样本成熟度解析结果路径；留空默认 data/market/sample_maturity_latest.csv",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    trade_date = norm_ymd(args.trade_date)
    if len(trade_date) != 8:
        raise ValueError("--trade-date 必须是 YYYYMMDD")

    paths = build_paths(
        trade_date=trade_date,
        maturity_csv=args.maturity_csv,
    )
    paths.market_dir.mkdir(parents=True, exist_ok=True)

    maturity_info = load_maturity_info(
        maturity_csv=paths.maturity_csv,
        trade_date=trade_date,
        exec_date_override=args.exec_date,
        target_date_override=args.target_date,
    )

    df, pred_source_path = build_fill_truth(
        trade_date=trade_date,
        exec_date=maturity_info.exec_date,
        target_date=maturity_info.target_date,
        sample_maturity=maturity_info.sample_maturity,
        label_ready_fill=maturity_info.label_ready_fill,
        label_ready_ret=maturity_info.label_ready_ret,
        paths=paths,
    )
    df.to_csv(paths.out_csv, index=False, encoding="utf-8-sig")
    write_meta(
        df,
        trade_date,
        maturity_info.exec_date,
        maturity_info.target_date,
        paths,
        pred_source_path,
    )

    print(f"[build_fill_truth] trade_date={trade_date}")
    print(f"[build_fill_truth] exec_date={maturity_info.exec_date}")
    print(f"[build_fill_truth] target_date={maturity_info.target_date}")
    print(f"[build_fill_truth] sample_maturity={maturity_info.sample_maturity}")
    print(f"[build_fill_truth] label_ready_fill={maturity_info.label_ready_fill}")
    print(f"[build_fill_truth] label_ready_ret={maturity_info.label_ready_ret}")
    print(f"[build_fill_truth] rows={len(df)}")
    print(f"[build_fill_truth] pred_source={pred_source_path}")
    print(f"[build_fill_truth] maturity_csv={paths.maturity_csv}")
    print(f"[build_fill_truth] out={paths.out_csv}")
    print(f"[build_fill_truth] meta={paths.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
