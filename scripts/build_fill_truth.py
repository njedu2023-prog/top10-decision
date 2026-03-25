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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from top10decision.configs import entry_price_proxy_config, fill_truth_config


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
    return PROJECT_ROOT


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
            f"找不到 sample_maturity 文件：{maturity_csv}；"
            f"请先运行 resolve_sample_maturity.py"
        )

    df = safe_read_csv(maturity_csv)
    if df.empty:
        raise ValueError(f"sample_maturity 文件为空：{maturity_csv}")

    trade_col = first_existing(df, ["trade_date"])
    if trade_col is None:
        raise ValueError(f"sample_maturity 缺少 trade_date 列：{maturity_csv}")

    out = df.copy()
    out["trade_date"] = out[trade_col].map(norm_ymd)

    hit = out[out["trade_date"] == norm_ymd(trade_date)].copy()
    if hit.empty:
        raise ValueError(
            f"sample_maturity 中找不到 trade_date={trade_date} 的记录；文件={maturity_csv}"
        )

    fill_ready_col = first_existing(hit, ["PFILL_READY", "FILL_READY"])
    eret_ready_col = first_existing(hit, ["ERET_READY"])
    fully_ready_col = first_existing(hit, ["FULLY_READY"])

    row = hit.iloc[0].to_dict()

    exec_date = norm_ymd(exec_date_override) or norm_ymd(row.get("exec_date"))
    target_date = norm_ymd(target_date_override) or norm_ymd(row.get("target_date"))
    sample_maturity = str(row.get("sample_maturity", "") or "").strip()

    label_ready_fill = _parse_ready_flag(row.get(fill_ready_col, 0)) if fill_ready_col else 0
    label_ready_ret = _parse_ready_flag(row.get(eret_ready_col, 0)) if eret_ready_col else 0
    fully_ready = _parse_ready_flag(row.get(fully_ready_col, 0)) if fully_ready_col else 0

    if not exec_date:
        raise ValueError(f"sample_maturity trade_date={trade_date} 缺少 exec_date")
    if not target_date:
        raise ValueError(f"sample_maturity trade_date={trade_date} 缺少 target_date")

    return MaturityInfo(
        trade_date=norm_ymd(trade_date),
        exec_date=exec_date,
        target_date=target_date,
        sample_maturity=sample_maturity,
        label_ready_fill=label_ready_fill,
        label_ready_ret=label_ready_ret,
        fully_ready=fully_ready,
    )


# =========================
# 候选池读取
# =========================

def load_candidate_pool(paths: Paths, trade_date: str) -> Tuple[pd.DataFrame, str]:
    candidates: List[Path] = []
    if paths.pred_archive.exists():
        candidates.append(paths.pred_archive)
    if paths.pred_latest.exists():
        candidates.append(paths.pred_latest)

    for path in candidates:
        df = safe_read_csv(path)
        if df.empty:
            continue
        df = ensure_ts_code(df)

        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].map(norm_ymd)
            hit = df[df["trade_date"] == trade_date].copy()
            if not hit.empty:
                return hit, str(path)
        else:
            out = df.copy()
            out["trade_date"] = trade_date
            return out, str(path)

    raise FileNotFoundError(
        f"未找到 trade_date={trade_date} 的 pred_source 候选池；"
        f"尝试过：{paths.pred_archive} / {paths.pred_latest}"
    )


# =========================
# T+1 truth 读取
# =========================

@dataclass
class T1Truth:
    daily: pd.DataFrame
    stk_limit: pd.DataFrame
    limit_list_d: pd.DataFrame
    limit_break_d: pd.DataFrame
    suspend_d: pd.DataFrame


def _read_snapshot_csv(base: Path, filename: str) -> pd.DataFrame:
    path = base / filename
    return safe_read_csv(path)


def load_t1_truth(raw_root: Path, exec_date: str) -> T1Truth:
    base = snapshot_dir(raw_root, exec_date)
    return T1Truth(
        daily=_read_snapshot_csv(base, "daily.csv"),
        stk_limit=_read_snapshot_csv(base, "stk_limit.csv"),
        limit_list_d=_read_snapshot_csv(base, "limit_list_d.csv"),
        limit_break_d=_read_snapshot_csv(base, "limit_break_d.csv"),
        suspend_d=_read_snapshot_csv(base, "suspend_d.csv"),
    )


def prep_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "open_t1"])
    out = ensure_ts_code(df)
    open_col = first_existing(out, ["open", "开盘价", "open_price"])
    if open_col is None:
        return pd.DataFrame(columns=["ts_code", "open_t1"])
    out = out[["ts_code", open_col]].copy()
    out = out.rename(columns={open_col: "open_t1"})
    return out


def prep_stk_limit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "up_limit_t1", "down_limit_t1"])
    out = ensure_ts_code(df)
    up_col = first_existing(out, ["up_limit", "涨停价"])
    down_col = first_existing(out, ["down_limit", "跌停价"])
    keep = ["ts_code"]
    if up_col is not None:
        keep.append(up_col)
    if down_col is not None:
        keep.append(down_col)
    out = out[keep].copy()
    ren = {}
    if up_col is not None:
        ren[up_col] = "up_limit_t1"
    if down_col is not None:
        ren[down_col] = "down_limit_t1"
    out = out.rename(columns=ren)
    return out


def _pick_open_times_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "open_times",
            "open_num",
            "open_count",
            "炸板次数",
            "开板次数",
        ],
    )


def prep_limit_list(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "open_times_t1"])
    out = ensure_ts_code(df)
    col = _pick_open_times_col(out)
    if col is None:
        return pd.DataFrame(columns=["ts_code", "open_times_t1"])
    out = out[["ts_code", col]].copy()
    out = out.rename(columns={col: "open_times_t1"})
    return out


def prep_limit_break(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "break_open_times_t1"])
    out = ensure_ts_code(df)
    col = _pick_open_times_col(out)
    if col is None:
        return pd.DataFrame(columns=["ts_code", "break_open_times_t1"])
    out = out[["ts_code", col]].copy()
    out = out.rename(columns={col: "break_open_times_t1"})
    return out


def prep_suspend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "is_suspended_t1"])

    out = ensure_ts_code(df)

    status_col = first_existing(
        out,
        [
            "is_suspended",
            "suspend_type",
            "suspend",
            "停牌标记",
            "停牌",
        ],
    )
    if status_col is None:
        return pd.DataFrame(columns=["ts_code", "is_suspended_t1"])

    x = out[["ts_code", status_col]].copy()
    x["is_suspended_t1"] = normalize_bool_series(x[status_col]).fillna(False)
    return x[["ts_code", "is_suspended_t1"]].copy()


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
            return 1, "strong_open_tradable", float(open_t1), entry_price_proxy_config.mode_default

    has_break = (
        (pd.notna(open_times_t1) and open_times_t1 > 0)
        or (pd.notna(break_open_times_t1) and break_open_times_t1 > 0)
    )
    if has_break:
        if pd.notna(open_t1):
            return 1, "weak_intraday_break_without_minbar", float(open_t1), entry_price_proxy_config.mode_fallback
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
    out["buy_window_start"] = fill_truth_config.buy_window_start
    out["buy_window_end"] = fill_truth_config.buy_window_end

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
            if "sample_maturity" in df.columns else {}
        ),
        "label_ready_fill": int(df["label_ready_fill"].fillna(0).sum()) if "label_ready_fill" in df.columns else 0,
        "label_ready_ret": int(df["label_ready_ret"].fillna(0).sum()) if "label_ready_ret" in df.columns else 0,
        "y_fill_rate": float(df["y_fill"].mean()) if "y_fill" in df.columns and len(df) > 0 else None,
        "quality_counts": (
            df["fill_label_quality"].astype(str).value_counts(dropna=False).to_dict()
            if "fill_label_quality" in df.columns else {}
        ),
        "source": {
            "pred_source": pred_source_path,
            "features_limit": str(paths.features_limit),
            "raw_exec_snapshot": str(snapshot_dir(paths.raw_root, exec_date)),
            "sample_maturity_csv": str(paths.maturity_csv),
        },
        "output": str(paths.out_csv),
    }
    paths.out_meta.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fill_truth_{trade_date}.csv")
    ap.add_argument("--trade-date", required=True, help="T 日，格式 YYYYMMDD")
    ap.add_argument("--exec-date", default="", help="可选覆盖 T+1 日期；默认从 sample_maturity 读取")
    ap.add_argument("--target-date", default="", help="可选覆盖 T+2 日期；默认从 sample_maturity 读取")
    ap.add_argument("--maturity-csv", default="", help="sample_maturity_latest.csv 路径")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = norm_ymd(args.trade_date)
    if not trade_date:
        raise ValueError("trade_date 不能为空")

    paths = build_paths(trade_date=trade_date, maturity_csv=args.maturity_csv)

    maturity = load_maturity_info(
        maturity_csv=paths.maturity_csv,
        trade_date=trade_date,
        exec_date_override=args.exec_date,
        target_date_override=args.target_date,
    )

    df, pred_source_path = build_fill_truth(
        trade_date=maturity.trade_date,
        exec_date=maturity.exec_date,
        target_date=maturity.target_date,
        sample_maturity=maturity.sample_maturity,
        label_ready_fill=maturity.label_ready_fill,
        label_ready_ret=maturity.label_ready_ret,
        paths=paths,
    )

    paths.market_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.out_csv, index=False, encoding="utf-8-sig")
    write_meta(
        df=df,
        trade_date=maturity.trade_date,
        exec_date=maturity.exec_date,
        target_date=maturity.target_date,
        paths=paths,
        pred_source_path=pred_source_path,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "trade_date": maturity.trade_date,
                "exec_date": maturity.exec_date,
                "target_date": maturity.target_date,
                "rows": int(len(df)),
                "out_csv": str(paths.out_csv),
                "out_meta": str(paths.out_meta),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
