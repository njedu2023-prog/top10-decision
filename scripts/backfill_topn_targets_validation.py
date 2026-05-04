# -*- coding: utf-8 -*-
"""
TopN Targets 历史回放验证脚本

定位：
- 独立新增模块，不修改 top10-decision 原预测主链路。
- 扫描历史 data/decision/decision_candidates_YYYYMMDD.csv。
- 按 A 股交易日历将 D 日预测名单映射到下一个真实交易日 T。
- 验证 D 日 TopN Targets 在 T 日收盘后的真实涨跌情况。
- 输出历史明细、累计统计、按日期统计、按排名统计、EV/RiskPenalty 分层统计和独立 HTML 页面。

推荐运行：
    python scripts/backfill_topn_targets_validation.py

可选：
    python scripts/backfill_topn_targets_validation.py --start-date 20260301 --end-date 20260504
    python scripts/backfill_topn_targets_validation.py --topn 10
    python scripts/backfill_topn_targets_validation.py --force

注意：
- 本脚本只读原系统预测文件，不回写、不污染原 prediction/candidates 文件。
- 若缺少行情数据，不中断，标注 validation_status=缺行情 / 待验证。
- 若缺少正式交易日历，会尽力从本地行情/候选文件推断，并在 summary 中标注 calendar_quality。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DECISION_DIR = REPO_ROOT / "data" / "decision"
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation"
DOCS_DIR = REPO_ROOT / "docs"

HISTORY_CSV = OUTPUT_DIR / "topn_targets_validation_history.csv"
LATEST_CSV = OUTPUT_DIR / "topn_targets_validation_latest.csv"
SUMMARY_JSON = OUTPUT_DIR / "topn_targets_validation_summary.json"
SUMMARY_MD = OUTPUT_DIR / "topn_targets_validation_summary.md"
BY_DATE_CSV = OUTPUT_DIR / "topn_targets_validation_by_date.csv"
BY_RANK_CSV = OUTPUT_DIR / "topn_targets_validation_by_rank.csv"
BY_EV_RISK_CSV = OUTPUT_DIR / "topn_targets_validation_by_ev_risk.csv"
HTML_PATH = DOCS_DIR / "topn_targets_validation.html"

CANDIDATE_PATTERN = re.compile(r"decision_candidates_(\d{8})\.csv$")

# 尽量只搜索数据目录，避免扫 .git / venv / 大量无关文件。
MARKET_SEARCH_DIRS = [
    REPO_ROOT / "data" / "market",
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "pred",
    REPO_ROOT / "data" / "pred" / "archive",
    REPO_ROOT / "data",
]

DATE_COLUMNS = ["trade_date", "date", "交易日期", "日期"]
CODE_COLUMNS = ["ts_code", "code", "symbol", "股票代码", "证券代码"]
NAME_COLUMNS = ["name", "stock_name", "股票名称", "证券简称", "名称"]

CLOSE_COLUMNS = [
    "close",
    "close_price",
    "收盘",
    "收盘价",
    "T_close",
    "last_close",
]
PCT_COLUMNS = ["pct_chg", "pct_change", "change_pct", "涨跌幅"]
UP_LIMIT_COLUMNS = ["up_limit", "limit_up", "涨停价"]
DOWN_LIMIT_COLUMNS = ["down_limit", "limit_down", "跌停价"]

EV_COLUMNS = ["EV", "ev", "ExpectedValue", "expected_value"]
RISK_COLUMNS = ["RiskPenalty", "risk_penalty", "Risk", "risk"]
RANK_COLUMNS = ["rank", "Rank", "decision_rank", "EV_rank", "topn_rank", "TopN_rank"]
TOPN_FLAG_COLUMNS = ["is_topn", "topn", "TopN", "is_target", "target"]


@dataclass
class MarketLookupResult:
    status: str
    note: str
    d_close: Optional[float] = None
    t_close: Optional[float] = None
    t_return_pct: Optional[float] = None
    pct_chg: Optional[float] = None
    up_limit: Optional[float] = None
    down_limit: Optional[float] = None
    name: Optional[str] = None


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def norm_date(value) -> Optional[str]:
    """归一化日期为 YYYYMMDD 字符串。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # 处理 20260504.0
    if re.fullmatch(r"\d{8}\.0", s):
        s = s[:8]
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        return digits[:8]
    return None


def fmt_date(value: Optional[str]) -> str:
    d = norm_date(value)
    if not d:
        return ""
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().replace("%", "").replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def find_first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    except Exception as exc:
        print(f"[WARN] 读取 CSV 失败: {path} | {exc}", file=sys.stderr)
        return None


def list_candidate_files() -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    if not DECISION_DIR.exists():
        return files
    for path in sorted(DECISION_DIR.glob("decision_candidates_*.csv")):
        m = CANDIDATE_PATTERN.search(path.name)
        if m:
            files.append((m.group(1), path))
    return files


def discover_csv_files() -> List[Path]:
    seen = set()
    out: List[Path] = []
    for base in MARKET_SEARCH_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*.csv"):
            if p.resolve() in seen:
                continue
            seen.add(p.resolve())
            # 排除验证输出自身，避免回读自己。
            if "outputs/validation" in str(p).replace("\\", "/"):
                continue
            out.append(p)
    return out


class MarketStore:
    """
    轻量本地行情检索器。

    设计目标：
    - 不强依赖某一个固定行情文件名；
    - 尽量从仓库已有 CSV 中发现 trade_date / ts_code / close / pct_chg；
    - 缺数据时标注，不中断。
    """

    def __init__(self, csv_files: Sequence[Path]):
        self.csv_files = list(csv_files)
        self._frames: Dict[Path, Optional[pd.DataFrame]] = {}
        self._date_set: Optional[List[str]] = None

    def _load(self, path: Path) -> Optional[pd.DataFrame]:
        if path not in self._frames:
            self._frames[path] = safe_read_csv(path)
        return self._frames[path]

    def trade_dates(self) -> List[str]:
        if self._date_set is not None:
            return self._date_set

        dates = set()

        # 候选文件日期也加入，用于最低限度日历推断。
        for d, _ in list_candidate_files():
            dates.add(d)

        for path in self.csv_files:
            # 简单控制：太大的文件也可以读，但失败不中断。
            df = self._load(path)
            if df is None or df.empty:
                continue
            dcol = find_first_col(df, DATE_COLUMNS)
            if not dcol:
                continue
            for v in df[dcol].dropna().unique():
                nd = norm_date(v)
                if nd:
                    dates.add(nd)

        self._date_set = sorted(dates)
        return self._date_set

    def next_trade_date(self, d: str) -> Tuple[Optional[str], str]:
        d = norm_date(d)
        if not d:
            return None, "D 日无效"

        dates = self.trade_dates()
        later = [x for x in dates if x > d]
        if later:
            return later[0], "calendar_from_local_data"

        # 最后兜底：工作日推断。该口径不等于严格交易日历，只用于不中断。
        try:
            dt = datetime.strptime(d, "%Y%m%d")
            for _ in range(10):
                dt += timedelta(days=1)
                if dt.weekday() < 5:
                    return dt.strftime("%Y%m%d"), "calendar_estimated_weekday"
        except Exception:
            pass
        return None, "无法推断 T 日"

    def get_row(self, ts_code: str, trade_date: str) -> Optional[pd.Series]:
        ts_code = str(ts_code).strip()
        trade_date = norm_date(trade_date)
        if not ts_code or not trade_date:
            return None

        # 优先搜索文件名含 trade_date 的文件，降低误扫。
        ordered_files = sorted(
            self.csv_files,
            key=lambda p: (0 if trade_date in p.name else 1, len(str(p))),
        )

        for path in ordered_files:
            df = self._load(path)
            if df is None or df.empty:
                continue
            dcol = find_first_col(df, DATE_COLUMNS)
            ccol = find_first_col(df, CODE_COLUMNS)
            close_col = find_first_col(df, CLOSE_COLUMNS)
            if not dcol or not ccol or not close_col:
                continue

            try:
                tmp = df[[dcol, ccol] + [c for c in df.columns if c not in [dcol, ccol]]].copy()
                dates = tmp[dcol].map(norm_date)
                codes = tmp[ccol].astype(str).str.strip()
                mask = (dates == trade_date) & (codes == ts_code)
                if mask.any():
                    return tmp.loc[mask].iloc[0]
            except Exception:
                continue
        return None

    def validate_stock(self, ts_code: str, d_date: str, t_date: str, candidate_row: pd.Series) -> MarketLookupResult:
        d_row = self.get_row(ts_code, d_date)
        t_row = self.get_row(ts_code, t_date)

        d_close = first_float_from_row(candidate_row, ["D_close", "close", "close_price", "last_close", "pre_close"])
        if d_close is None and d_row is not None:
            d_close = first_float_from_row(d_row, CLOSE_COLUMNS)

        if t_row is None:
            if d_close is None:
                return MarketLookupResult(status="缺行情", note="未找到 T 日行情，且无法补齐 D_close", d_close=d_close)
            return MarketLookupResult(status="待验证", note="未找到 T 日行情", d_close=d_close)

        t_close = first_float_from_row(t_row, CLOSE_COLUMNS)
        pct = first_float_from_row(t_row, PCT_COLUMNS)
        up_limit = first_float_from_row(t_row, UP_LIMIT_COLUMNS)
        down_limit = first_float_from_row(t_row, DOWN_LIMIT_COLUMNS)
        name = first_str_from_row(t_row, NAME_COLUMNS)

        if t_close is None:
            return MarketLookupResult(status="缺行情", note="找到 T 日记录但缺少 T_close", d_close=d_close)

        if d_close is None or d_close == 0:
            # 若 pct_chg 存在，无法反推 D_close 的情况下仍可验证涨跌幅。
            if pct is not None:
                return MarketLookupResult(
                    status="已验证",
                    note="缺 D_close，使用行情 pct_chg",
                    d_close=d_close,
                    t_close=t_close,
                    t_return_pct=pct,
                    pct_chg=pct,
                    up_limit=up_limit,
                    down_limit=down_limit,
                    name=name,
                )
            return MarketLookupResult(status="缺行情", note="缺 D_close，无法计算涨跌幅", d_close=d_close, t_close=t_close)

        ret = (t_close / d_close - 1.0) * 100.0
        return MarketLookupResult(
            status="已验证",
            note="",
            d_close=d_close,
            t_close=t_close,
            t_return_pct=ret,
            pct_chg=pct,
            up_limit=up_limit,
            down_limit=down_limit,
            name=name,
        )


def first_float_from_row(row: pd.Series, columns: Sequence[str]) -> Optional[float]:
    for c in columns:
        if c in row.index:
            v = parse_float(row.get(c))
            if v is not None:
                return v
    # 大小写兼容
    lower_map = {str(c).lower(): c for c in row.index}
    for c in columns:
        key = c.lower()
        if key in lower_map:
            v = parse_float(row.get(lower_map[key]))
            if v is not None:
                return v
    return None


def first_str_from_row(row: pd.Series, columns: Sequence[str]) -> Optional[str]:
    for c in columns:
        if c in row.index and pd.notna(row.get(c)):
            s = str(row.get(c)).strip()
            if s:
                return s
    lower_map = {str(c).lower(): c for c in row.index}
    for c in columns:
        key = c.lower()
        if key in lower_map and pd.notna(row.get(lower_map[key])):
            s = str(row.get(lower_map[key])).strip()
            if s:
                return s
    return None


def is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "是", "top", "topn"}


def select_topn(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    """
    从 decision_candidates 中识别 TopN Targets。
    优先顺序：
    1. is_topn/topn 等标记字段；
    2. rank/decision_rank/EV_rank 等排名字段；
    3. EV 降序前 topn；
    4. 原文件顺序前 topn。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["_original_order"] = range(1, len(out) + 1)

    # 1. TopN 标记
    for col in TOPN_FLAG_COLUMNS:
        if col in out.columns:
            flagged = out[out[col].map(is_truthy)].copy()
            if not flagged.empty:
                return attach_rank(flagged, topn)

    # 2. 排名字段
    for col in RANK_COLUMNS:
        if col in out.columns:
            tmp = out.copy()
            tmp["_rank_num"] = pd.to_numeric(tmp[col], errors="coerce")
            ranked = tmp[tmp["_rank_num"].notna()].sort_values(["_rank_num", "_original_order"]).head(topn)
            if not ranked.empty:
                return attach_rank(ranked.drop(columns=["_rank_num"], errors="ignore"), topn)

    # 3. EV 降序
    ev_col = find_first_col(out, EV_COLUMNS)
    if ev_col:
        tmp = out.copy()
        tmp["_ev_num"] = pd.to_numeric(tmp[ev_col], errors="coerce")
        ranked = tmp.sort_values(["_ev_num", "_original_order"], ascending=[False, True]).head(topn)
        return attach_rank(ranked.drop(columns=["_ev_num"], errors="ignore"), topn)

    # 4. 原顺序
    return attach_rank(out.head(topn), topn)


def attach_rank(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    out = df.copy()
    if "TopN_rank" not in out.columns:
        if "_original_order" in out.columns:
            out = out.sort_values("_original_order")
        out["TopN_rank"] = range(1, len(out) + 1)
    return out.drop(columns=["_original_order"], errors="ignore")


def classify_result(ret: Optional[float], up_limit_hit: bool, down_limit_hit: bool) -> str:
    if ret is None or pd.isna(ret):
        return "未验证"
    if up_limit_hit:
        return "涨停"
    if down_limit_hit:
        return "跌停"
    if ret > 0.05:
        return "上涨"
    if ret < -0.05:
        return "下跌"
    return "平盘"


def detect_limit_hit(ret: Optional[float], close_price: Optional[float], up_limit: Optional[float], down_limit: Optional[float]) -> Tuple[bool, bool, str]:
    note = ""
    up_hit = False
    down_hit = False

    if close_price is not None and up_limit is not None and up_limit > 0:
        up_hit = close_price >= up_limit * 0.999
    elif ret is not None:
        # 无涨停价时用涨跌幅估算，覆盖主板/创业板/科创板常见区间，但必须标注估算。
        up_hit = ret >= 9.85 or ret >= 19.7
        if up_hit:
            note += "涨停估算;"

    if close_price is not None and down_limit is not None and down_limit > 0:
        down_hit = close_price <= down_limit * 1.001
    elif ret is not None:
        down_hit = ret <= -9.85 or ret <= -19.7
        if down_hit:
            note += "跌停估算;"

    return up_hit, down_hit, note


def build_validation_rows(
    d_date: str,
    candidate_path: Path,
    market: MarketStore,
    topn: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[List[Dict], Optional[str], str]:
    d_date = norm_date(d_date)
    if start_date and d_date < start_date:
        return [], None, "skip_before_start"
    if end_date and d_date > end_date:
        return [], None, "skip_after_end"

    df = safe_read_csv(candidate_path)
    if df is None or df.empty:
        return [], None, "candidate_empty"

    top_df = select_topn(df, topn=topn)
    if top_df.empty:
        return [], None, "topn_empty"

    t_date, cal_note = market.next_trade_date(d_date)
    rows: List[Dict] = []

    code_col = find_first_col(top_df, CODE_COLUMNS)
    if not code_col:
        return [], t_date, "no_ts_code"

    for _, row in top_df.iterrows():
        ts_code = str(row.get(code_col, "")).strip()
        if not ts_code:
            continue

        lookup = market.validate_stock(ts_code, d_date, t_date, row) if t_date else MarketLookupResult(status="异常", note="无法推断 T 日")
        ret = lookup.t_return_pct
        up_hit, down_hit, limit_note = detect_limit_hit(ret, lookup.t_close, lookup.up_limit, lookup.down_limit)
        result = classify_result(ret, up_hit, down_hit)

        base = row.to_dict()
        name = first_str_from_row(row, NAME_COLUMNS) or lookup.name or ""

        # 避免覆盖原字段：验证字段统一追加在后面。
        base.update({
            "D_trade_date": d_date,
            "D_trade_date_fmt": fmt_date(d_date),
            "T_trade_date": t_date or "",
            "T_trade_date_fmt": fmt_date(t_date),
            "validation_pair": f"D：{fmt_date(d_date)} → T：{fmt_date(t_date)}" if t_date else f"D：{fmt_date(d_date)} → T：",
            "ts_code_norm": ts_code,
            "name_norm": name,
            "D_close": lookup.d_close,
            "T_close": lookup.t_close,
            "T_return_pct": ret,
            "T_result": result,
            "T_limit_hit": bool(up_hit),
            "T_down_limit_hit": bool(down_hit),
            "validation_status": lookup.status,
            "validation_note": ";".join(x for x in [lookup.note, limit_note, cal_note] if x),
        })
        rows.append(base)

    return rows, t_date, cal_note


def summarize(history: pd.DataFrame) -> Dict:
    verified = history[history.get("validation_status", "") == "已验证"].copy() if not history.empty else pd.DataFrame()
    total = int(len(verified))

    if total == 0:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "verified_samples": 0,
            "up_count": 0,
            "up_rate": None,
            "limit_up_count": 0,
            "limit_up_rate": None,
            "down_count": 0,
            "down_rate": None,
            "flat_count": 0,
            "flat_rate": None,
            "mean_return_pct": None,
            "median_return_pct": None,
            "max_return_pct": None,
            "min_return_pct": None,
            "conclusion": "暂无已验证样本",
        }

    ret = pd.to_numeric(verified["T_return_pct"], errors="coerce")
    up_count = int((ret > 0.05).sum())
    down_count = int((ret < -0.05).sum())
    flat_count = int(((ret >= -0.05) & (ret <= 0.05)).sum())
    limit_up_count = int(verified["T_limit_hit"].astype(str).str.lower().isin(["true", "1"]).sum())

    def rate(x: int) -> float:
        return round(x / total * 100.0, 4) if total else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "verified_samples": total,
        "up_count": up_count,
        "up_rate": rate(up_count),
        "limit_up_count": limit_up_count,
        "limit_up_rate": rate(limit_up_count),
        "down_count": down_count,
        "down_rate": rate(down_count),
        "flat_count": flat_count,
        "flat_rate": rate(flat_count),
        "mean_return_pct": round(float(ret.mean()), 4) if ret.notna().any() else None,
        "median_return_pct": round(float(ret.median()), 4) if ret.notna().any() else None,
        "max_return_pct": round(float(ret.max()), 4) if ret.notna().any() else None,
        "min_return_pct": round(float(ret.min()), 4) if ret.notna().any() else None,
        "conclusion": "已生成 TopN Targets 历史后验验证统计",
    }


def group_by_date(history: pd.DataFrame) -> pd.DataFrame:
    verified = history[history.get("validation_status", "") == "已验证"].copy() if not history.empty else pd.DataFrame()
    if verified.empty:
        return pd.DataFrame(columns=["D_trade_date", "T_trade_date", "TopN数量", "上涨数", "上涨率", "涨停数", "涨停率", "平均涨跌幅", "中位涨跌幅"])

    verified["ret_num"] = pd.to_numeric(verified["T_return_pct"], errors="coerce")
    rows = []
    for (d, t), g in verified.groupby(["D_trade_date", "T_trade_date"], dropna=False):
        n = len(g)
        up = int((g["ret_num"] > 0.05).sum())
        limit_up = int(g["T_limit_hit"].astype(str).str.lower().isin(["true", "1"]).sum())
        rows.append({
            "D_trade_date": d,
            "T_trade_date": t,
            "TopN数量": n,
            "上涨数": up,
            "上涨率": round(up / n * 100, 4) if n else None,
            "涨停数": limit_up,
            "涨停率": round(limit_up / n * 100, 4) if n else None,
            "平均涨跌幅": round(float(g["ret_num"].mean()), 4),
            "中位涨跌幅": round(float(g["ret_num"].median()), 4),
        })
    return pd.DataFrame(rows).sort_values("D_trade_date")


def group_by_rank(history: pd.DataFrame) -> pd.DataFrame:
    verified = history[history.get("validation_status", "") == "已验证"].copy() if not history.empty else pd.DataFrame()
    cols = ["排名层级", "样本数", "上涨率", "涨停率", "平均涨跌幅", "中位涨跌幅"]
    if verified.empty or "TopN_rank" not in verified.columns:
        return pd.DataFrame(columns=cols)

    verified["rank_num"] = pd.to_numeric(verified["TopN_rank"], errors="coerce")
    verified["ret_num"] = pd.to_numeric(verified["T_return_pct"], errors="coerce")

    buckets = [
        ("Top1", lambda x: x == 1),
        ("Top1-2", lambda x: (x >= 1) & (x <= 2)),
        ("Top1-3", lambda x: (x >= 1) & (x <= 3)),
        ("Top1-4", lambda x: (x >= 1) & (x <= 4)),
        ("Top1-5", lambda x: (x >= 1) & (x <= 5)),
        ("Top6-10", lambda x: (x >= 6) & (x <= 10)),
    ]
    rows = []
    for label, func in buckets:
        g = verified[func(verified["rank_num"])]
        rows.append(metrics_row(label, g))
    return pd.DataFrame(rows, columns=cols)


def metrics_row(label: str, g: pd.DataFrame) -> Dict:
    n = len(g)
    if n == 0:
        return {"排名层级": label, "样本数": 0, "上涨率": None, "涨停率": None, "平均涨跌幅": None, "中位涨跌幅": None}
    ret = pd.to_numeric(g["T_return_pct"], errors="coerce")
    up = int((ret > 0.05).sum())
    limit_up = int(g["T_limit_hit"].astype(str).str.lower().isin(["true", "1"]).sum())
    return {
        "排名层级": label,
        "样本数": n,
        "上涨率": round(up / n * 100, 4),
        "涨停率": round(limit_up / n * 100, 4),
        "平均涨跌幅": round(float(ret.mean()), 4) if ret.notna().any() else None,
        "中位涨跌幅": round(float(ret.median()), 4) if ret.notna().any() else None,
    }


def group_by_ev_risk(history: pd.DataFrame) -> pd.DataFrame:
    verified = history[history.get("validation_status", "") == "已验证"].copy() if not history.empty else pd.DataFrame()
    cols = ["组合", "EV范围", "RiskPenalty范围", "样本数", "上涨率", "涨停率", "平均涨跌幅", "中位涨跌幅", "结论"]
    if verified.empty:
        return pd.DataFrame(columns=cols)

    ev_col = find_first_col(verified, EV_COLUMNS)
    risk_col = find_first_col(verified, RISK_COLUMNS)
    if not ev_col or not risk_col:
        return pd.DataFrame(columns=cols)

    verified["_ev_num"] = pd.to_numeric(verified[ev_col], errors="coerce")
    verified["_risk_num"] = pd.to_numeric(verified[risk_col], errors="coerce")
    verified = verified[verified["_ev_num"].notna() & verified["_risk_num"].notna()].copy()
    if verified.empty:
        return pd.DataFrame(columns=cols)

    def qbins(series: pd.Series) -> Tuple[float, float]:
        q1 = float(series.quantile(1 / 3))
        q2 = float(series.quantile(2 / 3))
        return q1, q2

    ev_q1, ev_q2 = qbins(verified["_ev_num"])
    risk_q1, risk_q2 = qbins(verified["_risk_num"])

    def ev_group(v: float) -> str:
        if v <= ev_q1:
            return "低EV"
        if v <= ev_q2:
            return "中EV"
        return "高EV"

    def risk_group(v: float) -> str:
        if v <= risk_q1:
            return "低Risk"
        if v <= risk_q2:
            return "中Risk"
        return "高Risk"

    verified["_ev_group"] = verified["_ev_num"].map(ev_group)
    verified["_risk_group"] = verified["_risk_num"].map(risk_group)

    conclusion_map = {
        "高EV + 低Risk": "最优候选池",
        "高EV + 中Risk": "可观察",
        "高EV + 高Risk": "高收益高波动",
        "中EV + 低Risk": "稳健但弹性不足",
        "中EV + 中Risk": "中性",
        "中EV + 高Risk": "风险偏高",
        "低EV + 低Risk": "防守型但弹性不足",
        "低EV + 中Risk": "弱解释力",
        "低EV + 高Risk": "应弱化或剔除",
    }

    rows = []
    ev_order = ["高EV", "中EV", "低EV"]
    risk_order = ["低Risk", "中Risk", "高Risk"]
    for eg in ev_order:
        for rg in risk_order:
            g = verified[(verified["_ev_group"] == eg) & (verified["_risk_group"] == rg)]
            n = len(g)
            ret = pd.to_numeric(g["T_return_pct"], errors="coerce") if n else pd.Series(dtype=float)
            up = int((ret > 0.05).sum()) if n else 0
            limit_up = int(g["T_limit_hit"].astype(str).str.lower().isin(["true", "1"]).sum()) if n else 0
            combo = f"{eg} + {rg}"
            rows.append({
                "组合": combo,
                "EV范围": range_label(eg, ev_q1, ev_q2),
                "RiskPenalty范围": range_label(rg, risk_q1, risk_q2),
                "样本数": n,
                "上涨率": round(up / n * 100, 4) if n else None,
                "涨停率": round(limit_up / n * 100, 4) if n else None,
                "平均涨跌幅": round(float(ret.mean()), 4) if n and ret.notna().any() else None,
                "中位涨跌幅": round(float(ret.median()), 4) if n and ret.notna().any() else None,
                "结论": conclusion_map.get(combo, ""),
            })
    return pd.DataFrame(rows, columns=cols)


def range_label(group: str, q1: float, q2: float) -> str:
    if group.startswith("低"):
        return f"≤ {q1:.6g}"
    if group.startswith("中"):
        return f"{q1:.6g} ~ {q2:.6g}"
    return f"> {q2:.6g}"


def write_summary_md(summary: Dict) -> None:
    lines = [
        "# TopN Targets 验证统计摘要",
        "",
        f"- 生成时间：{summary.get('generated_at', '')}",
        f"- 历史累计样本总数：{summary.get('verified_samples', 0)}",
        f"- 上涨数量：{summary.get('up_count', 0)}",
        f"- 上涨率：{fmt_pct(summary.get('up_rate'))}",
        f"- 涨停数量：{summary.get('limit_up_count', 0)}",
        f"- 涨停率：{fmt_pct(summary.get('limit_up_rate'))}",
        f"- 下跌数量：{summary.get('down_count', 0)}",
        f"- 下跌率：{fmt_pct(summary.get('down_rate'))}",
        f"- 平盘数量：{summary.get('flat_count', 0)}",
        f"- 平盘率：{fmt_pct(summary.get('flat_rate'))}",
        f"- 平均涨跌幅：{fmt_pct(summary.get('mean_return_pct'))}",
        f"- 中位涨跌幅：{fmt_pct(summary.get('median_return_pct'))}",
        f"- 最大涨幅：{fmt_pct(summary.get('max_return_pct'))}",
        f"- 最大跌幅：{fmt_pct(summary.get('min_return_pct'))}",
        "",
        summary.get("conclusion", ""),
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return str(v)


def render_html(summary: Dict, latest: pd.DataFrame, by_date: pd.DataFrame, by_rank: pd.DataFrame, by_ev_risk: pd.DataFrame, history: pd.DataFrame) -> None:
    latest_title = ""
    if not latest.empty:
        d = latest["D_trade_date_fmt"].iloc[0] if "D_trade_date_fmt" in latest.columns else ""
        t = latest["T_trade_date_fmt"].iloc[0] if "T_trade_date_fmt" in latest.columns else ""
        latest_title = f"D：{d} → T：{t}"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TopN Targets 验证系统</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #24292f; }}
    h1, h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }}
    .meta {{ color: #57606a; margin-bottom: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; background: #f6f8fa; }}
    .card b {{ display: block; font-size: 20px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f6f8fa; position: sticky; top: 0; }}
    .table-wrap {{ overflow-x: auto; }}
    .up {{ color: #cf222e; font-weight: 600; }}
    .down {{ color: #1a7f37; font-weight: 600; }}
    .flat {{ color: #57606a; }}
    .limit {{ font-weight: 800; }}
    .note {{ color: #57606a; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>TopN Targets 验证系统</h1>
  <div class="meta">生成时间：{html.escape(str(summary.get("generated_at", "")))}；最新验证：{html.escape(latest_title)}</div>

  <h2>累计统计</h2>
  <div class="cards">
    <div class="card">历史累计样本总数<b>{summary.get("verified_samples", 0)}</b></div>
    <div class="card">上涨率<b>{fmt_pct(summary.get("up_rate"))}</b></div>
    <div class="card">涨停率<b>{fmt_pct(summary.get("limit_up_rate"))}</b></div>
    <div class="card">下跌率<b>{fmt_pct(summary.get("down_rate"))}</b></div>
    <div class="card">平盘率<b>{fmt_pct(summary.get("flat_rate"))}</b></div>
    <div class="card">平均涨跌幅<b>{fmt_pct(summary.get("mean_return_pct"))}</b></div>
    <div class="card">中位涨跌幅<b>{fmt_pct(summary.get("median_return_pct"))}</b></div>
  </div>

  <h2>最新一期 TopN Targets 验证明细</h2>
  <div class="note">保留 D 日 TopN 原字段，并追加 T 日真实涨跌验证字段；排序保持 D 日预测名单顺序不变。</div>
  <div class="table-wrap">{df_to_html(latest, max_rows=30)}</div>

  <h2>按 D 日统计</h2>
  <div class="table-wrap">{df_to_html(by_date, max_rows=120)}</div>

  <h2>TopN 排名有效性统计</h2>
  <div class="table-wrap">{df_to_html(by_rank, max_rows=20)}</div>

  <h2>EV × RiskPenalty 分层统计</h2>
  <div class="table-wrap">{df_to_html(by_ev_risk, max_rows=30)}</div>

  <h2>历史明细最近 100 行</h2>
  <div class="table-wrap">{df_to_html(history.tail(100), max_rows=100)}</div>
</body>
</html>
"""
    HTML_PATH.write_text(html_text, encoding="utf-8")


def df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p class='note'>暂无数据</p>"

    show = df.head(max_rows).copy()

    # 限制列数，优先保留关键信息，避免 HTML 过宽到不可读。
    priority = [
        "TopN_rank", "D_trade_date_fmt", "T_trade_date_fmt", "ts_code", "ts_code_norm",
        "name", "name_norm", "P_fill", "E_ret", "Cost", "RiskPenalty", "EV",
        "D_close", "T_close", "T_return_pct", "T_result", "T_limit_hit",
        "validation_status", "validation_note",
    ]
    cols = [c for c in priority if c in show.columns] + [c for c in show.columns if c not in priority]
    show = show[cols[:40]]

    def fmt_cell(col: str, val) -> str:
        if pd.isna(val):
            return ""
        if col == "T_return_pct":
            try:
                f = float(val)
                cls = "up" if f > 0.05 else "down" if f < -0.05 else "flat"
                sign = "+" if f > 0 else ""
                return f"<span class='{cls}'>{sign}{f:.2f}%</span>"
            except Exception:
                return html.escape(str(val))
        if col in {"上涨率", "涨停率", "下跌率", "平盘率", "平均涨跌幅", "中位涨跌幅"}:
            try:
                return f"{float(val):.2f}%"
            except Exception:
                return html.escape(str(val))
        return html.escape(str(val))

    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    rows = []
    for _, r in show.iterrows():
        tds = "".join(f"<td>{fmt_cell(c, r[c])}</td>" for c in show.columns)
        rows.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def save_outputs(history: pd.DataFrame) -> None:
    ensure_dirs()

    if history.empty:
        history.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(LATEST_CSV, index=False, encoding="utf-8-sig")
        summary = summarize(history)
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_md(summary)
        render_html(summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), history)
        return

    # 保持确定性排序：D 日、TopN 排名、原顺序。
    if "TopN_rank" in history.columns:
        history["_rank_num_sort"] = pd.to_numeric(history["TopN_rank"], errors="coerce")
        history = history.sort_values(["D_trade_date", "_rank_num_sort"], na_position="last").drop(columns=["_rank_num_sort"])
    else:
        history = history.sort_values(["D_trade_date"])

    history.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    # latest = 最大 D 日的验证明细
    latest = pd.DataFrame()
    if "D_trade_date" in history.columns:
        max_d = history["D_trade_date"].dropna().astype(str).max()
        latest = history[history["D_trade_date"].astype(str) == max_d].copy()
    latest.to_csv(LATEST_CSV, index=False, encoding="utf-8-sig")

    summary = summarize(history)
    by_date = group_by_date(history)
    by_rank = group_by_rank(history)
    by_ev_risk = group_by_ev_risk(history)

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summary)
    by_date.to_csv(BY_DATE_CSV, index=False, encoding="utf-8-sig")
    by_rank.to_csv(BY_RANK_CSV, index=False, encoding="utf-8-sig")
    by_ev_risk.to_csv(BY_EV_RISK_CSV, index=False, encoding="utf-8-sig")
    render_html(summary, latest, by_date, by_rank, by_ev_risk, history)


def run_backfill(start_date: Optional[str], end_date: Optional[str], topn: int, force: bool) -> pd.DataFrame:
    ensure_dirs()

    start_date = norm_date(start_date) if start_date else None
    end_date = norm_date(end_date) if end_date else None

    files = list_candidate_files()
    if not files:
        print(f"[WARN] 未找到候选文件: {DECISION_DIR}/decision_candidates_YYYYMMDD.csv")
        history = pd.DataFrame()
        save_outputs(history)
        return history

    market = MarketStore(discover_csv_files())

    all_rows: List[Dict] = []
    for d_date, path in files:
        rows, t_date, note = build_validation_rows(
            d_date=d_date,
            candidate_path=path,
            market=market,
            topn=topn,
            start_date=start_date,
            end_date=end_date,
        )
        if rows:
            print(f"[OK] D={d_date} rows={len(rows)} T={t_date} note={note}")
            all_rows.extend(rows)
        else:
            if note and not note.startswith("skip_"):
                print(f"[SKIP] D={d_date} {path.name} note={note}")

    history = pd.DataFrame(all_rows)
    save_outputs(history)
    print(f"[DONE] history_rows={len(history)}")
    print(f"[DONE] wrote: {HISTORY_CSV}")
    print(f"[DONE] wrote: {HTML_PATH}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill TopN Targets validation history.")
    parser.add_argument("--start-date", default=None, help="起始 D 日，格式 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="结束 D 日，格式 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--topn", type=int, default=10, help="默认 TopN 数量")
    parser.add_argument("--force", action="store_true", help="保留参数：当前实现每次重算历史输出，天然幂等")
    args = parser.parse_args()

    run_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        topn=args.topn,
        force=args.force,
    )


if __name__ == "__main__":
    main()
