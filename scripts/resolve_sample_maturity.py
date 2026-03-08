#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resolve_sample_maturity.py

职责：
- 将“样本成熟度 / 训练日期机制”独立成正式基础设施
- 给定 current_run_date、raw 快照目录、候选 trade_date
- 输出每个 trade_date 的：
  trade_date
  exec_date
  target_date
  sample_maturity
  PFILL_READY
  ERET_READY
  FULLY_READY

核心规则（严格锚定 Core Algorithm Spec）：
- T 日先预测未来
- T 日只训练今天刚成熟的过去样本
- P_fill / PredOpen_T+1：用 T 日真值训练 T-1 对 T 的预测
- E_ret / PredClose_T+2 / PremiumRet：用 T 日真值训练 T-2 对 T 的预测
- T 日训练出的新模型，只能从 T+1 生效
- 当天若训练条件不足，允许 skip，但不得打死主预测链

本脚本只负责“日期成熟度解析”，不参与训练，不参与真值构建。

设计原则：
1. 不猜日期：exec_date / target_date 必须由 raw 快照中真实存在的交易日序列解析出来
2. 不扩池：候选样本由外部传入的 trade_date 决定，本脚本不自行扩展样本池
3. 可复用：后续 run_decision_daily.yml / build_fill_truth.py / train_pfill.py 统一使用这里的结果
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


# =========================
# 数据结构
# =========================

@dataclass(frozen=True)
class SampleMaturityRow:
    trade_date: str
    exec_date: str
    target_date: str
    sample_maturity: str
    PFILL_READY: int
    ERET_READY: int
    FULLY_READY: int


# =========================
# 基础工具
# =========================

def is_yyyymmdd(s: str) -> bool:
    return isinstance(s, str) and len(s) == 8 and s.isdigit()


def norm_date_str(s: object) -> Optional[str]:
    """
    归一化日期到 YYYYMMDD。
    允许输入：
    - 20260307
    - "20260307"
    - "2026-03-07"
    - "2026/03/07"
    """
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None

    text = text.replace("-", "").replace("/", "")
    if is_yyyymmdd(text):
        return text
    return None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sort_trade_dates(dates: Iterable[str]) -> List[str]:
    uniq = {d for d in dates if is_yyyymmdd(d)}
    return sorted(uniq)


# =========================
# raw 快照交易日解析
# =========================

def discover_raw_trade_dates(raw_root: Path) -> List[str]:
    """
    从 raw 根目录发现所有可用交易日。

    兼容目录形态（示例）：
    - data/market/raw/2026/20260307/
    - data/market/raw/20260307/
    - data/market/raw/2026/20260307/*.csv
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root 不存在：{raw_root}")

    found: Set[str] = set()

    # 递归扫描 1~2 层已够用，但直接 rglob 更稳
    for p in raw_root.rglob("*"):
        if not p.exists():
            continue
        name = p.name

        # 目录名恰好是 YYYYMMDD
        if p.is_dir() and is_yyyymmdd(name):
            found.add(name)
            continue

        # 文件名可能包含 YYYYMMDD，谨慎提取最后一级父目录优先
        # 这里不主动从任意文件名“猜”日期，避免误判
        # 只额外接受 parent 是 YYYYMMDD 的情况
        parent_name = p.parent.name if p.parent else ""
        if is_yyyymmdd(parent_name):
            found.add(parent_name)

    dates = sorted(found)
    if not dates:
        raise RuntimeError(f"未在 raw_root 下发现任何交易日目录：{raw_root}")

    return dates


def next_n_trade_date(trade_dates: Sequence[str], base_date: str, n_after: int) -> str:
    """
    返回 base_date 之后第 n_after 个交易日。
    n_after=1 -> 下一个交易日
    n_after=2 -> 下下个交易日

    若不存在，返回空字符串。
    """
    if n_after <= 0:
        raise ValueError("n_after 必须 >= 1")

    try:
        idx = trade_dates.index(base_date)
    except ValueError:
        return ""

    target_idx = idx + n_after
    if target_idx >= len(trade_dates):
        return ""
    return trade_dates[target_idx]


# =========================
# 候选 trade_date 读取
# =========================

def load_trade_dates_from_csv(path: Path, column: str = "trade_date") -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"trade_dates csv 不存在：{path}")

    rows: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"csv 表头为空：{path}")
        if column not in reader.fieldnames:
            raise RuntimeError(
                f"csv 缺少指定列：{column}，实际列为：{reader.fieldnames}"
            )
        for row in reader:
            d = norm_date_str(row.get(column))
            if d:
                rows.append(d)

    return sort_trade_dates(rows)


def load_trade_dates_from_json(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"trade_dates json 不存在：{path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    out: List[str] = []

    # 支持：
    # ["20260301", "20260302"]
    # [{"trade_date":"20260301"}, ...]
    # {"trade_dates":["20260301", ...]}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                d = norm_date_str(item.get("trade_date"))
                if d:
                    out.append(d)
            else:
                d = norm_date_str(item)
                if d:
                    out.append(d)

    elif isinstance(data, dict):
        vals = data.get("trade_dates", [])
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, dict):
                    d = norm_date_str(item.get("trade_date"))
                    if d:
                        out.append(d)
                else:
                    d = norm_date_str(item)
                    if d:
                        out.append(d)
    else:
        raise RuntimeError(f"不支持的 json 结构：{path}")

    return sort_trade_dates(out)


def parse_trade_dates_args(
    trade_dates_inline: Optional[str],
    trade_dates_file: Optional[Path],
) -> List[str]:
    """
    候选 trade_date 输入优先级：
    1. --trade-dates "20260301,20260302"
    2. --trade-dates-file xxx.csv/json/txt
    """
    out: List[str] = []

    if trade_dates_inline:
        for part in trade_dates_inline.split(","):
            d = norm_date_str(part)
            if d:
                out.append(d)

    if trade_dates_file:
        suffix = trade_dates_file.suffix.lower()
        if suffix == ".csv":
            out.extend(load_trade_dates_from_csv(trade_dates_file))
        elif suffix == ".json":
            out.extend(load_trade_dates_from_json(trade_dates_file))
        else:
            # txt / md / 无后缀，按逐行处理
            if not trade_dates_file.exists():
                raise FileNotFoundError(f"trade_dates file 不存在：{trade_dates_file}")
            for line in trade_dates_file.read_text(encoding="utf-8").splitlines():
                d = norm_date_str(line.strip())
                if d:
                    out.append(d)

    out = sort_trade_dates(out)
    if not out:
        raise RuntimeError("未提供任何有效的 candidate trade_date")

    return out


# =========================
# 核心解析逻辑
# =========================

def compute_maturity_label(
    pfill_ready: bool,
    eret_ready: bool,
) -> str:
    if pfill_ready and eret_ready:
        return "FULLY_READY"
    if pfill_ready:
        return "PFILL_READY"
    return "UNREADY"


def resolve_sample_maturity_rows(
    current_run_date: str,
    all_trade_dates_from_raw: Sequence[str],
    candidate_trade_dates: Sequence[str],
) -> List[SampleMaturityRow]:
    """
    基于 raw 中真实存在的交易日序列，解析候选样本的成熟度。

    规则：
    - exec_date = trade_date 之后第 1 个交易日
    - target_date = trade_date 之后第 2 个交易日
    - PFILL_READY: exec_date 非空 且 exec_date <= current_run_date
    - ERET_READY : target_date 非空 且 target_date <= current_run_date
    - FULLY_READY = PFILL_READY and ERET_READY
    """
    raw_dates = sort_trade_dates(all_trade_dates_from_raw)
    rows: List[SampleMaturityRow] = []

    for trade_date in sort_trade_dates(candidate_trade_dates):
        # 若 trade_date 本身都不在 raw 序列中，不做猜测，直接视为未就绪
        if trade_date not in raw_dates:
            rows.append(
                SampleMaturityRow(
                    trade_date=trade_date,
                    exec_date="",
                    target_date="",
                    sample_maturity="UNREADY",
                    PFILL_READY=0,
                    ERET_READY=0,
                    FULLY_READY=0,
                )
            )
            continue

        exec_date = next_n_trade_date(raw_dates, trade_date, 1)
        target_date = next_n_trade_date(raw_dates, trade_date, 2)

        pfill_ready = bool(exec_date) and exec_date <= current_run_date
        eret_ready = bool(target_date) and target_date <= current_run_date
        fully_ready = pfill_ready and eret_ready

        rows.append(
            SampleMaturityRow(
                trade_date=trade_date,
                exec_date=exec_date,
                target_date=target_date,
                sample_maturity=compute_maturity_label(pfill_ready, eret_ready),
                PFILL_READY=1 if pfill_ready else 0,
                ERET_READY=1 if eret_ready else 0,
                FULLY_READY=1 if fully_ready else 0,
            )
        )

    return rows


# =========================
# 输出
# =========================

def write_csv(rows: Sequence[SampleMaturityRow], output_csv: Path) -> None:
    ensure_parent_dir(output_csv)
    fieldnames = [
        "trade_date",
        "exec_date",
        "target_date",
        "sample_maturity",
        "PFILL_READY",
        "ERET_READY",
        "FULLY_READY",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(
    rows: Sequence[SampleMaturityRow],
    output_json: Path,
    current_run_date: str,
    raw_root: str,
) -> None:
    ensure_parent_dir(output_json)
    payload = {
        "current_run_date": current_run_date,
        "raw_root": raw_root,
        "count": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(rows: Sequence[SampleMaturityRow]) -> None:
    total = len(rows)
    pfill_ready = sum(r.PFILL_READY for r in rows)
    eret_ready = sum(r.ERET_READY for r in rows)
    fully_ready = sum(r.FULLY_READY for r in rows)

    print(
        json.dumps(
            {
                "total": total,
                "PFILL_READY": pfill_ready,
                "ERET_READY": eret_ready,
                "FULLY_READY": fully_ready,
            },
            ensure_ascii=False,
        )
    )


# =========================
# CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解析样本成熟度（trade_date / exec_date / target_date / PFILL_READY / ERET_READY / FULLY_READY）"
    )

    parser.add_argument(
        "--current-run-date",
        required=True,
        help="当前运行日，YYYYMMDD",
    )
    parser.add_argument(
        "--raw-root",
        default="data/market/raw",
        help="raw 快照根目录，默认 data/market/raw",
    )

    # 候选 trade_date：支持内联字符串 + 文件
    parser.add_argument(
        "--trade-dates",
        default="",
        help='候选 trade_date，逗号分隔，如 "20260301,20260302"',
    )
    parser.add_argument(
        "--trade-dates-file",
        default="",
        help="候选 trade_date 文件路径，支持 csv/json/txt",
    )

    # csv 专用列名
    parser.add_argument(
        "--trade-date-column",
        default="trade_date",
        help="当 trade_dates_file 为 csv 时，读取的列名，默认 trade_date",
    )

    parser.add_argument(
        "--output-csv",
        default="data/market/sample_maturity_latest.csv",
        help="输出 csv 路径",
    )
    parser.add_argument(
        "--output-json",
        default="data/market/sample_maturity_latest.json",
        help="输出 json 路径",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    current_run_date = norm_date_str(args.current_run_date)
    if not current_run_date:
        raise RuntimeError("--current-run-date 非法，必须是 YYYYMMDD")

    raw_root = Path(args.raw_root)
    trade_dates_file = Path(args.trade_dates_file) if args.trade_dates_file else None

    # 先发现 raw 中真实存在的交易日
    raw_trade_dates = discover_raw_trade_dates(raw_root)

    # 再读取候选样本 trade_date
    candidate_trade_dates: List[str] = []
    if trade_dates_file and trade_dates_file.suffix.lower() == ".csv":
        candidate_trade_dates.extend(
            load_trade_dates_from_csv(trade_dates_file, column=args.trade_date_column)
        )
        if args.trade_dates:
            candidate_trade_dates.extend(
                [d for d in (norm_date_str(x) for x in args.trade_dates.split(",")) if d]
            )
        candidate_trade_dates = sort_trade_dates(candidate_trade_dates)
        if not candidate_trade_dates:
            raise RuntimeError("未提供任何有效的 candidate trade_date")
    else:
        candidate_trade_dates = parse_trade_dates_args(
            trade_dates_inline=args.trade_dates,
            trade_dates_file=trade_dates_file,
        )

    rows = resolve_sample_maturity_rows(
        current_run_date=current_run_date,
        all_trade_dates_from_raw=raw_trade_dates,
        candidate_trade_dates=candidate_trade_dates,
    )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    write_csv(rows, output_csv)
    write_json(
        rows,
        output_json=output_json,
        current_run_date=current_run_date,
        raw_root=str(raw_root),
    )
    print_summary(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[resolve_sample_maturity] ERROR: {e}", file=sys.stderr)
        raise
