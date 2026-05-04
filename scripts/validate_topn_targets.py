# -*- coding: utf-8 -*-
"""
TopN Targets 每日增量验证脚本

定位：
- 独立新增模块，不修改 top10-decision 原预测主链路。
- 用于每日 T 日收盘后，验证已经成熟的 D 日 TopN Targets。
- 默认会调用 backfill_topn_targets_validation.py 的核心能力，重算历史验证输出，保证幂等和一致性。
- 支持指定 D 日单独验证，也支持默认自动验证最近窗口。

推荐运行：
    python scripts/validate_topn_targets.py

可选：
    python scripts/validate_topn_targets.py --d-date 20260506
    python scripts/validate_topn_targets.py --lookback-days 20
    python scripts/validate_topn_targets.py --topn 10

说明：
- 当前 V1 采用“安全重算”策略：每日增量入口默认只回放最近 lookback-days 对应的 D 日范围。
- 这样可以避免重复追加，也可以自动修正历史缺行情变为已验证的状态。
- 若需要一次性全历史回放，请直接运行：
    python scripts/backfill_topn_targets_validation.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# 允许脚本从仓库根目录或 scripts 目录运行。
try:
    from scripts.backfill_topn_targets_validation import (
        DECISION_DIR,
        HISTORY_CSV,
        list_candidate_files,
        norm_date,
        run_backfill,
    )
except ModuleNotFoundError:
    from backfill_topn_targets_validation import (
        DECISION_DIR,
        HISTORY_CSV,
        list_candidate_files,
        norm_date,
        run_backfill,
    )


def latest_candidate_date() -> Optional[str]:
    files = list_candidate_files()
    if not files:
        return None
    return sorted(d for d, _ in files)[-1]


def infer_start_date_by_lookback(end_date: str, lookback_days: int) -> str:
    """
    用自然日窗口粗略限制回放范围。
    真正 D→T 仍由 backfill 脚本内部交易日历处理。
    """
    end = datetime.strptime(end_date, "%Y%m%d")
    start = end - timedelta(days=max(lookback_days, 1))
    return start.strftime("%Y%m%d")


def run_incremental(
    d_date: Optional[str] = None,
    lookback_days: int = 20,
    topn: int = 10,
    force: bool = False,
) -> pd.DataFrame:
    """
    每日增量验证入口。

    策略：
    1. 如果指定 --d-date，则只验证该 D 日。
    2. 如果未指定，则找到最新 candidate 日期，向前回看 lookback_days 自然日。
    3. 调用 backfill 的统一逻辑生成全部 validation 输出。
    """
    if d_date:
        d = norm_date(d_date)
        if not d:
            raise ValueError(f"无效 d_date: {d_date}")
        return run_backfill(start_date=d, end_date=d, topn=topn, force=force)

    latest_d = latest_candidate_date()
    if not latest_d:
        print(f"[WARN] 未找到候选文件目录或候选文件: {DECISION_DIR}")
        return run_backfill(start_date=None, end_date=None, topn=topn, force=force)

    start_d = infer_start_date_by_lookback(latest_d, lookback_days)
    print(f"[INFO] incremental window: {start_d} -> {latest_d}, topn={topn}")

    # 注意：这里不是追加，而是窗口重算；输出文件由 backfill 统一生成，保证幂等。
    return run_backfill(start_date=start_d, end_date=latest_d, topn=topn, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TopN Targets incrementally.")
    parser.add_argument("--d-date", default=None, help="指定 D 日单独验证，格式 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=20, help="未指定 D 日时，默认回看最近多少自然日")
    parser.add_argument("--topn", type=int, default=10, help="默认 TopN 数量")
    parser.add_argument("--force", action="store_true", help="保留参数：当前实现重算输出，天然幂等")
    args = parser.parse_args()

    history = run_incremental(
        d_date=args.d_date,
        lookback_days=args.lookback_days,
        topn=args.topn,
        force=args.force,
    )

    print(f"[DONE] incremental validation rows={len(history)}")
    print(f"[DONE] history output: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
