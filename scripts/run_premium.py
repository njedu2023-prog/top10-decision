#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium Runner

用法：
  python scripts/run_premium.py train
  python scripts/run_premium.py predict
  python scripts/run_premium.py all

可选参数：
  --trade_date YYYYMMDD   （预留：未来支持指定2日 trade_date 推理/训练窗口锚定）
  --verbose              （打印更多信息）
"""

from __future__ import annotations

import argparse
import sys

# 允许直接运行脚本时正确导入 src 包
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig
from top10decision.premium.predict import predict_latest
from top10decision.premium.train import train_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Premium module (train/predict).")
    p.add_argument("cmd", choices=["train", "predict", "all"], help="执行命令")
    p.add_argument("--trade_date", default="", help="预留：指定 trade_date（YYYYMMDD，2日）")
    p.add_argument("--verbose", action="store_true", help="输出更多日志")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PremiumConfig.load()

    if args.verbose:
        print("[premium] repo_root:", cfg.repo_root())
        print("[premium] out_dir:", cfg.out_root())
        print("[premium] decision_glob:", cfg.decision_input_glob)
        print("[premium] close_glob:", cfg.close_input_glob)
        print("[premium] model_version:", cfg.model_version)

    # P0：trade_date 先作为预留，不强制使用（避免误导）
    if args.trade_date and args.verbose:
        print("[premium] NOTE: --trade_date is reserved in P0 and currently not enforced.")

    if args.cmd in ("train", "all"):
        r = train_models(cfg)
        print(f"[premium][train] trained={r.trained} n_days={r.n_days} n_samples={r.n_samples} reason={r.reason}")

        # 如果只是 train 到这里
        if args.cmd == "train":
            return 0 if r.trained or "不足" in r.reason or "没有可用样本" in r.reason else 1

    if args.cmd in ("predict", "all"):
        pr = predict_latest(cfg)
        print(f"[premium][predict] ok={pr.ok} trade_date={pr.trade_date} reason={pr.reason}")
        if pr.rank_csv:
            print(f"[premium][predict] rank_csv: {pr.rank_csv}")
        if pr.rank_md:
            print(f"[premium][predict] rank_md: {pr.rank_md}")
        return 0 if pr.ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
