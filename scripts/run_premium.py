#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium Runner

用法：
  python scripts/run_premium.py predict
  python scripts/run_premium.py train
  python scripts/run_premium.py all

可选参数：
  --trade_date YYYYMMDD   （预留：未来支持指定 trade_date 回放/锚定）
  --verbose              （打印更多信息）

说明（工程锁死）：
- Actions / 自动化默认只跑 predict（保证端到端产出，不被训练样本不足卡死）
- train 仍保留给你手动调试用
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接运行脚本时正确导入 src 包
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top10decision.premium.config import PremiumConfig
from top10decision.premium.predict import predict_latest

# train 可能在某些阶段不存在/被你暂时移除，这里做防御导入
try:
    from top10decision.premium.train import train_models  # type: ignore
except Exception:  # pragma: no cover
    train_models = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Premium module (train/predict).")
    p.add_argument("cmd", choices=["train", "predict", "all"], help="执行命令")
    p.add_argument("--trade_date", default="", help="预留：指定 trade_date（YYYYMMDD）")
    p.add_argument("--verbose", action="store_true", help="输出更多日志")
    return p.parse_args()


def _safe_get(obj, name: str, default=""):
    return getattr(obj, name, default)


def main() -> int:
    args = parse_args()
    cfg = PremiumConfig.load()

    if args.verbose:
        print("[premium] repo_root:", cfg.repo_root())
        # out_root/reports_root 可能存在于新 config；没有也不致命
        if hasattr(cfg, "out_root"):
            print("[premium] out_dir:", cfg.out_root())
        if hasattr(cfg, "reports_root"):
            print("[premium] reports_dir:", cfg.reports_root())

        # 兼容旧/新字段名
        print("[premium] pred_source_latest:", _safe_get(cfg, "pred_source_latest", ""))
        print("[premium] decision_glob:", _safe_get(cfg, "decision_glob", _safe_get(cfg, "decision_input_glob", "")))
        print("[premium] market_cache_dir:", _safe_get(cfg, "market_cache_dir", "data/market"))
        print("[premium] market_fetch_mode:", _safe_get(cfg, "market_fetch_mode", "cache_first"))
        print("[premium] top_n:", _safe_get(cfg, "top_n", _safe_get(cfg, "topk", "")))
        print("[premium] horizon:", _safe_get(cfg, "horizon_trade_days", _safe_get(cfg, "horizon", "")))

    # P0：trade_date 先作为预留，不强制使用（避免误导）
    if args.trade_date and args.verbose:
        print("[premium] NOTE: --trade_date is reserved and currently not enforced.")

    # train（保留给你手动用；Actions 可不跑）
    if args.cmd in ("train", "all"):
        if train_models is None:
            print("[premium][train] skipped: train_models not available.")
            if args.cmd == "train":
                return 0
        else:
            r = train_models(cfg)
            # 兼容不同 TrainResult 结构
            trained = _safe_get(r, "trained", False)
            n_days = _safe_get(r, "n_days", "")
            n_samples = _safe_get(r, "n_samples", "")
            reason = _safe_get(r, "reason", "")
            print(f"[premium][train] trained={trained} n_days={n_days} n_samples={n_samples} reason={reason}")

            if args.cmd == "train":
                # 样本不足不算失败（避免自动化被卡死）
                if trained:
                    return 0
                if isinstance(reason, str) and ("不足" in reason or "没有可用样本" in reason):
                    return 0
                return 1

    # predict（主线）
    if args.cmd in ("predict", "all"):
        pr = predict_latest(cfg)
        ok = _safe_get(pr, "ok", False)
        trade_date = _safe_get(pr, "trade_date", "")
        reason = _safe_get(pr, "reason", "")
        print(f"[premium][predict] ok={ok} trade_date={trade_date} reason={reason}")

        # 兼容不同返回结构：有就打印
        for k in ("target_date", "pending", "verify_pending", "out_top30", "out_full", "out_verify", "report_md"):
            v = _safe_get(pr, k, None)
            if v not in (None, "", False):
                print(f"[premium][predict] {k}: {v}")

        # 旧字段兼容（你之前的 rank_csv/rank_md）
        rank_csv = _safe_get(pr, "rank_csv", "")
        rank_md = _safe_get(pr, "rank_md", "")
        if rank_csv:
            print(f"[premium][predict] rank_csv: {rank_csv}")
        if rank_md:
            print(f"[premium][predict] rank_md: {rank_md}")

        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
