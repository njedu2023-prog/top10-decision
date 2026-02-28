#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — 配置（Config）

本文件职责：
- 定义 Premium 子系统的路径、文件命名、训练窗口、TopK 等核心参数
- 提供“环境变量覆盖”能力，保证 GitHub Actions / 本地运行一致
- 提供统一的输出路径生成函数（csv/md/models/learning/_last_run.txt）

已确认口径（系统级冻结）：
- PremiumRet(2→3) 固定为：RealPremiumRet = Close[3] / Close[2] - 1
- Premium 输出表的 trade_date 语义固定为：第2日（2日预测表对应的交易日）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """
    在当前文件所在位置向上寻找仓库根目录（以 requirements.txt 或 .git 为锚点）。
    """
    if start is None:
        start = Path(__file__).resolve()
    cur = start
    for _ in range(12):
        if (cur / "requirements.txt").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    # 兜底：如果找不到，就用当前工作目录
    return Path.cwd().resolve()


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


@dataclass(frozen=True)
class PremiumConfig:
    """
    Premium 子系统统一配置。

    你可以用环境变量覆盖（推荐 Actions 用）：
    - PREMIUM_TOPK
    - PREMIUM_TRAIN_WINDOW_DAYS
    - PREMIUM_MIN_TRAIN_DAYS
    - PREMIUM_DECISION_INPUT_GLOB
    - PREMIUM_CLOSE_INPUT_GLOB
    - PREMIUM_OUT_DIR
    - PREMIUM_MODEL_VERSION
    """

    # ===== 业务参数 =====
    topk: int = 10
    train_window_days: int = 60        # 训练窗口（滚动回看天数）
    min_train_days: int = 20           # 最小可训练天数（不足则跳过训练，避免噪声）
    up_threshold: float = 0.0          # 分类标签阈值：RealPremiumRet > threshold 判为上涨

    # ===== 输入源（默认用 glob，避免写死文件名）=====
    # 说明：
    # - decision 输入：第2日预测表（由第1日生成），通常位于 outputs/decision/ 下
    # - close 输入：真实收盘价表（含 trade_date/ts_code/close），来源可后续落地
    decision_input_glob: str = "outputs/decision/*.csv"
    close_input_glob: str = "data/close/*.csv"

    # ===== 输出根目录 =====
    out_dir: str = "outputs/premium"

    # ===== 模型版本（便于回溯/对比）=====
    model_version: str = "premium_v0"

    # ===== 文件命名规范（按 2日 trade_date）=====
    # rank 产物
    rank_csv_tpl: str = "premium_rank_{trade_date}.csv"
    rank_md_tpl: str = "premium_rank_{trade_date}.md"
    # eval 落库
    eval_history_csv: str = "premium_eval_history.csv"
    # models
    lr_model_name: str = "premium_lr.joblib"
    lgbm_model_name: str = "premium_lgbm.joblib"
    # last run
    last_run_file: str = "_last_run.txt"

    @staticmethod
    def load() -> "PremiumConfig":
        """
        从默认值加载，并允许环境变量覆盖（便于 Actions/手动运行）。
        """
        cfg = PremiumConfig(
            topk=_env_int("PREMIUM_TOPK", 10),
            train_window_days=_env_int("PREMIUM_TRAIN_WINDOW_DAYS", 60),
            min_train_days=_env_int("PREMIUM_MIN_TRAIN_DAYS", 20),
            up_threshold=_env_float("PREMIUM_UP_THRESHOLD", 0.0),
            decision_input_glob=_env_str("PREMIUM_DECISION_INPUT_GLOB", "outputs/decision/*.csv"),
            close_input_glob=_env_str("PREMIUM_CLOSE_INPUT_GLOB", "data/close/*.csv"),
            out_dir=_env_str("PREMIUM_OUT_DIR", "outputs/premium"),
            model_version=_env_str("PREMIUM_MODEL_VERSION", "premium_v0"),
        )
        return cfg

    # ===== 路径生成（统一入口）=====

    def repo_root(self) -> Path:
        return _find_repo_root(Path(__file__).resolve()).resolve()

    def out_root(self) -> Path:
        return (self.repo_root() / self.out_dir).resolve()

    def out_rank_dir(self) -> Path:
        return (self.out_root() / "rank").resolve()

    def out_models_dir(self) -> Path:
        return (self.out_root() / "models").resolve()

    def out_learning_dir(self) -> Path:
        return (self.out_root() / "learning").resolve()

    def out_last_run_path(self) -> Path:
        return (self.out_root() / self.last_run_file).resolve()

    # rank files
    def rank_csv_path(self, trade_date: str) -> Path:
        return (self.out_rank_dir() / self.rank_csv_tpl.format(trade_date=trade_date)).resolve()

    def rank_md_path(self, trade_date: str) -> Path:
        return (self.out_rank_dir() / self.rank_md_tpl.format(trade_date=trade_date)).resolve()

    # learning files
    def eval_history_path(self) -> Path:
        return (self.out_learning_dir() / self.eval_history_csv).resolve()

    # model files
    def lr_model_path(self) -> Path:
        return (self.out_models_dir() / self.lr_model_name).resolve()

    def lgbm_model_path(self) -> Path:
        return (self.out_models_dir() / self.lgbm_model_name).resolve()


__all__ = ["PremiumConfig"]
