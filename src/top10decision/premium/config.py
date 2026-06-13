#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — 配置（Config）

本文件职责：
- 定义 Premium 子系统的路径、文件命名、TopN/TopK、horizon 等核心参数
- 提供“环境变量覆盖”能力，保证 GitHub Actions / 本地运行一致
- 提供统一的输出路径生成函数（csv/md/models/learning/_last_run.txt）
- 引入 Market Truth Layer（行情事实层）配置与缓存路径

重要说明（向后兼容 + 新契约落地）：
- 旧口径（训练/模型/rank）字段：保留不删（避免历史模块引用炸裂）
- 新口径（Premium V2 主线）字段：补齐分位点/冷启动参数/校准落库路径
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


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


def _env_quantiles(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    读取形如 "0.05,0.25,0.5,0.75,0.95" 的分位点配置。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        qs = tuple(float(p) for p in parts)
        # 基本校验：0<q<1 且递增
        if any((q <= 0.0 or q >= 1.0) for q in qs):
            return default
        if any(qs[i] >= qs[i + 1] for i in range(len(qs) - 1)):
            return default
        return qs
    except Exception:
        return default


@dataclass(frozen=True)
class PremiumConfig:
    """
    Premium 子系统统一配置。

    =========================
    ✅ 新口径（Premium V2 主线）
    =========================
    - 主输入（候选=全量）：data/pred/pred_source_latest.csv
    - horizon = 2 个交易日（T -> T+2）
    - 默认 N=30（Top30 展示），同时输出 full
    - 行情真值缓存：data/market/daily_{YYYYMMDD}.csv（Market Truth Layer 负责）
    - 报告：docs/reports/premium_{T}.md + premium_latest.md
    - 分布预测：Close[T+2] 分位数（P05/P25/P50/P75/P95）

    =========================
    ♻️ 旧口径（训练/模型/rank）
    =========================
    - topk/train_window_days/min_train_days/up_threshold 等：保留
    - rank_csv_tpl/rank_md_tpl/models/learning：保留
    """

    # ===== 新口径（V2 锁死）=====
    top_n: int = 30
    horizon_trade_days: int = 2
    pred_source_latest: str = "data/pred/pred_source_latest.csv"
    decision_glob: str = "outputs/decision/*.csv"  # 仅 merge 标签（不得过滤）
    reports_dir: str = "docs/reports"

    # V2 分布输出：分位点（锁死默认，可 env 覆盖）
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

    # V2 冷启动参数（保证“无模型也能跑”）
    base_sigma: float = 0.05
    score_scale: float = 0.01

    # ===== 旧口径（保留）=====
    topk: int = 10
    train_window_days: int = 60
    min_train_days: int = 20
    up_threshold: float = 0.0

    # 旧口径残留（不确定引用方，先保留）
    close_input_glob: str = "data/market/daily_*.csv"

    # ===== Market Truth Layer（行情事实层）=====
    market_cache_dir: str = "data/market"
    market_daily_tpl: str = "daily_{trade_date}.csv"
    top3_raw_base_url: str = "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main"
    top3_local_dir: str = "a-share-top3-data"
    market_fetch_mode: str = "cache_first"  # cache_only / cache_first

    # ===== 输出根目录（新旧共用）=====
    out_dir: str = "outputs/premium"

    # ===== 模型版本（追溯字段）=====
    model_version: str = "premium_v4_a_share_d_t_t1"

    # ===== 旧 rank/learning/models 命名（保留）=====
    rank_csv_tpl: str = "premium_rank_{trade_date}.csv"
    rank_md_tpl: str = "premium_rank_{trade_date}.md"
    eval_history_csv: str = "premium_eval_history.csv"
    lr_model_name: str = "premium_lr.joblib"
    lgbm_model_name: str = "premium_lgbm.joblib"
    last_run_file: str = "_last_run.txt"

    @staticmethod
    def load() -> "PremiumConfig":
        """
        从默认值加载，并允许环境变量覆盖（便于 Actions/手动运行）。
        """
        cfg = PremiumConfig(
            # ===== 新口径（V2）=====
            top_n=_env_int("PREMIUM_TOP_N", 30),
            horizon_trade_days=_env_int("PREMIUM_HORIZON", 2),
            pred_source_latest=_env_str("PREMIUM_PRED_SOURCE_LATEST", "data/pred/pred_source_latest.csv"),
            decision_glob=_env_str("PREMIUM_DECISION_GLOB", "outputs/decision/*.csv"),
            reports_dir=_env_str("PREMIUM_REPORTS_DIR", "docs/reports"),

            quantiles=_env_quantiles("PREMIUM_QUANTILES", (0.05, 0.25, 0.50, 0.75, 0.95)),
            base_sigma=_env_float("PREMIUM_BASE_SIGMA", 0.05),
            score_scale=_env_float("PREMIUM_SCORE_SCALE", 0.01),

            # ===== 旧口径（保留）=====
            topk=_env_int("PREMIUM_TOPK", 10),
            train_window_days=_env_int("PREMIUM_TRAIN_WINDOW_DAYS", 60),
            min_train_days=_env_int("PREMIUM_MIN_TRAIN_DAYS", 20),
            up_threshold=_env_float("PREMIUM_UP_THRESHOLD", 0.0),

            close_input_glob=_env_str("PREMIUM_CLOSE_INPUT_GLOB", "data/market/daily_*.csv"),

            # Market Truth Layer
            market_cache_dir=_env_str("PREMIUM_MARKET_CACHE_DIR", "data/market"),
            market_daily_tpl=_env_str("PREMIUM_MARKET_DAILY_TPL", "daily_{trade_date}.csv"),
            top3_raw_base_url=_env_str(
                "PREMIUM_TOP3_RAW_BASE_URL",
                "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main",
            ),
            top3_local_dir=_env_str("PREMIUM_TOP3_LOCAL_DIR", "a-share-top3-data"),
            market_fetch_mode=_env_str("PREMIUM_MARKET_FETCH_MODE", "cache_first"),

            out_dir=_env_str("PREMIUM_OUT_DIR", "outputs/premium"),
            model_version=_env_str("PREMIUM_MODEL_VERSION", "premium_v4_a_share_d_t_t1"),
        )
        return cfg

    # ===== 路径生成（统一入口）=====

    def repo_root(self) -> Path:
        return _find_repo_root(Path(__file__).resolve()).resolve()

    def out_root(self) -> Path:
        return (self.repo_root() / self.out_dir).resolve()

    # 旧目录（保留）
    def out_rank_dir(self) -> Path:
        return (self.out_root() / "rank").resolve()

    def out_models_dir(self) -> Path:
        return (self.out_root() / "models").resolve()

    def out_learning_dir(self) -> Path:
        return (self.out_root() / "learning").resolve()

    def out_last_run_path(self) -> Path:
        return (self.out_root() / self.last_run_file).resolve()

    # ===== 新口径输出（V2）=====
    def out_top30_csv(self, trade_date: str) -> Path:
        return (self.out_root() / f"premium_top30_{trade_date}.csv").resolve()

    def out_top10_csv(self, trade_date: str) -> Path:
        return (self.out_root() / f"premium_top10_{trade_date}.csv").resolve()

    def out_top20_csv(self, trade_date: str) -> Path:
        return (self.out_root() / f"premium_top20_{trade_date}.csv").resolve()

    def out_full_csv(self, trade_date: str) -> Path:
        return (self.out_root() / f"premium_full_{trade_date}.csv").resolve()

    def out_verify_csv(self, trade_date: str) -> Path:
        return (self.out_root() / f"premium_verify_{trade_date}.csv").resolve()

    def reports_root(self) -> Path:
        return (self.repo_root() / self.reports_dir).resolve()

    def report_md_path(self, trade_date: str) -> Path:
        return (self.reports_root() / f"premium_{trade_date}.md").resolve()

    def report_latest_md_path(self) -> Path:
        return (self.reports_root() / "premium_latest.md").resolve()

    def report_html_path(self, trade_date: str) -> Path:
        return (self.reports_root() / f"premium_{trade_date}.html").resolve()

    def report_latest_html_path(self) -> Path:
        return (self.reports_root() / "premium_latest.html").resolve()

    # V2 校准历史落库（新增）
    def calibration_history_path(self) -> Path:
        return (self.out_learning_dir() / "premium_calibration_history.csv").resolve()

    # ===== 旧 rank 文件（保留）=====
    def rank_csv_path(self, trade_date: str) -> Path:
        return (self.out_rank_dir() / self.rank_csv_tpl.format(trade_date=trade_date)).resolve()

    def rank_md_path(self, trade_date: str) -> Path:
        return (self.out_rank_dir() / self.rank_md_tpl.format(trade_date=trade_date)).resolve()

    # ===== 旧 learning/models（保留）=====
    def eval_history_path(self) -> Path:
        return (self.out_learning_dir() / self.eval_history_csv).resolve()

    def lr_model_path(self) -> Path:
        return (self.out_models_dir() / self.lr_model_name).resolve()

    def lgbm_model_path(self) -> Path:
        return (self.out_models_dir() / self.lgbm_model_name).resolve()

    # ===== Market Truth Layer paths =====
    def market_cache_root(self) -> Path:
        return (self.repo_root() / self.market_cache_dir).resolve()

    def market_daily_cache_path(self, trade_date: str) -> Path:
        return (self.market_cache_root() / self.market_daily_tpl.format(trade_date=trade_date)).resolve()

    # ===== 新口径主输入 =====
    def pred_source_latest_path(self) -> Path:
        return (self.repo_root() / self.pred_source_latest).resolve()


__all__ = ["PremiumConfig"]
