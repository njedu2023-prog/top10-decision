#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_eret.py

用途：
- 基于成熟窗口读取多个 data/market/eret_trainset_{trade_date}.csv
- 训练 E_ret 连续收益回归模型
- 输出：
    models/eret_lr.joblib
    models/eret_lgbm.joblib
    models/eret_meta.json
    models/eret_lr_{anchor_trade_date}.joblib
    models/eret_lgbm_{anchor_trade_date}.joblib
    models/eret_meta_{anchor_trade_date}.json

本版关键修复：
- E_ret 训练特征合同必须与线上 decision 推理可获得字段一致。
- 严禁未来真值、标签字段、训练权重、审计字段、T+1/T+2 真值字段进入 feature_cols。
- 重点剔除本次事故暴露出的危险字段：
    close_t2.1
    open_t1
    sample_weight
    is_cold_start
    feature_coverage_score
    prior_strength_score
    prior_theme_boost
    prior_seal_amount
    prior_open_times
    prior_turnover_rate
- 目的：降低线上推理缺失率，避免 LR / ElasticNet 在大面积缺失特征上整体负向外推。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None  # type: ignore


# =========================================================
# 路径 / 基础工具
# =========================================================
def detect_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def norm_ymd(x: object) -> str:
    s = str(x or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else s


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# =========================================================
# 成熟窗口解析
# =========================================================
def load_maturity_table(project_root: Path, maturity_csv: str = "") -> Tuple[pd.DataFrame, Path]:
    path = Path(maturity_csv) if maturity_csv else (project_root / "data" / "market" / "sample_maturity_latest.csv")
    if not path.exists():
        raise FileNotFoundError(f"缺少样本成熟度表：{path}。请先运行 scripts/resolve_sample_maturity.py")

    df = safe_read_csv(path)
    if df.empty:
        raise ValueError(f"样本成熟度表为空：{path}")
    if "trade_date" not in df.columns:
        raise ValueError(f"样本成熟度表缺少 trade_date 列：{path}")
    if "ERET_READY" not in df.columns:
        raise ValueError(f"样本成熟度表缺少 ERET_READY 列：{path}")

    out = df.copy()
    out["trade_date"] = out["trade_date"].map(norm_ymd)
    out["ERET_READY"] = pd.to_numeric(out["ERET_READY"], errors="coerce").fillna(0).astype(int)
    return out, path


def resolve_matured_trade_dates_for_eret(
    maturity_df: pd.DataFrame,
    anchor_trade_date: str,
    window_size: int = 0,
) -> List[str]:
    out = maturity_df.copy()
    out = out[(out["trade_date"] != "") & (out["trade_date"] <= anchor_trade_date)].copy()
    out = out[out["ERET_READY"] == 1].copy()

    dates = sorted(set(out["trade_date"].tolist()))
    if window_size and window_size > 0:
        dates = dates[-window_size:]
    return dates


# =========================================================
# 读取与拼窗
# =========================================================
def load_one_trainset(project_root: Path, trade_date: str) -> Tuple[pd.DataFrame, Path]:
    path = project_root / "data" / "market" / f"eret_trainset_{trade_date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到训练样本文件：{path}")

    df = safe_read_csv(path)
    if df.empty:
        raise ValueError(f"训练样本为空：{path}")
    if "realized_ret_t1_to_t2" not in df.columns:
        raise ValueError(f"训练样本缺少 realized_ret_t1_to_t2 列：{path}")

    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].map(norm_ymd)
        filtered = df[df["trade_date"] == trade_date].copy()
        if not filtered.empty:
            df = filtered
    if "trade_date" not in df.columns:
        df["trade_date"] = trade_date
    else:
        df["trade_date"] = df["trade_date"].replace("", trade_date).fillna(trade_date)

    df["trainset_trade_date"] = trade_date
    return df, path


def load_window_trainsets(
    project_root: Path,
    matured_trade_dates: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    dfs: List[pd.DataFrame] = []
    loaded_dates: List[str] = []
    missing_dates: List[str] = []
    loaded_paths: List[str] = []

    for td in matured_trade_dates:
        try:
            df, path = load_one_trainset(project_root, td)
            dfs.append(df)
            loaded_dates.append(td)
            loaded_paths.append(str(path))
        except FileNotFoundError:
            missing_dates.append(td)

    if not dfs:
        raise FileNotFoundError("成熟窗口内没有任何可用的 eret_trainset_*.csv。")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    if out.empty:
        raise ValueError("成熟窗口训练样本拼接后为空")

    return out, loaded_dates, missing_dates, loaded_paths


def prepare_train_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "label_ready_ret" in out.columns:
        out["label_ready_ret"] = pd.to_numeric(out["label_ready_ret"], errors="coerce").fillna(0).astype(int)
        out = out[out["label_ready_ret"] == 1].copy()

    if "eret_sample_eligible" in out.columns:
        out["eret_sample_eligible"] = pd.to_numeric(out["eret_sample_eligible"], errors="coerce").fillna(0).astype(int)
        out = out[out["eret_sample_eligible"] == 1].copy()

    if "realized_ret_t1_to_t2" not in out.columns:
        raise ValueError("训练样本缺少 realized_ret_t1_to_t2 列")

    out["realized_ret_t1_to_t2"] = pd.to_numeric(out["realized_ret_t1_to_t2"], errors="coerce")
    out = out[out["realized_ret_t1_to_t2"].notna()].copy()

    if out.empty:
        raise ValueError("过滤 label_ready_ret=1 / eret_sample_eligible=1 / 非空目标后无可训练样本")

    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].map(norm_ymd)

    # sample_weight 只允许作为训练权重，不允许进入 feature_cols。
    if "sample_weight" in out.columns:
        out["sample_weight"] = pd.to_numeric(out["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.2)
    else:
        out["sample_weight"] = 1.0

    return out


# =========================================================
# 切分
# =========================================================
def split_train_valid(
    df: pd.DataFrame,
    min_train_rows: int = 24,
    min_valid_rows: int = 8,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, bool]:
    if "trade_date" in df.columns:
        dates = sorted({norm_ymd(x) for x in df["trade_date"].dropna().tolist() if norm_ymd(x)})
        if len(dates) >= 3:
            valid_date = dates[-1]
            train_df = df[df["trade_date"].map(norm_ymd) < valid_date].copy()
            valid_df = df[df["trade_date"].map(norm_ymd) == valid_date].copy()
            if len(train_df) >= min_train_rows and len(valid_df) >= min_valid_rows:
                return train_df, valid_df, f"time_holdout:{valid_date}", False

    n = len(df)
    cut = max(int(n * 0.8), 1)
    train_df = df.iloc[:cut].copy()
    valid_df = df.iloc[cut:].copy()

    if len(valid_df) == 0:
        valid_df = df.iloc[-max(1, min(20, n)):].copy()
        train_df = df.iloc[: max(1, n - len(valid_df))].copy()

    if len(train_df) >= min_train_rows and len(valid_df) >= min_valid_rows:
        return train_df, valid_df, "row_holdout:80_20", False

    return df.copy(), None, "small_sample_full_train", True


# =========================================================
# 特征选择
# =========================================================

# 未来真值 / 标签 / 交易执行真值 / 审计字段，绝不允许进入线上推理特征合同。
LEAKAGE_COLS = {
    # 目标与其等价列
    "realized_ret_t1_to_t2",
    "premium_ret_t1_to_t2",

    # T+2 未来真值
    "target_date",
    "exit_date",
    "exit_price_t2_close",
    "close_t2",
    "open_t2",
    "high_t2",
    "low_t2",
    "vol_t2",
    "amount_t2",
    "pct_chg_t2",

    # T+1 真值 / 执行真值：T 日收盘后线上不可稳定获得
    "exec_date",
    "entry_date",
    "open_t1",
    "high_t1",
    "low_t1",
    "close_t1",
    "vol_t1",
    "amount_t1",
    "pct_chg_t1",
    "entry_price_t1",
    "entry_price_proxy_t1",
    "entry_price_proxy_mode",

    # 标签审计 / 时间成熟 / 非特征标记
    "sample_maturity",
    "label_ready_fill",
    "label_ready_ret",
    "y_fill",
    "fill_label_quality",
    "eret_sample_eligible",
    "eret_label_quality",
    "dataset_split",
    "eret_truth_version",
    "return_holding_mode",

    # 训练权重 / 覆盖度 / 冷启动审计字段：可用于训练治理，不可作为线上推理特征
    "sample_weight",
    "is_cold_start",
    "feature_coverage_score",

    # 明显无学习意义的过程列
    "buy_window_start",
    "buy_window_end",
}

ID_COLS = {
    "trade_date",
    "verify_date",
    "target_trade_date",
    "signal_date",
    "ts_code",
    "name",
    "name_fs",
    "name_limit",
    "trainset_trade_date",
    "run_id",
    "run_attempt",
    "commit_sha",
    "generated_at_utc",
    "generated_at_bjt",
}

# 当前线上 decision 输入中并不稳定存在的 prior_* 字段。
# 这些字段如果留在 feature_cols，会导致线上大面积填补，从而让 LR 发生整体外推。
ONLINE_UNAVAILABLE_COLS = {
    "prior_strength_score",
    "prior_theme_boost",
    "prior_seal_amount",
    "prior_open_times",
    "prior_turnover_rate",
    "prior_volume_ratio",
    "prior_limit_up_strength",
    "prior_board_rank",
    "prior_board_limit_up_count",
    "prior_prob",
    "prior_probability",
}

NON_FEATURE_PREFIXES = ("Unnamed:",)
MISSING_FLAG_SUFFIX = "__is_missing"


def _strip_duplicate_suffix(col: str) -> str:
    """
    pandas merge 后可能产生 close_t2.1 / xxx.1。
    如果 base 字段是泄露字段，也必须剔除。
    """
    s = str(col).strip()
    return re.sub(r"\.\d+$", "", s)


def _is_forbidden_feature_col(col: object) -> bool:
    s = str(col).strip()
    base = _strip_duplicate_suffix(s)
    low = s.lower()
    base_low = base.lower()

    if not s:
        return True
    if any(s.startswith(p) for p in NON_FEATURE_PREFIXES):
        return True
    if s.endswith(MISSING_FLAG_SUFFIX):
        return True

    forbidden_exact = {c.lower() for c in (LEAKAGE_COLS | ID_COLS | ONLINE_UNAVAILABLE_COLS)}
    if low in forbidden_exact or base_low in forbidden_exact:
        return True

    # 兜底：任何 T+2 字段或 merge 出来的 T+2 重复列都不能进特征。
    if re.search(r"(^|_)t2(\.|_|$)", low) or re.search(r"(^|_)t2(\.|_|$)", base_low):
        return True

    # 兜底：T+1 真实行情/成交字段不能进 T 日收盘后的线上推理。
    t1_truth_prefixes = (
        "open_t1",
        "high_t1",
        "low_t1",
        "close_t1",
        "vol_t1",
        "amount_t1",
        "pct_chg_t1",
        "entry_price_t1",
        "entry_price_proxy_t1",
    )
    if low.startswith(t1_truth_prefixes) or base_low.startswith(t1_truth_prefixes):
        return True

    return False


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    filtered: List[str] = []

    for c in df.columns:
        if _is_forbidden_feature_col(c):
            filtered.append(str(c))
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            filtered.append(str(c))
            continue
        cols.append(str(c))

    if not cols:
        raise ValueError("未找到可训练特征列")

    print(f"[train_eret] feature_contract=online_safe_v1")
    print(f"[train_eret] selected_features={len(cols)} filtered_features={len(filtered)}")
    if filtered:
        print(f"[train_eret] filtered_feature_sample={'|'.join(filtered[:30])}")

    return cols


def split_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def detect_category_like_cols(feature_cols: List[str]) -> List[str]:
    hints = {"symbol", "board", "area", "market", "market_regime", "latest_change_reason", "limit_type", "prior_board"}
    return [c for c in feature_cols if c in hints]


# =========================================================
# 模型
# =========================================================
def build_lr_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    reg = ElasticNet(
        alpha=0.001,
        l1_ratio=0.2,
        max_iter=5000,
        random_state=42,
    )

    return Pipeline(
        [
            ("pre", pre),
            ("reg", reg),
        ]
    )


def build_gbm_model() -> object:
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="regression",
            random_state=42,
            verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
    )


def encode_gbm_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(-999.0)
        else:
            cat = x[c].astype(str)
            uniq = pd.Index(cat.unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            x[c] = cat.map(mapping).fillna(-1).astype(float)
    return x


def fit_gbm_train_only(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> object:
    x_train = encode_gbm_frame(train_df, feature_cols)
    model = build_gbm_model()
    sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
    if sample_weight is not None:
        model.fit(x_train, train_df["realized_ret_t1_to_t2"].astype(float).values, sample_weight=sample_weight)
    else:
        model.fit(x_train, train_df["realized_ret_t1_to_t2"].astype(float).values)
    return model


def fit_gbm_with_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[object, pd.DataFrame]:
    train_all = pd.concat([train_df[feature_cols], valid_df[feature_cols]], axis=0).copy()
    encoded_all = train_all.copy()

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(encoded_all[c]):
            encoded_all[c] = pd.to_numeric(encoded_all[c], errors="coerce").fillna(-999.0)
        else:
            cat = encoded_all[c].astype(str)
            uniq = pd.Index(cat.unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            encoded_all[c] = cat.map(mapping).fillna(-1).astype(float)

    x_train = encoded_all.iloc[: len(train_df)].copy()
    x_valid = encoded_all.iloc[len(train_df):].copy()

    model = build_gbm_model()
    sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
    if sample_weight is not None:
        model.fit(x_train, train_df["realized_ret_t1_to_t2"].astype(float).values, sample_weight=sample_weight)
    else:
        model.fit(x_train, train_df["realized_ret_t1_to_t2"].astype(float).values)

    return model, x_valid


# =========================================================
# 评估
# =========================================================
def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        if len(y_true) < 2:
            return None
        return float(r2_score(y_true, y_pred))
    except Exception:
        return None


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        return None


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return None


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        if len(y_true) < 2:
            return None
        yt = pd.Series(y_true)
        yp = pd.Series(y_pred)
        corr = yt.corr(yp)
        return None if pd.isna(corr) else float(corr)
    except Exception:
        return None


def safe_directional_acc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        if len(y_true) == 0:
            return None
        true_sign = np.sign(y_true)
        pred_sign = np.sign(y_pred)
        return float((true_sign == pred_sign).mean())
    except Exception:
        return None


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "mae": safe_mae(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "corr": safe_corr(y_true, y_pred),
        "directional_acc": safe_directional_acc(y_true, y_pred),
    }


def _metric_value(metrics: Dict[str, object], key: str, default: float) -> float:
    try:
        value = metrics.get(key)
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def choose_selected_model(metrics_valid: Dict[str, Dict[str, object]]) -> Tuple[str, Dict[str, object]]:
    """
    选择线上 E_ret 模型。

    E_ret 是小样本回归，LR 偶尔会被极端样本拖出荒唐量级。线上不能固定
    LR > LGBM，而应按验证集质量选择。主排序用 RMSE，辅以 MAE / 方向准确率；
    指标缺失时偏向 LGBM，因为树模型对非线性和缺失填充更稳。
    """
    candidates: list[tuple[tuple[float, float, float, int], str, Dict[str, object]]] = []
    for preference, kind in enumerate(("lgbm", "lr")):
        metrics = metrics_valid.get(kind, {}) or {}
        rmse = _metric_value(metrics, "rmse", float("inf"))
        mae = _metric_value(metrics, "mae", float("inf"))
        directional_acc = _metric_value(metrics, "directional_acc", -1.0)
        candidates.append(((rmse, mae, -directional_acc, preference), kind, metrics))

    candidates.sort(key=lambda x: x[0])
    selected_kind = candidates[0][1] if candidates else "lgbm"
    selected_metrics = candidates[0][2] if candidates else {}
    return selected_kind, {
        "selected_model": selected_kind,
        "selection_rule": "min_rmse_then_mae_then_directional_acc_prefer_lgbm",
        "selected_metrics": selected_metrics,
        "candidate_metrics": metrics_valid,
    }


def empty_metrics(reason: str) -> Dict[str, Optional[float] | str]:
    return {
        "mae": None,
        "rmse": None,
        "r2": None,
        "corr": None,
        "directional_acc": None,
        "reason": reason,
    }


# =========================================================
# meta / skip
# =========================================================
def build_skip_meta(
    anchor_trade_date: str,
    maturity_path: Path,
    loaded_trainset_paths: List[str],
    matured_trade_dates: List[str],
    loaded_trade_dates: List[str],
    missing_trade_dates: List[str],
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    skip_reason: str,
) -> Dict[str, object]:
    sample_maturity_distribution: Dict[str, int] = {}
    if "sample_maturity" in df.columns:
        sample_maturity_distribution = {
            str(k): int(v)
            for k, v in df["sample_maturity"].astype(str).value_counts(dropna=False).to_dict().items()
        }

    return {
        "train_time_utc": utc_now_iso(),
        "anchor_trade_date": anchor_trade_date,
        "status": "skipped",
        "skip_reason": skip_reason,
        "target_column": "realized_ret_t1_to_t2",
        "split_mode": "skipped",
        "small_sample_mode": False,
        "model_paths": {
            "lr": "",
            "lgbm": "",
            "lr_dated": "",
            "lgbm_dated": "",
        },
        "model_backend": {
            "lr": "sklearn.ElasticNet",
            "lgbm": "lightgbm.LGBMRegressor" if HAS_LGBM else "sklearn.HistGradientBoostingRegressor",
        },
        "window": {
            "maturity_csv": str(maturity_path),
            "matured_trade_dates": matured_trade_dates,
            "loaded_trade_dates": loaded_trade_dates,
            "missing_trade_dates": missing_trade_dates,
            "loaded_trainset_paths": loaded_trainset_paths,
            "n_matured_dates": int(len(matured_trade_dates)),
            "n_loaded_dates": int(len(loaded_trade_dates)),
            "window_start": loaded_trade_dates[0] if loaded_trade_dates else "",
            "window_end": loaded_trade_dates[-1] if loaded_trade_dates else "",
        },
        "rows": {
            "raw_all": int(len(raw_df)),
            "ready_only": int(len(df)),
            "train": 0,
            "valid": 0,
        },
        "eret_truth_ready_only": True,
        "sample_maturity_distribution": sample_maturity_distribution,
        "target_distribution": {
            "mean": float(df["realized_ret_t1_to_t2"].mean()) if "realized_ret_t1_to_t2" in df.columns and len(df) else None,
            "median": float(df["realized_ret_t1_to_t2"].median()) if "realized_ret_t1_to_t2" in df.columns and len(df) else None,
            "min": float(df["realized_ret_t1_to_t2"].min()) if "realized_ret_t1_to_t2" in df.columns and len(df) else None,
            "max": float(df["realized_ret_t1_to_t2"].max()) if "realized_ret_t1_to_t2" in df.columns and len(df) else None,
        },
        "features": {
            "n_total": 0,
            "n_numeric": 0,
            "n_categorical": 0,
            "feature_cols": [],
            "numeric_cols": [],
            "categorical_cols": [],
            "feature_contract_version": "online_safe_v1",
            "filtered_feature_cols_due_to_contract": [],
        },
        "missing_ratio": {},
        "metrics_valid": {
            "lr": empty_metrics(skip_reason),
            "lgbm": empty_metrics(skip_reason),
        },
        "notes": [
            "本次未训练模型。",
            "原因：成熟窗口可训练样本不足时，跳过训练但不断链。",
            "不会覆盖已有 latest 模型文件。",
            "本文件已升级为成熟窗口训练，不再限定单日训练。",
            "E_ret feature contract 已升级为 online_safe_v1。",
        ],
    }


def write_meta_files(project_root: Path, anchor_trade_date: str, meta: Dict[str, object]) -> Tuple[Path, Path]:
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = models_dir / "eret_meta.json"
    meta_dated_path = models_dir / f"eret_meta_{anchor_trade_date}.json"
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    meta_path.write_text(meta_text, encoding="utf-8")
    meta_dated_path.write_text(meta_text, encoding="utf-8")
    return meta_path, meta_dated_path


# =========================================================
# 主流程
# =========================================================
def train_window_as_of(
    project_root: Path,
    anchor_trade_date: str,
    maturity_csv: str = "",
    window_size: int = 0,
    min_train_rows: int = 24,
    min_valid_rows: int = 8,
) -> Dict[str, object]:
    maturity_df, maturity_path = load_maturity_table(project_root, maturity_csv=maturity_csv)
    matured_trade_dates = resolve_matured_trade_dates_for_eret(
        maturity_df=maturity_df,
        anchor_trade_date=anchor_trade_date,
        window_size=window_size,
    )
    if not matured_trade_dates:
        raise ValueError(f"在 sample_maturity 中找不到 trade_date <= {anchor_trade_date} 且 ERET_READY=1 的成熟样本日")

    raw_df, loaded_trade_dates, missing_trade_dates, loaded_trainset_paths = load_window_trainsets(
        project_root=project_root,
        matured_trade_dates=matured_trade_dates,
    )
    df = prepare_train_df(raw_df)

    if len(df) < min_train_rows:
        meta = build_skip_meta(
            anchor_trade_date=anchor_trade_date,
            maturity_path=maturity_path,
            loaded_trainset_paths=loaded_trainset_paths,
            matured_trade_dates=matured_trade_dates,
            loaded_trade_dates=loaded_trade_dates,
            missing_trade_dates=missing_trade_dates,
            raw_df=raw_df,
            df=df,
            skip_reason="insufficient_rows_skip_train",
        )
        meta_path, meta_dated_path = write_meta_files(project_root, anchor_trade_date, meta)

        print(f"[train_eret] anchor_trade_date={anchor_trade_date}")
        print(f"[train_eret] matured_trade_dates={len(matured_trade_dates)}")
        print(f"[train_eret] loaded_trade_dates={loaded_trade_dates}")
        print(f"[train_eret] missing_trade_dates={missing_trade_dates}")
        print(f"[train_eret] skipped=True")
        print(f"[train_eret] skip_reason=insufficient_rows_skip_train")
        print(f"[train_eret] out_meta={meta_path}")
        print(f"[train_eret] out_meta_dated={meta_dated_path}")
        return meta

    train_df, valid_df, split_mode, small_sample_mode = split_train_valid(
        df,
        min_train_rows=min_train_rows,
        min_valid_rows=min_valid_rows,
    )

    feature_cols = select_feature_columns(df)
    num_cols, cat_cols = split_feature_types(df, feature_cols)
    category_like_cols = detect_category_like_cols(feature_cols)

    filtered_feature_cols_due_to_contract = [
        str(c) for c in df.columns if _is_forbidden_feature_col(c)
    ]

    # LR / ElasticNet
    lr_pipe = build_lr_pipeline(num_cols, cat_cols)
    lr_sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
    lr_pipe.fit(
        train_df[feature_cols],
        train_df["realized_ret_t1_to_t2"].astype(float).values,
        reg__sample_weight=lr_sample_weight,
    )

    if valid_df is not None:
        lr_valid_pred = lr_pipe.predict(valid_df[feature_cols])
        lr_metrics = evaluate_regression(
            valid_df["realized_ret_t1_to_t2"].astype(float).values,
            np.asarray(lr_valid_pred, dtype=float),
        )
    else:
        lr_metrics = empty_metrics("small_sample_mode_skip_valid")

    # GBM / LGBM
    if valid_df is not None:
        gbm_model, x_valid_gbm = fit_gbm_with_valid(train_df, valid_df, feature_cols)
        gbm_valid_pred = gbm_model.predict(x_valid_gbm)
        gbm_metrics = evaluate_regression(
            valid_df["realized_ret_t1_to_t2"].astype(float).values,
            np.asarray(gbm_valid_pred, dtype=float),
        )
    else:
        gbm_model = fit_gbm_train_only(train_df, feature_cols)
        gbm_metrics = empty_metrics("small_sample_mode_skip_valid")

    metrics_valid = {
        "lr": lr_metrics,
        "lgbm": gbm_metrics,
    }
    selected_model, model_selection = choose_selected_model(metrics_valid)

    # 输出
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    lr_path = models_dir / "eret_lr.joblib"
    gbm_path = models_dir / "eret_lgbm.joblib"
    meta_path = models_dir / "eret_meta.json"

    lr_dated_path = models_dir / f"eret_lr_{anchor_trade_date}.joblib"
    gbm_dated_path = models_dir / f"eret_lgbm_{anchor_trade_date}.joblib"
    meta_dated_path = models_dir / f"eret_meta_{anchor_trade_date}.json"

    joblib.dump(lr_pipe, lr_path)
    joblib.dump(gbm_model, gbm_path)
    joblib.dump(lr_pipe, lr_dated_path)
    joblib.dump(gbm_model, gbm_dated_path)

    sample_maturity_distribution: Dict[str, int] = {}
    if "sample_maturity" in df.columns:
        sample_maturity_distribution = {
            str(k): int(v)
            for k, v in df["sample_maturity"].astype(str).value_counts(dropna=False).to_dict().items()
        }

    train_dates = sorted(set(train_df["trade_date"].map(norm_ymd).tolist())) if "trade_date" in train_df.columns else []
    valid_dates = sorted(set(valid_df["trade_date"].map(norm_ymd).tolist())) if valid_df is not None and "trade_date" in valid_df.columns else []

    missing_ratio = {c: round(float(df[c].isna().mean()), 6) for c in feature_cols}
    target_s = pd.to_numeric(df["realized_ret_t1_to_t2"], errors="coerce")

    meta: Dict[str, object] = {
        "train_time_utc": utc_now_iso(),
        "anchor_trade_date": anchor_trade_date,
        "status": "trained",
        "skip_reason": "",
        "target_column": "realized_ret_t1_to_t2",
        "split_mode": split_mode,
        "small_sample_mode": bool(small_sample_mode),
        "model_paths": {
            "lr": str(lr_path),
            "lgbm": str(gbm_path),
            "lr_dated": str(lr_dated_path),
            "lgbm_dated": str(gbm_dated_path),
        },
        "meta_path_dated": str(meta_dated_path),
        "model_backend": {
            "lr": "sklearn.ElasticNet",
            "lgbm": "lightgbm.LGBMRegressor" if HAS_LGBM else "sklearn.HistGradientBoostingRegressor",
        },
        "selected_model": selected_model,
        "model_selection": model_selection,
        "window": {
            "maturity_csv": str(maturity_path),
            "matured_trade_dates": matured_trade_dates,
            "loaded_trade_dates": loaded_trade_dates,
            "missing_trade_dates": missing_trade_dates,
            "loaded_trainset_paths": loaded_trainset_paths,
            "n_matured_dates": int(len(matured_trade_dates)),
            "n_loaded_dates": int(len(loaded_trade_dates)),
            "window_start": loaded_trade_dates[0] if loaded_trade_dates else "",
            "window_end": loaded_trade_dates[-1] if loaded_trade_dates else "",
            "train_dates": train_dates,
            "valid_dates": valid_dates,
        },
        "rows": {
            "raw_all": int(len(raw_df)),
            "ready_only": int(len(df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)) if valid_df is not None else 0,
        },
        "eret_truth_ready_only": True,
        "sample_maturity_distribution": sample_maturity_distribution,
        "target_distribution": {
            "mean": float(target_s.mean()) if target_s.notna().any() else None,
            "median": float(target_s.median()) if target_s.notna().any() else None,
            "min": float(target_s.min()) if target_s.notna().any() else None,
            "max": float(target_s.max()) if target_s.notna().any() else None,
            "positive_rate": float((target_s > 0).mean()) if target_s.notna().any() else None,
        },
        "features": {
            "feature_contract_version": "online_safe_v1",
            "n_total": int(len(feature_cols)),
            "n_numeric": int(len(num_cols)),
            "n_categorical": int(len(cat_cols)),
            "feature_cols": feature_cols,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "category_like_cols_detected": category_like_cols,
            "filtered_feature_cols_due_to_contract": filtered_feature_cols_due_to_contract,
            "filtered_feature_count_due_to_contract": int(len(filtered_feature_cols_due_to_contract)),
        },
        "missing_ratio": missing_ratio,
        "metrics_valid": metrics_valid,
        "notes": [
            "仅训练 label_ready_ret=1 且 eret_sample_eligible=1 的成熟样本。",
            "本文件已升级为成熟窗口训练，训练样本来自 sample_maturity 中 ERET_READY=1 的全部历史样本日。",
            "采用时间切分优先，样本不足时退化为行切分。",
            "若切分后样本太少，则进入 small_sample_mode：全量训练，跳过 valid 评估。",
            "若整个成熟窗口样本不足，则跳过训练并写出 skip meta，不覆盖旧模型。",
            "E_ret feature contract 已升级为 online_safe_v1。",
            "未来真值列、T+1/T+2 真值列、sample_weight、冷启动/覆盖度审计列、线上不稳定 prior_* 列已从特征中剔除。",
            "同时输出 latest 与 dated 模型文件，便于追溯。",
            "线上 E_ret 模型按验证 RMSE/MAE/方向准确率选择，不再固定 LR 优先。",
        ],
    }

    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    meta_path.write_text(meta_text, encoding="utf-8")
    meta_dated_path.write_text(meta_text, encoding="utf-8")

    print(f"[train_eret] anchor_trade_date={anchor_trade_date}")
    print(f"[train_eret] maturity_csv={maturity_path}")
    print(f"[train_eret] matured_trade_dates={len(matured_trade_dates)}")
    print(f"[train_eret] loaded_trade_dates={loaded_trade_dates}")
    print(f"[train_eret] missing_trade_dates={missing_trade_dates}")
    print(f"[train_eret] split_mode={split_mode}")
    print(f"[train_eret] small_sample_mode={small_sample_mode}")
    print(f"[train_eret] rows_raw={len(raw_df)} ready={len(df)} train={len(train_df)} valid={len(valid_df) if valid_df is not None else 0}")
    print(f"[train_eret] feature_contract=online_safe_v1")
    print(f"[train_eret] features_total={len(feature_cols)}")
    print(f"[train_eret] filtered_features_due_to_contract={len(filtered_feature_cols_due_to_contract)}")
    print(f"[train_eret] lr_rmse={lr_metrics.get('rmse')} gbm_rmse={gbm_metrics.get('rmse')}")
    print(f"[train_eret] selected_model={selected_model}")
    print(f"[train_eret] out_lr={lr_path}")
    print(f"[train_eret] out_lgbm={gbm_path}")
    print(f"[train_eret] out_meta={meta_path}")

    return meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="训练 E_ret 学习模型（成熟窗口版）")
    ap.add_argument("--trade-date", required=True, help="训练窗口锚点日 YYYYMMDD")
    ap.add_argument("--maturity-csv", default="", help="样本成熟度表路径；留空默认 data/market/sample_maturity_latest.csv")
    ap.add_argument("--window-size", type=int, default=0, help="训练窗口长度（按成熟 trade_date 个数截断）；0=使用 anchor 之前全部 ERET_READY 样本")
    ap.add_argument("--min-train-rows", type=int, default=24, help="最少训练样本数；不足则 skip，不覆盖旧模型")
    ap.add_argument("--min-valid-rows", type=int, default=8, help="最少验证样本数；不足则进入 small_sample_mode")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    anchor_trade_date = norm_ymd(args.trade_date)
    if len(anchor_trade_date) != 8:
        raise ValueError("--trade-date 必须是 YYYYMMDD")

    project_root = detect_project_root()
    train_window_as_of(
        project_root=project_root,
        anchor_trade_date=anchor_trade_date,
        maturity_csv=args.maturity_csv,
        window_size=args.window_size,
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
