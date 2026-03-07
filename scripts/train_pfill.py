#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pfill.py

用途：
- 读取 data/market/pfill_trainset_{trade_date}.csv
- 训练 P_fill 二分类模型
- 输出：
    models/pfill_lr.joblib
    models/pfill_lgbm.joblib
    models/pfill_meta.json
    models/pfill_lr_{trade_date}.joblib
    models/pfill_lgbm_{trade_date}.joblib
    models/pfill_meta_{trade_date}.json

当前策略：
- 主模型 1：LogisticRegression
- 主模型 2：LightGBM（若环境缺失则降级为 HistGradientBoosting）
- 只训练 label_ready_fill=1 的成熟样本
- 时间切分优先；若单日样本不足，再退化为行切分
- 若切分后 train/valid 标签类别不足，则进入 small_sample_mode：
  使用全量样本训练，跳过稳定评估，但不断链

新增规则：
- 若某个 trade_date 的可训练样本只有单一标签（全 1 或全 0），
  则不报错退出，而是：
  1) 跳过训练
  2) 写出 pfill_meta / pfill_meta_{trade_date}.json
  3) 记录 skip_reason=single_label
  4) 不覆盖已有模型文件
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMClassifier = None  # type: ignore


# =========================================================
# 路径
# =========================================================
def detect_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def norm_ymd(x: object) -> str:
    s = str(x or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else s


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# =========================================================
# 读取与清洗
# =========================================================
def load_trainset(project_root: Path, trade_date: str) -> Tuple[pd.DataFrame, Path]:
    path = project_root / "data" / "market" / f"pfill_trainset_{trade_date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到训练样本文件：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"训练样本为空：{path}")
    if "y_fill" not in df.columns:
        raise ValueError("训练样本缺少 y_fill 列")
    return df, path


def prepare_train_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_fill"] = pd.to_numeric(out["y_fill"], errors="coerce").fillna(0).astype(int)

    if "label_ready_fill" in out.columns:
        out["label_ready_fill"] = (
            pd.to_numeric(out["label_ready_fill"], errors="coerce").fillna(0).astype(int)
        )
        out = out[out["label_ready_fill"] == 1].copy()

    if out.empty:
        raise ValueError("过滤 label_ready_fill=1 后无可训练样本")

    return out


def get_label_set(df: pd.DataFrame) -> List[int]:
    if "y_fill" not in df.columns or df.empty:
        return []
    uniq = sorted(pd.to_numeric(df["y_fill"], errors="coerce").dropna().astype(int).unique().tolist())
    return uniq


# =========================================================
# 切分
# =========================================================
def split_train_valid(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, bool]:
    """
    返回：
    - train_df
    - valid_df（small_sample_mode 时为 None）
    - split_mode
    - small_sample_mode

    优先按 trade_date 时间切分：
    - 若至少有 3 个不同 trade_date：最后一个日期做 valid，其余做 train
    - 否则：按行切分 80/20
    - 若切分后 train/valid 任一边标签类别不足，则进入 small_sample_mode：
      使用全量样本训练，跳过验证评估
    """
    if "trade_date" in df.columns:
        dates = sorted({norm_ymd(x) for x in df["trade_date"].dropna().tolist() if norm_ymd(x)})
        if len(dates) >= 3:
            valid_date = dates[-1]
            train_df = df[df["trade_date"].map(norm_ymd) < valid_date].copy()
            valid_df = df[df["trade_date"].map(norm_ymd) == valid_date].copy()
            if len(train_df) > 0 and len(valid_df) > 0:
                if train_df["y_fill"].nunique() >= 2 and valid_df["y_fill"].nunique() >= 2:
                    return train_df, valid_df, f"time_holdout:{valid_date}", False

    n = len(df)
    cut = max(int(n * 0.8), 1)
    train_df = df.iloc[:cut].copy()
    valid_df = df.iloc[cut:].copy()

    if len(valid_df) == 0:
        valid_df = df.iloc[-max(1, min(20, n)) :].copy()
        train_df = df.iloc[: max(1, n - len(valid_df))].copy()

    if len(train_df) > 0 and len(valid_df) > 0:
        if train_df["y_fill"].nunique() >= 2 and valid_df["y_fill"].nunique() >= 2:
            return train_df, valid_df, "row_holdout:80_20", False

    # 小样本容错：全量训练，不做验证评估
    return df.copy(), None, "small_sample_full_train", True


# =========================================================
# 特征选择
# =========================================================
LEAKAGE_COLS = {
    "y_fill",
    "fill_label_quality",
    "entry_price_proxy_t1",
    "entry_price_proxy_mode",
    "exec_date",
    "target_date",
    "sample_maturity",
    "label_ready_fill",
    "label_ready_ret",
    "open_t1",
    "high_t1",
    "low_t1",
    "close_t1",
    "up_limit_t1",
    "down_limit_t1",
    "limit_type_t1",
    "open_times_t1",
    "break_open_times_t1",
    "first_seal_time_t1",
    "last_seal_time_t1",
    "seal_amount_t1",
    "is_suspended_t1",
    "dataset_split",
    "label_version",
    "buy_window_start",
    "buy_window_end",
}

ID_COLS = {
    "trade_date",
    "ts_code",
    "name",
}

NON_FEATURE_PREFIXES = ("Unnamed:",)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in LEAKAGE_COLS or c in ID_COLS:
            continue
        if any(str(c).startswith(p) for p in NON_FEATURE_PREFIXES):
            continue
        cols.append(c)
    if not cols:
        raise ValueError("未找到可训练特征列")
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

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=None,
    )

    return Pipeline(
        [
            ("pre", pre),
            ("clf", clf),
        ]
    )


def build_gbm_model() -> object:
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="binary",
            random_state=42,
            class_weight="balanced",
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300,
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
        model.fit(x_train, train_df["y_fill"].astype(int).values, sample_weight=sample_weight)
    else:
        model.fit(x_train, train_df["y_fill"].astype(int).values)
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
    x_valid = encoded_all.iloc[len(train_df) :].copy()

    model = build_gbm_model()
    sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
    if sample_weight is not None:
        model.fit(x_train, train_df["y_fill"].astype(int).values, sample_weight=sample_weight)
    else:
        model.fit(x_train, train_df["y_fill"].astype(int).values)

    return model, x_valid


# =========================================================
# 评估
# =========================================================
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        return float(log_loss(y_true, y_prob))
    except Exception:
        return None


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return None


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "auc": safe_auc(y_true, y_prob),
        "logloss": safe_logloss(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
    }


def empty_metrics(reason: str) -> Dict[str, Optional[float] | str]:
    return {
        "auc": None,
        "logloss": None,
        "brier": None,
        "reason": reason,
    }


# =========================================================
# meta / skip
# =========================================================
def build_skip_meta(
    trade_date: str,
    trainset_path: Path,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    skip_reason: str,
    label_set: List[int],
) -> Dict[str, object]:
    sample_maturity_distribution: Dict[str, int] = {}
    if "sample_maturity" in df.columns:
        sample_maturity_distribution = {
            str(k): int(v)
            for k, v in df["sample_maturity"].astype(str).value_counts(dropna=False).to_dict().items()
        }

    return {
        "train_time_utc": utc_now_iso(),
        "trade_date": trade_date,
        "trainset_path": str(trainset_path),
        "status": "skipped",
        "skip_reason": skip_reason,
        "label_set": label_set,
        "split_mode": "skipped",
        "small_sample_mode": False,
        "model_paths": {
            "lr": "",
            "lgbm": "",
            "lr_dated": "",
            "lgbm_dated": "",
        },
        "model_backend": {
            "lr": "sklearn.LogisticRegression",
            "lgbm": "lightgbm.LGBMClassifier" if HAS_LGBM else "sklearn.HistGradientBoostingClassifier",
        },
        "rows": {
            "raw_all": int(len(raw_df)),
            "ready_only": int(len(df)),
            "train": 0,
            "valid": 0,
        },
        "fill_truth_ready_only": True,
        "sample_maturity_distribution": sample_maturity_distribution,
        "label_distribution": {
            "ready_all_pos": int((df["y_fill"] == 1).sum()) if "y_fill" in df.columns else 0,
            "ready_all_neg": int((df["y_fill"] == 0).sum()) if "y_fill" in df.columns else 0,
            "train_pos": 0,
            "train_neg": 0,
            "valid_pos": 0,
            "valid_neg": 0,
        },
        "features": {
            "n_total": 0,
            "n_numeric": 0,
            "n_categorical": 0,
            "feature_cols": [],
        },
        "missing_ratio": {},
        "metrics_valid": {
            "lr": empty_metrics(skip_reason),
            "lgbm": empty_metrics(skip_reason),
        },
        "notes": [
            "本次未训练模型。",
            "原因：单一标签或不满足训练条件时，跳过训练但不断链。",
            "不会覆盖已有 latest 模型文件。",
        ],
    }


def write_meta_files(project_root: Path, trade_date: str, meta: Dict[str, object]) -> Tuple[Path, Path]:
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = models_dir / "pfill_meta.json"
    meta_dated_path = models_dir / f"pfill_meta_{trade_date}.json"
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    meta_path.write_text(meta_text, encoding="utf-8")
    meta_dated_path.write_text(meta_text, encoding="utf-8")
    return meta_path, meta_dated_path


# =========================================================
# 主流程
# =========================================================
def train_one_trade_date(project_root: Path, trade_date: str) -> Dict[str, object]:
    raw_df, trainset_path = load_trainset(project_root, trade_date)
    df = prepare_train_df(raw_df)

    label_set = get_label_set(df)
    if len(label_set) < 2:
        meta = build_skip_meta(
            trade_date=trade_date,
            trainset_path=trainset_path,
            raw_df=raw_df,
            df=df,
            skip_reason="single_label_skip_train",
            label_set=label_set,
        )
        meta_path, meta_dated_path = write_meta_files(project_root, trade_date, meta)
        print(f"[train_pfill] trade_date={trade_date}")
        print(f"[train_pfill] trainset={trainset_path}")
        print(f"[train_pfill] skipped=True")
        print(f"[train_pfill] skip_reason=single_label_skip_train")
        print(f"[train_pfill] label_set={label_set}")
        print(f"[train_pfill] out_meta={meta_path}")
        print(f"[train_pfill] out_meta_dated={meta_dated_path}")
        return meta

    train_df, valid_df, split_mode, small_sample_mode = split_train_valid(df)

    feature_cols = select_feature_columns(df)
    num_cols, cat_cols = split_feature_types(df, feature_cols)

    # LR
    lr_pipe = build_lr_pipeline(num_cols, cat_cols)
    lr_sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
    lr_pipe.fit(
        train_df[feature_cols],
        train_df["y_fill"].values,
        clf__sample_weight=lr_sample_weight,
    )

    if valid_df is not None:
        lr_valid_prob = lr_pipe.predict_proba(valid_df[feature_cols])[:, 1]
        lr_metrics = evaluate_probs(valid_df["y_fill"].values, lr_valid_prob)
    else:
        lr_metrics = empty_metrics("small_sample_mode_skip_valid")

    # GBM / LGBM
    if valid_df is not None:
        gbm_model, x_valid_gbm = fit_gbm_with_valid(train_df, valid_df, feature_cols)
        if hasattr(gbm_model, "predict_proba"):
            gbm_valid_prob = gbm_model.predict_proba(x_valid_gbm)[:, 1]
        else:
            raw = gbm_model.predict(x_valid_gbm)
            gbm_valid_prob = np.asarray(raw, dtype=float)
        gbm_metrics = evaluate_probs(valid_df["y_fill"].values, gbm_valid_prob)
    else:
        gbm_model = fit_gbm_train_only(train_df, feature_cols)
        gbm_metrics = empty_metrics("small_sample_mode_skip_valid")

    # 输出
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    lr_path = models_dir / "pfill_lr.joblib"
    gbm_path = models_dir / "pfill_lgbm.joblib"
    meta_path = models_dir / "pfill_meta.json"

    lr_dated_path = models_dir / f"pfill_lr_{trade_date}.joblib"
    gbm_dated_path = models_dir / f"pfill_lgbm_{trade_date}.joblib"
    meta_dated_path = models_dir / f"pfill_meta_{trade_date}.json"

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

    meta: Dict[str, object] = {
        "train_time_utc": utc_now_iso(),
        "trade_date": trade_date,
        "trainset_path": str(trainset_path),
        "status": "trained",
        "skip_reason": "",
        "label_set": label_set,
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
            "lr": "sklearn.LogisticRegression",
            "lgbm": "lightgbm.LGBMClassifier" if HAS_LGBM else "sklearn.HistGradientBoostingClassifier",
        },
        "rows": {
            "raw_all": int(len(raw_df)),
            "ready_only": int(len(df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)) if valid_df is not None else 0,
        },
        "fill_truth_ready_only": True,
        "sample_maturity_distribution": sample_maturity_distribution,
        "label_distribution": {
            "ready_all_pos": int((df["y_fill"] == 1).sum()),
            "ready_all_neg": int((df["y_fill"] == 0).sum()),
            "train_pos": int((train_df["y_fill"] == 1).sum()),
            "train_neg": int((train_df["y_fill"] == 0).sum()),
            "valid_pos": int((valid_df["y_fill"] == 1).sum()) if valid_df is not None else 0,
            "valid_neg": int((valid_df["y_fill"] == 0).sum()) if valid_df is not None else 0,
        },
        "features": {
            "n_total": int(len(feature_cols)),
            "n_numeric": int(len(num_cols)),
            "n_categorical": int(len(cat_cols)),
            "feature_cols": feature_cols,
        },
        "missing_ratio": {c: round(float(df[c].isna().mean()), 6) for c in feature_cols},
        "metrics_valid": {
            "lr": lr_metrics,
            "lgbm": gbm_metrics,
        },
        "notes": [
            "仅训练 label_ready_fill=1 的成熟样本。",
            "采用时间切分优先，样本不足时退化为行切分。",
            "若切分后标签类别不足，则进入 small_sample_mode：全量训练，跳过 valid 评估。",
            "若整天样本只有单一标签，则跳过训练并写出 skip meta，不覆盖旧模型。",
            "泄露列（T+1 真值列/标签列/成熟度列）已从特征中剔除。",
            "同时输出 latest 与 dated 模型文件，便于追溯。",
        ],
    }

    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    meta_path.write_text(meta_text, encoding="utf-8")
    meta_dated_path.write_text(meta_text, encoding="utf-8")

    print(f"[train_pfill] trade_date={trade_date}")
    print(f"[train_pfill] trainset={trainset_path}")
    print(f"[train_pfill] split_mode={split_mode}")
    print(f"[train_pfill] small_sample_mode={small_sample_mode}")
    print(
        f"[train_pfill] rows_raw={len(raw_df)} ready={len(df)} "
        f"train={len(train_df)} valid={len(valid_df) if valid_df is not None else 0}"
    )
    print(f"[train_pfill] features_total={len(feature_cols)}")
    print(f"[train_pfill] lr_auc={lr_metrics.get('auc')} gbm_auc={gbm_metrics.get('auc')}")
    print(f"[train_pfill] out_lr={lr_path}")
    print(f"[train_pfill] out_lgbm={gbm_path}")
    print(f"[train_pfill] out_meta={meta_path}")

    return meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="训练 P_fill 学习模型")
    ap.add_argument("--trade-date", required=True, help="训练样本交易日 YYYYMMDD")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    trade_date = norm_ymd(args.trade_date)
    if len(trade_date) != 8:
        raise ValueError("--trade-date 必须是 YYYYMMDD")

    project_root = detect_project_root()
    train_one_trade_date(project_root, trade_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
