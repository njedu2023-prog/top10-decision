#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — LightGBM 模型（回归头）

目标：
- 训练一个回归模型，输出：
    pred_ret_mean = E[real_premium_ret]

设计原则：
- P0 必须稳定：样本很少/标签极端/特征缺失也不能炸
- 支持 save/load（joblib）
- 固定 feature_cols，推理时严格对齐
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


@dataclass
class LGBMRegBundle:
    """
    可落盘的 LGBM 回归模型包：
    - model: lightgbm.Booster 或 None（降级常数输出）
    - y_mean: 训练集标签均值（用于降级/兜底）
    - feature_cols: 训练时使用的列顺序（推理时必须对齐）
    """
    model: Optional[object]
    y_mean: float
    feature_cols: list

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X2 = _align_X(X, self.feature_cols)
        if self.model is None:
            return np.full(shape=(len(X2),), fill_value=float(self.y_mean), dtype=float)
        # lightgbm.Booster.predict
        yhat = self.model.predict(X2.values)
        return np.asarray(yhat, dtype=float)


def _align_X(X: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X2 = X.copy()
    for c in feature_cols:
        if c not in X2.columns:
            X2[c] = 0.0
    X2 = X2[feature_cols]
    return X2.astype(float)


def fit_lgbm_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Optional[list] = None,
    min_samples: int = 30,
) -> LGBMRegBundle:
    """
    训练 LGBM 回归模型。

    - X: 特征 DataFrame（数值）
    - y: real_premium_ret（连续值）
    - min_samples: 样本少于该阈值时，直接降级为均值预测（更稳）
    """
    if feature_cols is None:
        feature_cols = list(X.columns)

    X2 = _align_X(X, feature_cols)

    yy = pd.to_numeric(y, errors="coerce").astype(float)
    yy = yy.replace([np.inf, -np.inf], np.nan)
    valid = yy.notna()

    if valid.sum() == 0:
        return LGBMRegBundle(model=None, y_mean=0.0, feature_cols=feature_cols)

    y_train = yy[valid]
    X_train = X2.loc[valid].astype(float)

    y_mean = float(y_train.mean())

    # 1) lightgbm 不可用 -> 只能降级
    if lgb is None:
        return LGBMRegBundle(model=None, y_mean=y_mean, feature_cols=feature_cols)

    # 2) 样本太少 -> 降级（避免过拟合 + 训练不稳定）
    if int(len(y_train)) < int(min_samples):
        return LGBMRegBundle(model=None, y_mean=y_mean, feature_cols=feature_cols)

    # 3) 正常训练
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train.values, label=y_train.values)
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=300,
        valid_sets=[dtrain],
        valid_names=["train"],
        verbose_eval=False,
    )

    return LGBMRegBundle(model=booster, y_mean=y_mean, feature_cols=feature_cols)


def save_lgbm(bundle: LGBMRegBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_lgbm(path: str) -> LGBMRegBundle:
    obj = joblib.load(path)
    if not isinstance(obj, LGBMRegBundle):
        raise TypeError(f"[premium.model_lgbm] 加载对象类型不对：{type(obj)}")
    return obj


__all__ = [
    "LGBMRegBundle",
    "fit_lgbm_regressor",
    "save_lgbm",
    "load_lgbm",
]
