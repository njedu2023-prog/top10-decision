#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Premium 子系统 — Logistic Regression 模型（分类头）

目标：
- 训练一个二分类模型，输出：
    pred_up_prob = P(real_premium_ret > threshold)

特点：
- P0 版本必须稳定：样本少/类别单一也不能炸
- 支持 save/load（joblib）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class LRModelBundle:
    """
    一个可落盘的 LR 模型包：
    - model: LogisticRegression 或 None（若降级成常数概率）
    - pos_rate: 训练集正类比例（用于降级/校准）
    - threshold: y 的阈值语义（默认 0.0，表示 real_premium_ret > 0）
    - feature_cols: 训练时使用的列顺序（推理时必须对齐）
    """
    model: Optional[LogisticRegression]
    pos_rate: float
    threshold: float
    feature_cols: list

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X2 = _align_X(X, self.feature_cols)
        if self.model is None:
            # 常数概率输出
            return np.full(shape=(len(X2),), fill_value=float(self.pos_rate), dtype=float)
        proba = self.model.predict_proba(X2.values)[:, 1]
        return proba.astype(float)


def _align_X(X: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    对齐推理输入的列顺序；缺失列补 0；多余列忽略。
    """
    X2 = X.copy()
    for c in feature_cols:
        if c not in X2.columns:
            X2[c] = 0.0
    X2 = X2[feature_cols]
    # 保证 float
    return X2.astype(float)


def fit_lr_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.0,
    feature_cols: Optional[list] = None,
) -> LRModelBundle:
    """
    训练 LR 分类模型。

    - X: 特征 DataFrame（数值）
    - y: 二分类标签（0/1）
    - threshold: 仅用于记录语义
    """
    if feature_cols is None:
        feature_cols = list(X.columns)

    # 安全对齐
    X2 = _align_X(X, feature_cols)

    y = np.asarray(y).astype(int)
    if len(y) == 0:
        return LRModelBundle(model=None, pos_rate=0.5, threshold=threshold, feature_cols=feature_cols)

    pos_rate = float(np.mean(y == 1))
    neg_rate = float(np.mean(y == 0))

    # 类别单一：无法训练 LR，降级常数概率
    if pos_rate < 1e-9 or neg_rate < 1e-9:
        return LRModelBundle(model=None, pos_rate=pos_rate, threshold=threshold, feature_cols=feature_cols)

    # 正常训练
    # class_weight='balanced'：样本少时更稳一些
    model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
        random_state=42,
    )
    model.fit(X2.values, y)

    return LRModelBundle(model=model, pos_rate=pos_rate, threshold=threshold, feature_cols=feature_cols)


def save_lr(bundle: LRModelBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_lr(path: str) -> LRModelBundle:
    obj = joblib.load(path)
    if not isinstance(obj, LRModelBundle):
        raise TypeError(f"[premium.model_lr] 加载对象类型不对：{type(obj)}")
    return obj


def build_y_from_real_ret(real_ret: pd.Series, threshold: float = 0.0) -> np.ndarray:
    """
    将 real_premium_ret 转为二分类标签：
    y = 1 if real_ret > threshold else 0
    """
    rr = pd.to_numeric(real_ret, errors="coerce")
    y = (rr > float(threshold)).astype(int).fillna(0).values
    return y


__all__ = [
    "LRModelBundle",
    "fit_lr_classifier",
    "save_lr",
    "load_lr",
    "build_y_from_real_ret",
]
