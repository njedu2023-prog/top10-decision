#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top10decision.premium

Premium（溢价预测）子系统对外 API：
- train_models(): 训练 LR + LGBM（落盘模型 + eval_history）
- predict_latest(): 基于最新 decision 表生成 premium_rank CSV+MD
"""

from .config import PremiumConfig
from .train import TrainResult, train_models
from .predict import PredictResult, predict_latest
from .labels import LabelBuildMeta, build_premium_labels
from .features import FeatureBuildResult, build_features_from_decision_df

__all__ = [
    "PremiumConfig",
    "TrainResult",
    "PredictResult",
    "train_models",
    "predict_latest",
    "LabelBuildMeta",
    "build_premium_labels",
    "FeatureBuildResult",
    "build_features_from_decision_df",
]
