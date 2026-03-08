#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pfill_engine.py

定位：
- P_fill 在线推理 / 计算引擎
- 负责把“候选池 + 证据层”转换为可直接参与 EV 的 p_fill_pred
- 优先加载学习模型；若模型缺失或推理失败，则自动回退到规则模型
- 不写文件，不直接做排序，不处理 weights / reports

职责边界：
- 输入：已经按候选池裁剪后的 DataFrame（通常来自 ingest.build_model_input）
- 输出：附加 P_fill 推理结果后的 DataFrame
- 不负责：
  1) 跨仓库拉数
  2) 真值构建
  3) 训练
  4) EV 融合
  5) 落盘输出

当前版本：
- v2：在线推理严格对齐训练态 feature contract
- 若 models/pfill_lr.joblib 或 models/pfill_lgbm.joblib 存在，则优先走学习模型
- 学习模型输出做裁剪到 [0.02, 0.98]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from top10decision.models.fill_model import fill_model_rule

PRED_MIN = 0.02
PRED_MAX = 0.98


def _detect_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clip_prob_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(PRED_MIN).clip(lower=PRED_MIN, upper=PRED_MAX)


def _existing_model_path(root: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = root / "models" / name
        if p.exists():
            return p
    return None


def _existing_meta_path(root: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = root / "models" / name
        if p.exists():
            return p
    return None


@dataclass
class PFillModelBundle:
    model: Any
    model_kind: str
    model_path: str
    feature_mode: str
    meta_path: str
    feature_cols: list[str]


def _load_pfill_meta(meta_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_feature_cols(meta: Dict[str, Any]) -> list[str]:
    features = meta.get("features", {}) if isinstance(meta, dict) else {}
    cols = features.get("feature_cols", []) if isinstance(features, dict) else []
    if not isinstance(cols, list):
        return []

    out: list[str] = []
    for c in cols:
        if c is None:
            continue
        s = str(c).strip()
        if not s:
            continue
        out.append(s)
    return out


def _resolve_pfill_model(project_root: Optional[Path] = None) -> Tuple[Optional[PFillModelBundle], Dict[str, Any]]:
    root = project_root or _detect_project_root()

    lgbm_path = _existing_model_path(root, ["pfill_lgbm.joblib"])
    lr_path = _existing_model_path(root, ["pfill_lr.joblib"])
    meta_path = _existing_meta_path(root, ["pfill_meta.json"])

    chosen = lgbm_path or lr_path
    if chosen is None:
        return None, {
            "pfill_model_loaded": False,
            "pfill_model_kind": "",
            "pfill_model_path": "",
            "pfill_model_feature_mode": "",
            "pfill_model_meta_path": str(meta_path) if meta_path else "",
            "pfill_model_expected_n_features": 0,
            "pfill_model_degrade_reason": "model_missing_use_rule",
        }

    meta: Dict[str, Any] = {}
    feature_cols: list[str] = []
    if meta_path is not None:
        meta = _load_pfill_meta(meta_path)
        feature_cols = _resolve_feature_cols(meta)

    try:
        model = joblib.load(chosen)
        model_kind = "lgbm" if "lgbm" in chosen.name.lower() else "lr"
        feature_mode = "meta_feature_contract" if feature_cols else "pipeline_auto"
        return PFillModelBundle(
            model=model,
            model_kind=model_kind,
            model_path=str(chosen),
            feature_mode=feature_mode,
            meta_path=str(meta_path) if meta_path else "",
            feature_cols=feature_cols,
        ), {
            "pfill_model_loaded": True,
            "pfill_model_kind": model_kind,
            "pfill_model_path": str(chosen),
            "pfill_model_feature_mode": feature_mode,
            "pfill_model_meta_path": str(meta_path) if meta_path else "",
            "pfill_model_expected_n_features": len(feature_cols),
            "pfill_model_degrade_reason": "",
        }
    except Exception as e:
        return None, {
            "pfill_model_loaded": False,
            "pfill_model_kind": "",
            "pfill_model_path": str(chosen),
            "pfill_model_feature_mode": "",
            "pfill_model_meta_path": str(meta_path) if meta_path else "",
            "pfill_model_expected_n_features": len(feature_cols),
            "pfill_model_degrade_reason": f"model_load_failed:{type(e).__name__}",
        }


LEAKAGE_COLS = {
    "realized_ret_t1_to_t2",
    "premium_ret_t1_to_t2",
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
    "exec_date",
    "entry_date",
    "entry_price_t1",
    "entry_price_proxy_t1",
    "entry_price_proxy_mode",
    "sample_maturity",
    "label_ready_fill",
    "label_ready_ret",
    "y_fill",
    "fill_label_quality",
    "eret_sample_eligible",
    "eret_label_quality",
    "dataset_split",
    "sample_weight",
    "eret_truth_version",
    "return_holding_mode",
    "buy_window_start",
    "buy_window_end",
    "p_fill_pred",
    "p_fill_pred_raw",
    "p_fill_pred_rule",
    "p_fill_pred_final",
}

ID_COLS = {
    "trade_date",
    "ts_code",
    "name",
    "run_id",
    "commit_sha",
    "generated_at_utc",
    "signal_date",
}


def _select_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = []
    for c in df.columns:
        if c in LEAKAGE_COLS or c in ID_COLS:
            continue
        if str(c).startswith("Unnamed:"):
            continue
        feature_cols.append(c)

    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in ID_COLS]

    return df[feature_cols].copy()


def _normalize_feature_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for c in out.columns:
        col = out[c]

        if pd.api.types.is_bool_dtype(col):
            out[c] = col.astype("int64")
            continue

        if pd.api.types.is_numeric_dtype(col):
            out[c] = pd.to_numeric(col, errors="coerce").replace([np.inf, -np.inf], np.nan)
            continue

        if pd.api.types.is_datetime64_any_dtype(col):
            out[c] = col.astype("string")
            continue

        if pd.api.types.is_categorical_dtype(col):
            out[c] = col.astype("object")
            continue

        out[c] = col.replace({None: np.nan})

    return out


def _build_feature_frame(df: pd.DataFrame, bundle: PFillModelBundle) -> pd.DataFrame:
    if bundle.feature_cols:
        x = pd.DataFrame(index=df.index)
        for col in bundle.feature_cols:
            if col in df.columns:
                x[col] = df[col]
            else:
                x[col] = np.nan
        return _normalize_feature_frame(x)

    return _normalize_feature_frame(_select_feature_frame(df))


def _predict_by_model(bundle: PFillModelBundle, df: pd.DataFrame) -> pd.Series:
    x = _build_feature_frame(df, bundle)

    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(x)
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            pred = arr[:, 1]
        else:
            pred = arr.reshape(-1)
    else:
        pred = bundle.model.predict(x)

    if isinstance(pred, pd.Series):
        out = pred.copy()
    else:
        out = pd.Series(np.asarray(pred).reshape(-1), index=df.index, name="p_fill_pred_model")
    return _clip_prob_series(out)


def apply_pfill_engine(
    df: pd.DataFrame,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    输入：
    - df: 候选池 + 证据层 DataFrame

    输出：
    - 原表附加以下字段：
      p_fill_pred
      p_fill_pred_rule
      p_fill_pred_model
      p_fill_pred_final
      p_fill_model_loaded
      p_fill_model_kind
      p_fill_model_path
      p_fill_model_feature_mode
      p_fill_pred_src
      p_fill_degrade_reason
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    rule_pred = fill_model_rule(out)
    rule_pred = _clip_prob_series(rule_pred)

    bundle, audit = _resolve_pfill_model(project_root=project_root)

    model_pred = pd.Series([np.nan] * len(out), index=out.index, name="p_fill_pred_model")
    pred_src = "rule"
    degrade_reason = str(audit.get("pfill_model_degrade_reason", "") or "")

    if bundle is not None:
        try:
            model_pred = _predict_by_model(bundle, out)
            pred_src = f"model:{bundle.model_kind}"
            degrade_reason = ""
        except Exception as e:
            pred_src = "rule"
            degrade_reason = f"model_predict_failed:{type(e).__name__}"

    final_pred = pd.to_numeric(model_pred, errors="coerce")
    final_pred = final_pred.where(final_pred.notna(), rule_pred)
    final_pred = _clip_prob_series(final_pred)

    out["p_fill_pred_rule"] = rule_pred
    out["p_fill_pred_model"] = pd.to_numeric(model_pred, errors="coerce")
    out["p_fill_pred_final"] = final_pred
    out["p_fill_pred"] = final_pred

    out["p_fill_model_loaded"] = bool(audit.get("pfill_model_loaded", False))
    out["p_fill_model_kind"] = str(audit.get("pfill_model_kind", ""))
    out["p_fill_model_path"] = str(audit.get("pfill_model_path", ""))
    out["p_fill_model_feature_mode"] = str(audit.get("pfill_model_feature_mode", ""))
    out["p_fill_model_meta_path"] = str(audit.get("pfill_model_meta_path", ""))
    out["p_fill_model_expected_n_features"] = int(audit.get("pfill_model_expected_n_features", 0) or 0)
    out["p_fill_pred_src"] = pred_src
    out["p_fill_degrade_reason"] = degrade_reason

    return out


__all__ = [
    "apply_pfill_engine",
]
