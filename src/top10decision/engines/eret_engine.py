#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eret_engine.py

定位：
- E_ret 在线推理 / 计算引擎
- 负责把“候选池 + 证据层”转换为可直接参与 EV 的 eret_pred
- 优先加载学习模型；若模型缺失或推理失败，则自动回退到规则模型
- 不写文件，不直接做排序，不处理 weights / reports

职责边界：
- 输入：已经按候选池裁剪后的 DataFrame（通常来自 ingest.build_model_input）
- 输出：附加 E_ret 推理结果后的 DataFrame
- 不负责：
  1) 跨仓库拉数
  2) 真值构建
  3) 训练
  4) EV 融合
  5) 落盘输出

当前版本：
- v1：先兼容现有 overnight_model_rule 的规则逻辑
- 若 models/eret_lr.joblib 或 models/eret_lgbm.joblib 存在，则优先走学习模型
- 学习模型输出做合理裁剪，避免极端离群值直接污染 EV
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from top10decision.models.overnight_model import overnight_model_rule
except Exception:  # pragma: no cover
    def overnight_model_rule(df: pd.DataFrame, regime: str = "RISK_ON") -> pd.Series:
        # 最低兜底：若规则模型导入失败，给一个很弱的零收益先验
        return pd.Series(np.zeros(len(df)), index=df.index, name="eret_rule_fallback")


PRED_MIN = -0.30
PRED_MAX = 0.30


def _detect_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clip_ret_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=PRED_MIN, upper=PRED_MAX)


def _existing_model_path(root: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = root / "models" / name
        if p.exists():
            return p
    return None


@dataclass
class ERetModelBundle:
    model: Any
    model_kind: str
    model_path: str
    feature_mode: str


def _resolve_eret_model(project_root: Optional[Path] = None) -> Tuple[Optional[ERetModelBundle], Dict[str, Any]]:
    root = project_root or _detect_project_root()

    lgbm_path = _existing_model_path(root, ["eret_lgbm.joblib"])
    lr_path = _existing_model_path(root, ["eret_lr.joblib"])

    chosen = lgbm_path or lr_path
    if chosen is None:
        return None, {
            "eret_model_loaded": False,
            "eret_model_kind": "",
            "eret_model_path": "",
            "eret_model_feature_mode": "",
            "eret_model_degrade_reason": "model_missing_use_rule",
        }

    try:
        model = joblib.load(chosen)
        model_kind = "lgbm" if "lgbm" in chosen.name.lower() else "lr"
        feature_mode = "pipeline_auto" if hasattr(model, "predict") else "unknown"
        return ERetModelBundle(
            model=model,
            model_kind=model_kind,
            model_path=str(chosen),
            feature_mode=feature_mode,
        ), {
            "eret_model_loaded": True,
            "eret_model_kind": model_kind,
            "eret_model_path": str(chosen),
            "eret_model_feature_mode": feature_mode,
            "eret_model_degrade_reason": "",
        }
    except Exception as e:
        return None, {
            "eret_model_loaded": False,
            "eret_model_kind": "",
            "eret_model_path": str(chosen),
            "eret_model_feature_mode": "",
            "eret_model_degrade_reason": f"model_load_failed:{type(e).__name__}",
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
    "e_ret_pred",
    "eret_pred",
    "eret_pred_raw",
    "eret_pred_rule",
    "eret_pred_final",
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


def _predict_by_model(bundle: ERetModelBundle, df: pd.DataFrame) -> pd.Series:
    x = _select_feature_frame(df)
    pred = bundle.model.predict(x)
    if isinstance(pred, pd.Series):
        out = pred.copy()
    else:
        out = pd.Series(np.asarray(pred).reshape(-1), index=df.index, name="eret_pred_model")
    return _clip_ret_series(out)


def _get_regime_name(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "RISK_ON"
    for col in ["regime_name", "regime"]:
        if col in df.columns:
            try:
                v = df[col].dropna().astype(str).str.strip()
                v = v[v != ""]
                if not v.empty:
                    return str(v.iloc[0])
            except Exception:
                pass
    return "RISK_ON"


def apply_eret_engine(
    df: pd.DataFrame,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    输入：
    - df: 候选池 + 证据层 DataFrame

    输出：
    - 原表附加以下字段：
      eret_pred
      e_ret_pred
      eret_pred_rule
      eret_pred_model
      eret_pred_final
      eret_model_loaded
      eret_model_kind
      eret_model_path
      eret_model_feature_mode
      eret_pred_src
      eret_degrade_reason
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    regime_name = _get_regime_name(out)

    rule_pred = overnight_model_rule(out, regime=regime_name)
    if not isinstance(rule_pred, pd.Series):
        rule_pred = pd.Series(np.asarray(rule_pred).reshape(-1), index=out.index, name="eret_rule")
    rule_pred = _clip_ret_series(rule_pred)

    bundle, audit = _resolve_eret_model(project_root=project_root)

    model_pred = pd.Series([np.nan] * len(out), index=out.index, name="eret_pred_model")
    pred_src = "rule"
    degrade_reason = str(audit.get("eret_model_degrade_reason", "") or "")

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
    final_pred = _clip_ret_series(final_pred)

    out["eret_pred_rule"] = rule_pred
    out["eret_pred_model"] = pd.to_numeric(model_pred, errors="coerce")
    out["eret_pred_final"] = final_pred
    out["eret_pred"] = final_pred
    out["e_ret_pred"] = final_pred

    out["eret_model_loaded"] = bool(audit.get("eret_model_loaded", False))
    out["eret_model_kind"] = str(audit.get("eret_model_kind", ""))
    out["eret_model_path"] = str(audit.get("eret_model_path", ""))
    out["eret_model_feature_mode"] = str(audit.get("eret_model_feature_mode", ""))
    out["eret_pred_src"] = pred_src
    out["eret_degrade_reason"] = degrade_reason
    out["eret_regime_used"] = regime_name

    return out


__all__ = [
    "apply_eret_engine",
]
