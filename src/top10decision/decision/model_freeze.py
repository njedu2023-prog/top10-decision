from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


FREEZE_SCHEMA_VERSION = "decision_model_freeze_v1"
DEFAULT_FREEZE_PATH = Path("models/decision_model_freeze.json")
DATE_PATTERN = re.compile(r"^20\d{6}$")


class DecisionModelFreezeError(RuntimeError):
    """Raised when a frozen Decision production contract drifts."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DecisionModelFreezeError(f"model freeze manifest missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionModelFreezeError(f"model freeze manifest unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DecisionModelFreezeError("model freeze manifest must be a JSON object")
    return payload


def load_model_freeze(
    root: Path | str = Path("."),
    *,
    required: bool = False,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = root_path / DEFAULT_FREEZE_PATH
    if not path.exists() and not required:
        return {}
    payload = _read_json(path)
    if payload.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise DecisionModelFreezeError(
            f"unsupported model freeze schema: {payload.get('schema_version')}"
        )
    if not isinstance(payload.get("active"), bool):
        raise DecisionModelFreezeError("model freeze active must be boolean")
    cutoff = str(payload.get("training_cutoff_signal_date") or "")
    if payload["active"] and not DATE_PATTERN.fullmatch(cutoff):
        raise DecisionModelFreezeError(
            "active model freeze requires training_cutoff_signal_date=YYYYMMDD"
        )
    production = payload.get("production")
    if payload["active"] and not isinstance(production, dict):
        raise DecisionModelFreezeError("active model freeze requires production metadata")
    pinned = payload.get("pinned_files", {})
    if payload["active"] and not isinstance(pinned, dict):
        raise DecisionModelFreezeError("active model freeze requires pinned_files")
    return payload


def model_freeze_active(manifest: dict[str, Any]) -> bool:
    return manifest.get("active") is True


def apply_frozen_history_cutoff(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not model_freeze_active(manifest) or frame.empty:
        return frame.copy(), {
            "active": model_freeze_active(manifest),
            "freeze_id": str(manifest.get("freeze_id") or ""),
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "rows_removed": 0,
        }
    if "signal_date" not in frame.columns:
        raise DecisionModelFreezeError("training history has no signal_date column")

    cutoff = str(manifest["training_cutoff_signal_date"])
    signal_dates = (
        frame["signal_date"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.slice(0, 8)
    )
    valid = signal_dates.str.fullmatch(r"20\d{6}", na=False)
    eligible = valid & signal_dates.le(cutoff)
    filtered = frame.loc[eligible].copy().reset_index(drop=True)
    if filtered.empty:
        raise DecisionModelFreezeError(
            f"model freeze removed all training rows at cutoff {cutoff}"
        )
    kept_dates = sorted(filtered["signal_date"].astype(str).unique())
    audit = {
        "active": True,
        "freeze_id": str(manifest.get("freeze_id") or ""),
        "training_cutoff_signal_date": cutoff,
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(frame) - len(filtered)),
        "history_start": kept_dates[0] if kept_dates else "",
        "history_end": kept_dates[-1] if kept_dates else "",
    }
    return filtered, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_files(
    root: Path | str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    pinned = manifest.get("pinned_files", {}) or {}
    if not model_freeze_active(manifest):
        return {
            "active": False,
            "freeze_id": str(manifest.get("freeze_id") or ""),
            "pinned_files": int(len(pinned)),
            "validated": True,
            "enforced": False,
        }
    mismatches: list[dict[str, str]] = []
    for relative_path, expected in sorted(pinned.items()):
        path = root_path / str(relative_path)
        if not path.is_file():
            mismatches.append(
                {
                    "path": str(relative_path),
                    "expected": str(expected),
                    "actual": "MISSING",
                }
            )
            continue
        actual = _sha256(path)
        if actual != str(expected):
            mismatches.append(
                {
                    "path": str(relative_path),
                    "expected": str(expected),
                    "actual": actual,
                }
            )
    if mismatches:
        detail = "; ".join(
            f"{item['path']} expected={item['expected']} actual={item['actual']}"
            for item in mismatches
        )
        raise DecisionModelFreezeError(f"frozen file drift detected: {detail}")
    return {
        "active": model_freeze_active(manifest),
        "freeze_id": str(manifest.get("freeze_id") or ""),
        "pinned_files": int(len(pinned)),
        "validated": True,
        "enforced": True,
    }


def validate_runtime_artifacts(
    root: Path | str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not model_freeze_active(manifest):
        return {"active": False, "validated": True}
    root_path = Path(root).resolve()
    model_meta = _read_json(
        root_path / "outputs/auction_v3/models/model_meta_latest.json"
    )
    backtest = _read_json(
        root_path / "outputs/auction_v3/metrics/backtest_latest.json"
    )
    expected = manifest.get("production", {}) or {}
    selector = backtest.get("trade_selector", {}) or {}
    checks = {
        "model_version": (
            str(model_meta.get("model_version") or "")
            == str(expected.get("model_version") or "")
        ),
        "model_artifact_sha256": (
            str(model_meta.get("model_artifact_sha256") or "")
            == str(expected.get("model_artifact_sha256") or "")
        ),
        "model_promoted": (
            (model_meta.get("promoted") is True)
            == (expected.get("promoted") is True)
        ),
        "trade_selector_version": (
            str(selector.get("version") or "")
            == str(expected.get("trade_selector_version") or "")
        ),
        "trade_selector_artifact_sha256": (
            str(selector.get("production_artifact_sha256") or "")
            == str(expected.get("trade_selector_artifact_sha256") or "")
        ),
        "trade_selector_promoted": (
            (selector.get("promoted") is True)
            == (expected.get("trade_selector_promoted") is True)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        runtime_values = {
            "model_version": str(model_meta.get("model_version") or ""),
            "model_artifact_sha256": str(
                model_meta.get("model_artifact_sha256") or ""
            ),
            "model_promoted": model_meta.get("promoted") is True,
            "trade_selector_version": str(selector.get("version") or ""),
            "trade_selector_artifact_sha256": str(
                selector.get("production_artifact_sha256") or ""
            ),
            "trade_selector_promoted": selector.get("promoted") is True,
        }
        expected_values = {
            "model_version": str(expected.get("model_version") or ""),
            "model_artifact_sha256": str(
                expected.get("model_artifact_sha256") or ""
            ),
            "model_promoted": expected.get("promoted") is True,
            "trade_selector_version": str(
                expected.get("trade_selector_version") or ""
            ),
            "trade_selector_artifact_sha256": str(
                expected.get("trade_selector_artifact_sha256") or ""
            ),
            "trade_selector_promoted": (
                expected.get("trade_selector_promoted") is True
            ),
        }
        raise DecisionModelFreezeError(
            "frozen runtime artifact drift detected: "
            + ", ".join(failed)
            + "; expected="
            + json.dumps(expected_values, ensure_ascii=True, sort_keys=True)
            + "; actual="
            + json.dumps(runtime_values, ensure_ascii=True, sort_keys=True)
        )

    action_plan_path = root_path / "outputs/decision/action_plan_latest.json"
    action_plan_checks: dict[str, bool] = {}
    if action_plan_path.is_file():
        action_plan = _read_json(action_plan_path)
        action_model = action_plan.get("model", {}) or {}
        action_selector = action_model.get("trade_selector", {}) or {}
        action_plan_checks = {
            "formal_status": (
                str(action_plan.get("status_code") or "")
                == str(expected.get("formal_status") or "")
            ),
            "formal_buy_count": (
                int(action_plan.get("formal_buy_count") or 0)
                == int(expected.get("formal_buy_count") or 0)
            ),
            "action_model_version": (
                str(action_model.get("version") or "")
                == str(expected.get("model_version") or "")
            ),
            "action_model_artifact_sha256": (
                str(action_model.get("artifact_sha256") or "")
                == str(expected.get("model_artifact_sha256") or "")
            ),
            "action_model_promoted": (
                (action_model.get("promoted") is True)
                == (expected.get("promoted") is True)
            ),
            "action_selector_version": (
                str(action_selector.get("version") or "")
                == str(expected.get("trade_selector_version") or "")
            ),
            "action_selector_artifact_sha256": (
                str(action_model.get("trade_selector_artifact_sha256") or "")
                == str(expected.get("trade_selector_artifact_sha256") or "")
            ),
            "action_selector_promoted": (
                (action_selector.get("promoted") is True)
                == (expected.get("trade_selector_promoted") is True)
            ),
        }
        failed_action_plan = [
            name for name, passed in action_plan_checks.items() if not passed
        ]
        if failed_action_plan:
            raise DecisionModelFreezeError(
                "frozen action plan drift detected: "
                + ", ".join(failed_action_plan)
            )
    history_end = str(
        ((model_meta.get("data_coverage") or {}).get("history_end")) or ""
    )
    cutoff = str(manifest.get("training_cutoff_signal_date") or "")
    if history_end and history_end > cutoff:
        raise DecisionModelFreezeError(
            f"frozen history_end {history_end} exceeds cutoff {cutoff}"
        )
    return {
        "active": True,
        "freeze_id": str(manifest.get("freeze_id") or ""),
        "validated": True,
        "checks": checks,
        "action_plan_checks": action_plan_checks,
        "history_end": history_end,
        "training_cutoff_signal_date": cutoff,
    }


__all__ = [
    "DecisionModelFreezeError",
    "apply_frozen_history_cutoff",
    "load_model_freeze",
    "model_freeze_active",
    "validate_pinned_files",
    "validate_runtime_artifacts",
]
