"""Evaluate a frozen label-blind prediction file against official test Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from validate_baselines import RESULTS, ROOT, load_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-predictions", required=True, type=Path)
    args = parser.parse_args()
    blind_path = args.blind_predictions.resolve()
    if not blind_path.is_relative_to(RESULTS.resolve()) or blind_path.name != "blind_predictions.csv":
        raise ValueError("Blind predictions must be in the versioned baseline results directory")
    gate = load_json(RESULTS / "validation_gate.json")
    if gate["status"] != "PASS":
        raise ValueError("Validation Gate has not passed")
    blind = pd.read_csv(blind_path)
    required = {"row_id", "Drug_ID", "original_test_row", "Y_pred", "split_seed",
                "model_seed", "experiment_id", "benchmark_protocol"}
    if not required.issubset(blind.columns) or "Y_true" in blind.columns or "residual" in blind.columns:
        raise ValueError("Blind prediction schema is invalid")
    if len(blind) != 182 or blind["experiment_id"].nunique() != 1 or blind["benchmark_protocol"].nunique() != 1:
        raise ValueError("Expected exactly one complete 182-row blind run")
    exp_id = str(blind["experiment_id"].iloc[0])
    protocol = str(blind["benchmark_protocol"].iloc[0])
    exp_dir = RESULTS / exp_id
    if protocol not in ("tdc_compatible_5split", "full_train_refit") or not blind_path.is_relative_to(exp_dir.resolve()):
        raise ValueError("Experiment/protocol path mismatch")
    frozen_path = exp_dir / "frozen_config.json"
    if gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) != sha256(frozen_path):
        raise ValueError("Frozen configuration changed")
    blind_audit_path = exp_dir / "final" / protocol / "blind_audit.json"
    if not blind_audit_path.is_file():
        raise ValueError("Complete label-blind audit is required before test Y is read")
    blind_audit = load_json(blind_audit_path)
    matching_runs = [item for item in blind_audit.get("runs", [])
                     if item.get("run_name") == blind_path.parent.name]
    if (blind_audit.get("status") != "PASS"
            or blind_audit.get("experiment_id") != exp_id
            or blind_audit.get("benchmark_protocol") != protocol
            or blind_audit.get("frozen_config_sha256") != sha256(frozen_path)
            or len(matching_runs) != 1
            or matching_runs[0].get("blind_predictions_sha256") != sha256(blind_path)):
        raise ValueError("Blind predictions differ from audited predictions")
    expected_ids = [f"caco2_wang:test:{i}" for i in range(182)]
    if (blind["row_id"].tolist() != expected_ids or blind["original_test_row"].tolist() != list(range(182))
            or not np.isfinite(blind["Y_pred"].to_numpy(dtype=float)).all()):
        raise ValueError("Blind prediction row coverage/order or values invalid")
    raw_manifest = load_json(ROOT / "data/caco_trackA/raw/dataset_manifest.json")
    raw_test = ROOT / raw_manifest["files"]["test"]["path"]
    if sha256(raw_test) != raw_manifest["files"]["test"]["sha256"]:
        raise ValueError("Official test file changed")
    truth = pd.read_csv(raw_test, usecols=["Drug_ID", "Y"])
    if len(truth) != 182 or truth["Drug_ID"].tolist() != blind["Drug_ID"].tolist():
        raise ValueError("Official test identifiers differ from blind predictions")
    if not np.isfinite(truth["Y"].to_numpy(dtype=float)).all():
        raise ValueError("Official test Y contains nonfinite values")
    evaluated = blind.copy()
    evaluated.insert(evaluated.columns.get_loc("Y_pred"), "Y_true", truth["Y"].to_numpy(dtype=float))
    evaluated.insert(evaluated.columns.get_loc("Y_pred") + 1, "residual",
                     evaluated["Y_pred"] - evaluated["Y_true"])
    y_true = evaluated["Y_true"].to_numpy(dtype=float)
    y_pred = evaluated["Y_pred"].to_numpy(dtype=float)
    metrics = {"experiment_id": exp_id, "benchmark_protocol": protocol,
               "rows": 182, "MAE": float(mean_absolute_error(y_true, y_pred)),
               "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
               "R2": float(r2_score(y_true, y_pred)),
               "blind_predictions_sha256": sha256(blind_path),
               "frozen_config_sha256": sha256(frozen_path),
               "test_raw_sha256": raw_manifest["files"]["test"]["sha256"]}
    output = blind_path.parent / "evaluated_predictions.csv"
    metric_path = blind_path.parent / "metrics.json"
    if output.exists() or metric_path.exists():
        raise FileExistsError("Evaluation output already exists")
    evaluated.to_csv(output, index=False)
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(f"Evaluated {len(evaluated)} fixed test rows; MAE {metrics['MAE']:.6f}")


if __name__ == "__main__":
    main()
