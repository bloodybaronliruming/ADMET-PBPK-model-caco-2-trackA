"""Audit blind and evaluated final runs, then summarize one frozen protocol."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from validate_baselines import RESULTS, ROOT, load_json, sha256


BLIND_COLUMNS = {"row_id", "Drug_ID", "original_test_row", "Y_pred", "split_seed",
                 "model_seed", "experiment_id", "benchmark_protocol"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_once(path: Path, document: dict) -> None:
    content = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == content, f"Existing audit differs: {path}")
    else:
        path.write_bytes(content)


def planned_runs(protocol: str, algorithm: str) -> list[tuple[str, int | None, int | None, int]]:
    if protocol == "tdc_compatible_5split":
        return [(f"split_seed_{seed}", seed, None, 637) for seed in range(1, 6)]
    if protocol == "full_train_refit":
        if algorithm in ("rf", "et", "xgb"):
            return [(f"model_seed_{seed}", None, seed, 728) for seed in range(1, 6)]
        return [("single", None, None, 728)]
    raise ValueError("Unknown benchmark protocol")


def frozen_setup(experiment: str, protocol: str) -> tuple[Path, dict, str]:
    gate = load_json(RESULTS / "validation_gate.json")
    require(gate.get("status") == "PASS" and len(gate.get("frozen_configs", {})) == 16,
            "Complete Validation Gate is required")
    folder = RESULTS / experiment
    frozen_path = folder / "frozen_config.json"
    require(frozen_path.is_file(), "Frozen configuration missing")
    frozen_sha = sha256(frozen_path)
    require(gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) == frozen_sha,
            "Frozen configuration differs from gate")
    frozen = load_json(frozen_path)
    require(frozen["experiment_id"] == experiment and protocol in frozen["benchmark_protocols"],
            "Experiment or protocol differs from frozen configuration")
    return folder, frozen, frozen_sha


def audit_blind(experiment: str, protocol: str) -> tuple[dict, Path]:
    folder, frozen, frozen_sha = frozen_setup(experiment, protocol)
    protocol_dir = folder / "final" / protocol
    expected_ids = [f"caco2_wang:test:{i}" for i in range(182)]
    raw = load_json(ROOT / "data/caco_trackA/raw/dataset_manifest.json")
    # Only the identifier column is read. Test Y is not loaded in this phase.
    drug_ids = pd.read_csv(ROOT / raw["files"]["test"]["path"], usecols=["Drug_ID"])["Drug_ID"].astype(str).tolist()
    require(len(drug_ids) == 182, "Fixed test identifier count changed")
    records = []
    for run_name, split_seed, model_seed, train_n in planned_runs(protocol, frozen["algorithm"]):
        run_dir = protocol_dir / run_name
        metadata_path = run_dir / "model_metadata.json"
        blind_path = run_dir / "blind_predictions.csv"
        require(metadata_path.is_file() and blind_path.is_file(), f"Incomplete blind run: {run_name}")
        metadata = load_json(metadata_path)
        require(metadata["experiment_id"] == experiment
                and metadata["benchmark_protocol"] == protocol
                and metadata["split_seed"] == split_seed
                and metadata["model_seed"] == model_seed
                and metadata["train_n"] == train_n
                and metadata["frozen_config_sha256"] == frozen_sha
                and metadata["test_y_read"] is False,
                f"Training metadata mismatch: {run_name}")
        if frozen["algorithm"] == "null":
            require(metadata["artifact_format"] == "null_constant"
                    and metadata["artifact_path"] is None
                    and metadata["artifact_sha256"] is None,
                    f"Null artifact mismatch: {run_name}")
        else:
            artifact = ROOT / metadata["artifact_path"]
            require(metadata["artifact_format"] == "joblib" and artifact.is_file()
                    and sha256(artifact) == metadata["artifact_sha256"],
                    f"Model artifact mismatch: {run_name}")
        if frozen["algorithm"] == "xgb":
            require(str(metadata.get("actual_training_device", "")).startswith("cuda"),
                    f"GPU training not confirmed: {run_name}")
        blind = pd.read_csv(blind_path)
        require(BLIND_COLUMNS.issubset(blind.columns)
                and "Y_true" not in blind.columns and "residual" not in blind.columns
                and len(blind) == 182,
                f"Blind schema or row count mismatch: {run_name}")
        require(blind["row_id"].tolist() == expected_ids
                and blind["original_test_row"].tolist() == list(range(182))
                and blind["Drug_ID"].astype(str).tolist() == drug_ids
                and blind["experiment_id"].tolist() == [experiment] * 182
                and blind["benchmark_protocol"].tolist() == [protocol] * 182,
                f"Blind identity or order mismatch: {run_name}")
        actual_split = None if split_seed is None else [split_seed] * 182
        actual_model = None if model_seed is None else [model_seed] * 182
        require((blind["split_seed"].isna().all() if actual_split is None
                 else blind["split_seed"].tolist() == actual_split)
                and (blind["model_seed"].isna().all() if actual_model is None
                     else blind["model_seed"].tolist() == actual_model),
                f"Blind seed mismatch: {run_name}")
        predicted = blind["Y_pred"].to_numpy(dtype=np.float64)
        require(np.isfinite(predicted).all(), f"Nonfinite blind prediction: {run_name}")
        if frozen["algorithm"] == "null":
            require(np.allclose(predicted, metadata["constant_prediction"], rtol=0, atol=1e-12),
                    f"Null prediction differs from training constant: {run_name}")
        records.append({"run_name": run_name, "split_seed": split_seed,
                        "model_seed": model_seed, "train_n": train_n, "test_n": 182,
                        "model_metadata_sha256": sha256(metadata_path),
                        "blind_predictions_sha256": sha256(blind_path)})
    report = {"experiment_id": experiment, "benchmark_protocol": protocol,
              "frozen_config_sha256": frozen_sha, "run_count": len(records),
              "runs": records, "test_y_read": False, "status": "PASS"}
    output = protocol_dir / "blind_audit.json"
    write_once(output, report)
    return report, output


def audit_evaluated(experiment: str, protocol: str) -> None:
    blind_report, blind_audit_path = audit_blind(experiment, protocol)
    folder, frozen, frozen_sha = frozen_setup(experiment, protocol)
    protocol_dir = folder / "final" / protocol
    raw = load_json(ROOT / "data/caco_trackA/raw/dataset_manifest.json")
    test_path = ROOT / raw["files"]["test"]["path"]
    require(sha256(test_path) == raw["files"]["test"]["sha256"], "Fixed test file changed")
    truth = pd.read_csv(test_path, usecols=["Drug_ID", "Y"])
    require(len(truth) == 182 and np.isfinite(truth["Y"].to_numpy(dtype=float)).all(),
            "Fixed test labels invalid")
    actual_y = truth["Y"].to_numpy(dtype=float)
    records = []
    for item in blind_report["runs"]:
        run_dir = protocol_dir / item["run_name"]
        blind = pd.read_csv(run_dir / "blind_predictions.csv")
        evaluated_path = run_dir / "evaluated_predictions.csv"
        metrics_path = run_dir / "metrics.json"
        require(evaluated_path.is_file() and metrics_path.is_file(),
                f"Missing evaluated run: {item['run_name']}")
        evaluated = pd.read_csv(evaluated_path)
        metrics = load_json(metrics_path)
        require(len(evaluated) == 182
                and evaluated["row_id"].tolist() == blind["row_id"].tolist()
                and evaluated["original_test_row"].tolist() == list(range(182))
                and evaluated["Drug_ID"].astype(str).tolist() == truth["Drug_ID"].astype(str).tolist(),
                f"Evaluated identity/order mismatch: {item['run_name']}")
        predicted = blind["Y_pred"].to_numpy(dtype=float)
        require(np.allclose(evaluated["Y_pred"].to_numpy(dtype=float), predicted, rtol=0, atol=1e-12)
                and np.allclose(evaluated["Y_true"].to_numpy(dtype=float), actual_y, rtol=0, atol=1e-12)
                and np.allclose(evaluated["residual"].to_numpy(dtype=float), predicted - actual_y,
                                rtol=0, atol=1e-12),
                f"Evaluated values mismatch: {item['run_name']}")
        recomputed = {"MAE": float(mean_absolute_error(actual_y, predicted)),
                      "RMSE": float(math.sqrt(mean_squared_error(actual_y, predicted))),
                      "R2": float(r2_score(actual_y, predicted))}
        require(metrics["experiment_id"] == experiment
                and metrics["benchmark_protocol"] == protocol
                and metrics["rows"] == 182
                and metrics["blind_predictions_sha256"] == item["blind_predictions_sha256"]
                and metrics["frozen_config_sha256"] == frozen_sha
                and metrics["test_raw_sha256"] == raw["files"]["test"]["sha256"]
                and all(math.isclose(float(metrics[name]), value, rel_tol=0, abs_tol=1e-12)
                        for name, value in recomputed.items()),
                f"Saved metrics mismatch: {item['run_name']}")
        records.append({**item, **recomputed,
                        "evaluated_predictions_sha256": sha256(evaluated_path),
                        "metrics_sha256": sha256(metrics_path)})
    maes = [record["MAE"] for record in records]
    summary = {"experiment_id": experiment, "benchmark_protocol": protocol,
               "frozen_config_sha256": frozen_sha, "run_count": len(records),
               "run_level_MAE": [{"run_name": r["run_name"], "MAE": r["MAE"]} for r in records],
               "MAE_mean": statistics.mean(maes),
               "MAE_SD": statistics.stdev(maes) if len(maes) > 1 else None,
               "MAE_SD_definition": "sample SD across official split runs" if protocol == "tdc_compatible_5split"
                                    else "sample SD across model seeds" if len(maes) > 1 else "N/A: one deterministic full-train run",
               "label_scale": "original TDC Y", "test_rows_per_run": 182,
               "selection_changed_after_test": False,
               "blind_audit_sha256": sha256(blind_audit_path)}
    audit = {"status": "PASS", "experiment_id": experiment,
             "benchmark_protocol": protocol, "test_y_read_only_after_blind_audit": True,
             "runs": records, "summary": summary}
    write_once(protocol_dir / "evaluation_audit.json", audit)
    write_once(protocol_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--protocol", required=True,
                        choices=["tdc_compatible_5split", "full_train_refit"])
    parser.add_argument("--mode", required=True, choices=["blind", "evaluated"])
    args = parser.parse_args()
    if args.mode == "blind":
        report, path = audit_blind(args.experiment, args.protocol)
        print(f"Blind audit PASS: {report['run_count']} runs; test Y not read; {path.relative_to(ROOT)}")
    else:
        audit_evaluated(args.experiment, args.protocol)
        print(f"Evaluation audit PASS: {args.experiment}, {args.protocol}; run summary saved")


if __name__ == "__main__":
    main()
