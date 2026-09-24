"""Fit one frozen final baseline run using train_val only, after the gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn

from validate_baselines import (RESULTS, ROOT, load_candidates, load_inputs,
                                load_json, load_setup, make_pipeline, require_xgb_cuda, sha256)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_ids_sha256(indices: np.ndarray) -> str:
    payload = "".join(f"caco2_wang:train_val:{int(i)}\n" for i in indices)
    return hashlib.sha256(payload.encode()).hexdigest()


def current_versions() -> dict:
    return {"python": platform.python_version(), "PyTDC": version("PyTDC"),
            "RDKit": rdkit.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": version("xgboost")}


def selection_for(experiment: dict, algorithms: dict, candidate_index: dict,
                  frozen: dict) -> dict:
    saved = load_json(RESULTS / experiment["experiment_id"] / "validation/best_config.json")
    candidates = load_candidates(experiment, algorithms, candidate_index)
    choices = [item for item in candidates if item["candidate_id"] == saved["candidate_id"]]
    if (len(choices) != 1 or saved["params"] != choices[0]["params"]
            or saved["candidate_id"] != frozen["candidate_id"]
            or frozen["hyperparameters"] != choices[0]["params"]):
        raise ValueError("Frozen configuration differs from validated candidate")
    return choices[0]


def run_name_and_indices(args: argparse.Namespace, experiment: dict,
                         splits: dict[int, dict]) -> tuple[str, np.ndarray, int | None]:
    if args.protocol == "tdc_compatible_5split":
        if args.split_seed not in (1, 2, 3, 4, 5) or args.model_seed is not None:
            raise ValueError("Primary protocol requires exactly one official split seed")
        return f"split_seed_{args.split_seed}", splits[args.split_seed]["train"], args.split_seed
    if args.split_seed is not None:
        raise ValueError("Full-train refit cannot use a split seed")
    random_model = experiment["algorithm"] in ("rf", "et", "xgb")
    if random_model:
        if args.model_seed not in (1, 2, 3, 4, 5):
            raise ValueError("Random full-train refit requires model seed 1..5")
        return f"model_seed_{args.model_seed}", np.arange(728, dtype=np.int64), args.model_seed
    if args.model_seed is not None:
        raise ValueError("Deterministic full-train refit runs once without a model seed")
    return "single", np.arange(728, dtype=np.int64), None


def fit_final(args: argparse.Namespace) -> None:
    gate_path = RESULTS / "validation_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("Validation Gate has not passed; final training is locked")
    gate = load_json(gate_path)
    if gate.get("status") != "PASS" or len(gate.get("frozen_configs", {})) != 16:
        raise RuntimeError("Complete 16-experiment Validation Gate is required")
    reps, algorithms, config, candidate_index = load_setup()
    matches = [item for item in config["experiments"] if item["experiment_id"] == args.experiment]
    if len(matches) != 1:
        raise ValueError("Unknown experiment ID")
    experiment = matches[0]
    frozen_path = RESULTS / args.experiment / "frozen_config.json"
    if gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) != sha256(frozen_path):
        raise ValueError("Frozen configuration SHA-256 differs from gate")
    frozen = load_json(frozen_path)
    if (frozen["status"] != "CONFIG FROZEN" or frozen["experiment_id"] != args.experiment
            or frozen["algorithm"] != experiment["algorithm"]
            or frozen["representation"] != experiment["representation"]
            or frozen["preprocessing"] != experiment["preprocessing"]):
        raise ValueError("Frozen experiment definition changed")
    versions = current_versions()
    if versions != frozen["software_versions"]:
        raise ValueError("Current software versions differ from frozen configuration")
    candidate = selection_for(experiment, algorithms, candidate_index, frozen)
    labels, matrices, splits, input_audit = load_inputs(reps)
    if input_audit["test_partition_opened"] or input_audit["test_y_read"]:
        raise ValueError("Final training input loader opened fixed test")
    run_name, train_idx, random_state = run_name_and_indices(args, experiment, splits)
    expected_n = 637 if args.protocol == "tdc_compatible_5split" else 728
    if len(train_idx) != expected_n or len(set(train_idx.tolist())) != expected_n:
        raise ValueError("Training row count or uniqueness changed")
    if args.protocol == "full_train_refit" and train_idx.tolist() != list(range(728)):
        raise ValueError("Full-train refit must preserve all 728 source rows")
    destination = RESULTS / args.experiment / "final" / args.protocol / run_name
    if destination.exists():
        raise FileExistsError(f"Final model run already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".training_", dir=destination.parent))
    started_at = utc_now()
    start = time.monotonic()
    try:
        metadata = {
            "experiment_id": args.experiment,
            "benchmark_protocol": args.protocol,
            "split_seed": args.split_seed,
            "model_seed": args.model_seed,
            "random_state": random_state if experiment["algorithm"] in ("rf", "et", "xgb") else None,
            "train_n": expected_n,
            "train_row_ids_sha256": row_ids_sha256(train_idx),
            "train_raw_sha256": sha256(ROOT / "data/caco_trackA/raw/caco2_wang_train_val.csv"),
            "splits_manifest_sha256": sha256(ROOT / "data/caco_trackA/process/splits_manifest.json"),
            "frozen_config_sha256": sha256(frozen_path),
            "feature_manifest_sha256": frozen["feature_manifest_sha256"],
            "train_feature_matrix_sha256": (reps[experiment["representation"]]["train_val_matrix_sha256"]
                                            if experiment["representation"] else None),
            "candidate_id": candidate["candidate_id"],
            "hyperparameters": candidate["params"],
            "preprocessing": experiment["preprocessing"],
            "label_scale": "original TDC Y",
            "software_versions": versions,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_worktree_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "training_started_at_utc": started_at,
            "test_partition_opened": False,
            "test_y_read": False,
        }
        y_train = labels[train_idx]
        if experiment["algorithm"] == "null":
            statistic = frozen["null_statistic"]
            if statistic != experiment["statistic"] or statistic not in ("mean", "median"):
                raise ValueError("Null statistic differs from frozen experiment")
            constant = float(np.mean(y_train)) if statistic == "mean" else float(np.median(y_train))
            metadata.update({"artifact_format": "null_constant", "constant_prediction": constant,
                             "artifact_path": None, "artifact_sha256": None})
        else:
            representation = experiment["representation"]
            if reps[representation]["feature_manifest_sha256"] != frozen["feature_manifest_sha256"]:
                raise ValueError("Feature manifest differs from frozen configuration")
            x_train = matrices[representation][train_idx]
            if representation != "morgan_r2_2048" and np.isnan(x_train).all(axis=0).any():
                raise ValueError("All-missing training descriptor")
            pipeline = make_pipeline(experiment, algorithms[experiment["algorithm"]],
                                     candidate["params"], random_state or 1)
            pipeline.fit(x_train, y_train)
            if experiment["algorithm"] == "xgb":
                metadata["actual_training_device"] = require_xgb_cuda(pipeline)
            artifact = temp_dir / "model.joblib"
            joblib.dump(pipeline, artifact, compress=3)
            metadata.update({"artifact_format": "joblib",
                             "artifact_path": str((destination / "model.joblib").relative_to(ROOT)),
                             "artifact_sha256": sha256(artifact)})
        metadata["training_finished_at_utc"] = utc_now()
        metadata["training_runtime_seconds"] = time.monotonic() - start
        (temp_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"Final model run appeared during fitting: {destination}")
        os.replace(temp_dir, destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    print(f"Final model saved: {destination.relative_to(ROOT)}; train rows {expected_n}; test not opened")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-inputs", action="store_true")
    group.add_argument("--experiment")
    parser.add_argument("--protocol", choices=["tdc_compatible_5split", "full_train_refit"])
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    if args.audit_inputs:
        if args.protocol is not None or args.split_seed is not None or args.model_seed is not None:
            raise ValueError("Input audit takes no training protocol or seed")
        reps, _, config, _ = load_setup()
        _, _, splits, input_audit = load_inputs(reps)
        report = {"planned_experiments": len(config["experiments"]),
                  "official_split_seeds": sorted(splits),
                  "train_val_rows": input_audit["train_val_rows"],
                  "files_opened": input_audit["files_opened"],
                  "validation_gate_present": (RESULTS / "validation_gate.json").exists(),
                  "test_partition_opened": False, "test_y_read": False,
                  "model_training": False, "model_artifact_written": False}
        if args.audit_report:
            path = args.audit_report
            if path.exists():
                raise FileExistsError(f"Input audit output exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("Final training input audit passed; gate status recorded; no model trained or test opened")
        return
    if not args.protocol:
        raise ValueError("Final training requires --protocol")
    fit_final(args)


if __name__ == "__main__":
    main()
