"""Create label-blind predictions after the Validation Gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from validate_baselines import RESULTS, ROOT, load_json, sha256


def inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--protocol", required=True,
                        choices=["tdc_compatible_5split", "full_train_refit"])
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--model-metadata", required=True, type=Path)
    args = parser.parse_args()

    experiment_dir = RESULTS / args.experiment
    gate = load_json(RESULTS / "validation_gate.json")
    frozen_path = experiment_dir / "frozen_config.json"
    if gate["status"] != "PASS" or gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) != sha256(frozen_path):
        raise ValueError("Validation Gate or frozen configuration mismatch")
    frozen = load_json(frozen_path)
    if frozen["experiment_id"] != args.experiment:
        raise ValueError("Frozen experiment ID mismatch")
    if args.protocol == "tdc_compatible_5split":
        if args.split_seed not in (1, 2, 3, 4, 5) or args.model_seed is not None:
            raise ValueError("Primary protocol requires one official split seed")
        run_name = f"split_seed_{args.split_seed}"
        expected_n = 637
    else:
        if args.split_seed is not None:
            raise ValueError("Full-train refit must not use a split seed")
        random_model = frozen["algorithm"] in ("rf", "et", "xgb")
        if (random_model and args.model_seed not in (1, 2, 3, 4, 5)) or (not random_model and args.model_seed is not None):
            raise ValueError("Unexpected full-train model seed")
        run_name = f"model_seed_{args.model_seed}" if random_model else "single"
        expected_n = 728
    metadata_path = args.model_metadata.resolve()
    if not inside(metadata_path, experiment_dir) or not metadata_path.is_file():
        raise ValueError("Model metadata must be inside the experiment result directory")
    metadata = load_json(metadata_path)
    if (metadata["experiment_id"] != args.experiment or metadata["benchmark_protocol"] != args.protocol
            or metadata.get("split_seed") != args.split_seed or metadata.get("model_seed") != args.model_seed
            or metadata["train_n"] != expected_n or metadata["frozen_config_sha256"] != sha256(frozen_path)):
        raise ValueError("Model metadata differs from frozen experiment/protocol")

    raw_manifest = load_json(ROOT / "data/caco_trackA/raw/dataset_manifest.json")
    # Read only the identifier column; no test label is loaded here.
    raw_test = ROOT / raw_manifest["files"]["test"]["path"]
    identifiers = pd.read_csv(raw_test, usecols=["Drug_ID"])["Drug_ID"].tolist()
    row_map = pd.read_csv(ROOT / "data/caco_trackA/process/row_ids/test.csv",
                          usecols=["row_id", "source_row"])
    expected_ids = [f"caco2_wang:test:{i}" for i in range(182)]
    if (len(identifiers) != 182 or row_map["row_id"].tolist() != expected_ids
            or row_map["source_row"].tolist() != list(range(182))):
        raise ValueError("Fixed test identifiers/order changed")
    if frozen["representation"] is None:
        if metadata["artifact_format"] != "null_constant":
            raise ValueError("Null baseline requires a frozen constant")
        predicted = np.full(182, float(metadata["constant_prediction"]))
    else:
        feature_manifest_path = ROOT / "data/caco_trackA/process/features" / frozen["representation"] / "features_manifest.json"
        if sha256(feature_manifest_path) != frozen["feature_manifest_sha256"]:
            raise ValueError("Frozen feature manifest changed")
        feature_manifest = load_json(feature_manifest_path)
        test_info = feature_manifest["partitions"]["test"]
        feature_path = ROOT / test_info["matrix_path"]
        if sha256(feature_path) != test_info["matrix_sha256"]:
            raise ValueError("Fixed test feature matrix changed")
        if feature_path.suffix == ".npz":
            with np.load(feature_path, allow_pickle=False) as archive:
                if archive.files != ["X"]:
                    raise ValueError("Unexpected test matrix keys")
                x_test = archive["X"]
        else:
            x_test = np.load(feature_path, allow_pickle=False)
        if x_test.shape != (182, len(feature_manifest["feature_names_in_order"])):
            raise ValueError("Test feature shape changed")
        if metadata["artifact_format"] != "joblib":
            raise ValueError("Trained model must be a joblib artifact")
        model_path = ROOT / metadata["artifact_path"]
        if not inside(model_path, experiment_dir) or sha256(model_path) != metadata["artifact_sha256"]:
            raise ValueError("Model artifact path or checksum mismatch")
        model = joblib.load(model_path)
        predicted = np.asarray(model.predict(x_test), dtype=np.float64)
    if predicted.shape != (182,) or not np.isfinite(predicted).all():
        raise ValueError("Predictions must contain 182 finite values")
    frame = pd.DataFrame({"row_id": expected_ids, "Drug_ID": identifiers,
                          "original_test_row": list(range(182)), "Y_pred": predicted,
                          "split_seed": args.split_seed, "model_seed": args.model_seed,
                          "experiment_id": args.experiment, "benchmark_protocol": args.protocol})
    output = experiment_dir / "final" / args.protocol / run_name / "blind_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Blind prediction output exists: {output}")
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} label-blind predictions: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
