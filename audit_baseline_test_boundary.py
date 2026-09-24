"""Audit the fixed test identity and current validation-only data boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
SCRIPTS = ROOT / "caco_trackA_v1/scripts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    require(not output.exists(), f"Refusing to overwrite: {output}")

    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    test_record = raw_manifest["files"]["test"]
    raw_test = ROOT / test_record["path"]
    cache_test = ROOT / test_record["cache_path"]
    require(raw_manifest["benchmark_name"] == "caco2_wang", "Benchmark mismatch")
    require(test_record["rows"] == 182, "Manifest test count mismatch")
    require(sha256(raw_test) == sha256(cache_test) == test_record["sha256"], "Test raw/cache SHA mismatch")
    # The Y column is never parsed into a data frame in this audit.
    identifiers = pd.read_csv(raw_test, usecols=["Drug_ID", "Drug"])
    require(len(identifiers) == 182, "Fixed test row count mismatch")
    require(identifiers.notna().all().all(), "Missing test identifier or SMILES")

    row_map_path = PROCESS / "row_ids/test.csv"
    row_map = pd.read_csv(row_map_path)
    expected_ids = [f"caco2_wang:test:{i}" for i in range(182)]
    require(row_map["row_id"].tolist() == expected_ids, "Test row ID order mismatch")
    require(row_map["source_row"].tolist() == list(range(182)), "Test source order mismatch")
    require(row_map["source_partition"].tolist() == ["test"] * 182, "Test partition mismatch")

    feature_manifest = json.loads((PROCESS / "features/morgan_r2_2048/features_manifest.json").read_text())
    feature_record = feature_manifest["partitions"]["test"]
    require(feature_record["raw_sha256"] == test_record["sha256"], "Feature input raw SHA mismatch")
    require(feature_record["row_map_sha256"] == sha256(row_map_path), "Feature row map SHA mismatch")
    matrix_path = ROOT / feature_record["matrix_path"]
    require(feature_record["matrix_sha256"] == sha256(matrix_path), "Test feature SHA mismatch")
    with np.load(matrix_path, allow_pickle=False) as archive:
        require(archive.files == ["X"], "Test feature archive contains unexpected arrays")
        shape = list(archive["X"].shape)
    require(shape == [182, 2048] == feature_record["matrix_shape"], "Test feature dimensions mismatch")

    smoke_script = SCRIPTS / "check_validation_only_flow.py"
    smoke_report = ROOT / "caco_trackA_v1/results/split_leakage_smoke/validation_only_flow.json"
    smoke = json.loads(smoke_report.read_text())
    require(smoke["test_features_used"] is False, "Existing smoke flow used test features")
    require(smoke["test_labels_read"] is False, "Existing smoke flow read test labels")
    require(all("/test" not in path for path in smoke["inputs_opened"]), "Smoke inputs contain test")
    source = smoke_script.read_text()
    require('pd.read_csv(train_path, usecols=["Y"])' in source, "Smoke label source changed")
    require('RAW / "caco2_wang_train_val.csv"' in source, "Smoke raw source changed")
    require('RAW / "caco2_wang_test.csv"' not in source, "Smoke script references test raw data")

    formal_validation = SCRIPTS / "run_baseline_validation.sh"
    blind_predict = SCRIPTS / "predict_test.py"
    report = {
        "scope": "baseline checklist 0.1 fixed test and currently implemented validation boundary",
        "fixed_test_rows": 182,
        "test_raw_sha256": sha256(raw_test),
        "row_map_sha256": sha256(row_map_path),
        "test_row_ids_and_source_order_match": True,
        "test_feature_matrix": feature_record["matrix_path"],
        "test_feature_matrix_sha256": sha256(matrix_path),
        "test_feature_shape": shape,
        "test_feature_array_keys": ["X"],
        "existing_validation_smoke_script": str(smoke_script.relative_to(ROOT)),
        "existing_validation_smoke_script_sha256": sha256(smoke_script),
        "existing_validation_smoke_report": str(smoke_report.relative_to(ROOT)),
        "existing_validation_smoke_report_sha256": sha256(smoke_report),
        "existing_validation_smoke_test_data_used": False,
        "formal_validation_runner_exists": formal_validation.exists(),
        "formal_predict_test_exists": blind_predict.exists(),
        "formal_runner_boundary_verified": False,
        "test_y_values_inspected": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Verified fixed test: 182 rows; raw SHA, row IDs, source order and X-only features match")
    print("Existing validation smoke flow uses train_val only")
    print("Formal validation and blind prediction runners are not yet implemented")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
