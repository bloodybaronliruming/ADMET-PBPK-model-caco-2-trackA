"""Verify the saved Morgan inputs are structure-only and split-aligned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_morgan_features import (
    FP_SIZE,
    INCLUDE_CHIRALITY,
    RADIUS,
    ROOT,
    USE_BOND_TYPES,
    matrix_from_smiles,
    sha256,
)


PROCESS = ROOT / "data/caco_trackA/process"
RAW = ROOT / "data/caco_trackA/raw"
FEATURES = PROCESS / "features/morgan_r2_2048"


def read_verified_matrix(partition: str, manifest: dict) -> np.ndarray:
    record = manifest["partitions"][partition]
    path = ROOT / record["matrix_path"]
    if sha256(path) != record["matrix_sha256"]:
        raise ValueError(f"Feature file SHA-256 changed: {partition}")
    with np.load(path, allow_pickle=False) as saved:
        if saved.files != ["X"]:
            raise ValueError(f"Feature file contains keys besides X: {partition}")
        matrix = saved["X"]
    if matrix.shape != (record["rows"], FP_SIZE) or matrix.dtype != np.uint8:
        raise ValueError(f"Feature schema changed: {partition}")
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError(f"Non-binary Morgan feature value: {partition}")
    row_map_path = ROOT / record["row_map"]
    if sha256(row_map_path) != record["row_map_sha256"]:
        raise ValueError(f"row_id mapping changed: {partition}")
    source_path = RAW / f"caco2_wang_{partition}.csv"
    if sha256(source_path) != record["raw_sha256"]:
        raise ValueError(f"Raw source changed: {partition}")
    # Read only structure. Regeneration detects row order errors and any
    # label-derived features without accessing Y, including the fixed test Y.
    smiles = pd.read_csv(source_path, usecols=["Drug"])["Drug"].tolist()
    if not np.array_equal(matrix, matrix_from_smiles(smiles)):
        raise ValueError(f"Feature matrix differs from Drug-only regeneration: {partition}")
    return matrix


def main() -> None:
    manifest = json.loads((FEATURES / "features_manifest.json").read_text())
    if manifest["input_column"] != "Drug" or manifest["output_array_key"] != "X":
        raise ValueError("Unexpected feature input or output declaration")
    expected_fingerprint = {
        "method": "RDKit Morgan bit vector",
        "radius": RADIUS,
        "fp_size": FP_SIZE,
        "include_chirality": INCLUDE_CHIRALITY,
        "use_bond_types": USE_BOND_TYPES,
        "count_simulation": False,
    }
    if manifest["fingerprint"] != expected_fingerprint:
        raise ValueError("Fingerprint manifest differs from the implemented generator")
    columns = manifest["feature_names_in_order"]
    if columns != [f"morgan_{i:04d}" for i in range(FP_SIZE)]:
        raise ValueError("Feature names or order changed")
    train_x = read_verified_matrix("train_val", manifest)
    test_x = read_verified_matrix("test", manifest)
    split_manifest = json.loads((PROCESS / "splits_manifest.json").read_text())
    seed_records = {}
    for seed in split_manifest["seeds"]:
        groups = {}
        for name in ("train", "valid"):
            path = PROCESS / "splits" / f"seed_{seed}" / f"{name}.csv"
            frame = pd.read_csv(path, usecols=["row_id", "source_row"])
            positions = frame["source_row"].tolist()
            ids = frame["row_id"].tolist()
            if ids != [f"caco2_wang:train_val:{i}" for i in positions]:
                raise ValueError(f"Seed {seed} {name} row_id mismatch")
            if len(set(positions)) != len(positions) or any(
                not 0 <= i < len(train_x) for i in positions
            ):
                raise ValueError(f"Seed {seed} {name} duplicate or invalid source row")
            groups[name] = positions
        if set(groups["train"]) & set(groups["valid"]):
            raise ValueError(f"Seed {seed} train/valid overlap")
        if set(groups["train"] + groups["valid"]) != set(range(len(train_x))):
            raise ValueError(f"Seed {seed} does not cover all train_val rows")
        seed_records[str(seed)] = {name: len(rows) for name, rows in groups.items()}

    report = {
        "scope": "Morgan feature schema and five official split alignments; no model fitted",
        "feature_manifest": (FEATURES / "features_manifest.json").relative_to(ROOT).as_posix(),
        "feature_manifest_sha256": sha256(FEATURES / "features_manifest.json"),
        "checks": {
            "arrays_contain_only_X": True,
            "X_binary_and_shape_correct": True,
            "X_matches_full_Drug_only_regeneration": True,
            "Y_and_metadata_absent_from_X": True,
            "row_order_matches_raw_partitions": True,
            "five_splits_cover_train_val_without_overlap": True,
            "test_labels_read": False,
        },
        "train_val_rows": len(train_x),
        "test_rows": len(test_x),
        "feature_count": FP_SIZE,
        "splits": seed_records,
        "pending_model_checks": [
            "test exclusion from hyperparameter tuning",
            "test exclusion from feature selection and fitted preprocessing",
            "test exclusion from early stopping",
            "external feature table row_id joins, if used",
            "full 182-row final test prediction coverage",
        ],
    }
    path = PROCESS / "qc/feature_leakage_audit.json"
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Existing audit differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
