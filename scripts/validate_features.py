"""Audit all three frozen baseline representations against original Drug order."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_morgan_features import matrix_from_smiles as morgan_from_smiles
from build_physchem10 import matrix_from_smiles as physchem_from_smiles
from build_rdkit2d import calculate as rdkit2d_calculate
from build_rdkit2d import frozen_descriptors


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
REPRESENTATIONS = {
    "physchem10": ("features/physchem10/features_manifest.json", 10),
    "rdkit2d": ("features/rdkit2d/features_manifest.json", 210),
    "morgan_r2_2048": ("features/morgan_r2_2048/features_manifest.json", 2048),
}


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
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    split_manifest = json.loads((PROCESS / "splits_manifest.json").read_text())
    require(raw_manifest["benchmark_name"] == "caco2_wang", "Unexpected benchmark")
    require(split_manifest["seeds"] == [1, 2, 3, 4, 5], "Unexpected split seeds")
    names_2d, funcs_2d, frozen_sha = frozen_descriptors()

    source = {}
    for partition, count in (("train_val", 728), ("test", 182)):
        record = raw_manifest["files"][partition]
        path = ROOT / record["path"]
        require(record["rows"] == count and sha256(path) == record["sha256"], f"{partition}: raw input changed")
        smiles = pd.read_csv(path, usecols=["Drug"])["Drug"].tolist()
        require(len(smiles) == count and all(isinstance(s, str) and s for s in smiles), f"{partition}: invalid Drug column")
        row_map_path = PROCESS / "row_ids" / f"{partition}.csv"
        mapping = pd.read_csv(row_map_path)
        expected_ids = [f"caco2_wang:{partition}:{i}" for i in range(count)]
        require(mapping["row_id"].tolist() == expected_ids, f"{partition}: row ID order")
        require(mapping["source_row"].tolist() == list(range(count)), f"{partition}: source row order")
        require(mapping["source_partition"].tolist() == [partition] * count, f"{partition}: source partition")
        require(len(set(mapping["row_id"])) == count, f"{partition}: duplicate row mapping")
        source[partition] = (smiles, record, row_map_path)

    audit = {}
    for representation, (manifest_rel, width) in REPRESENTATIONS.items():
        manifest_path = PROCESS / manifest_rel
        manifest = json.loads(manifest_path.read_text())
        require(manifest["dataset"] == "caco2_wang", f"{representation}: benchmark mismatch")
        require(manifest["input_column"] == "Drug", f"{representation}: input is not Drug")
        require("Y" in manifest["input_fields_excluded"], f"{representation}: Y exclusion missing")
        names = manifest["feature_names_in_order"]
        require(len(names) == width and len(set(names)) == width, f"{representation}: names/order invalid")
        if representation == "rdkit2d":
            require(names == names_2d and manifest["descriptor_manifest_sha256"] == frozen_sha, "RDKit2D frozen list mismatch")
        if representation == "physchem10":
            require(names == [f["feature_name"] for f in manifest["features"]], "PhysChem-10 feature list mismatch")
        if representation == "morgan_r2_2048":
            require(names == [f"morgan_{i:04d}" for i in range(width)], "Morgan bit order mismatch")
            require(manifest["fingerprint"] == {
                "method": "RDKit Morgan bit vector", "radius": 2, "fp_size": 2048,
                "include_chirality": False, "use_bond_types": True, "count_simulation": False,
            }, "Morgan parameters changed")

        partitions = {}
        for partition, (smiles, raw_record, row_map_path) in source.items():
            item = manifest["partitions"][partition]
            require(item["rows"] == len(smiles), f"{representation}/{partition}: row count")
            require(item["raw_sha256"] == raw_record["sha256"], f"{representation}/{partition}: source SHA")
            require(item["row_map_sha256"] == sha256(row_map_path), f"{representation}/{partition}: map SHA")
            require(ROOT / item["row_map"] == row_map_path, f"{representation}/{partition}: map path")
            matrix_path = ROOT / item["matrix_path"]
            require(sha256(matrix_path) == item["matrix_sha256"], f"{representation}/{partition}: matrix SHA")
            if representation == "morgan_r2_2048":
                with np.load(matrix_path, allow_pickle=False) as archive:
                    require(archive.files == ["X"], f"{representation}/{partition}: unexpected array keys")
                    matrix = archive["X"]
                expected = morgan_from_smiles(smiles)
                require(matrix.dtype == np.uint8 and np.isin(matrix, [0, 1]).all(), f"{representation}/{partition}: nonbinary matrix")
            else:
                matrix = np.load(matrix_path, allow_pickle=False)
                require(matrix.dtype == np.float64, f"{representation}/{partition}: matrix dtype")
                if representation == "physchem10":
                    expected = physchem_from_smiles(smiles, partition)
                else:
                    expected, _ = rdkit2d_calculate(smiles, names_2d, funcs_2d, partition)
                require(not np.isinf(matrix).any(), f"{representation}/{partition}: infinity present")
                require(item["missing_count_by_feature"] == {
                    name: int(np.isnan(matrix[:, i]).sum()) for i, name in enumerate(names)
                }, f"{representation}/{partition}: missing counts differ")
            require(matrix.shape == tuple(item["matrix_shape"]) == (len(smiles), width), f"{representation}/{partition}: dimensions")
            require(np.array_equal(matrix, expected, equal_nan=True), f"{representation}/{partition}: row order or values differ from Drug-only calculation")
            partitions[partition] = {
                "rows": len(smiles),
                "columns": width,
                "matrix_sha256": item["matrix_sha256"],
                "missing_cells": int(np.isnan(matrix).sum()),
                "rows_with_missing": int(np.isnan(matrix).any(axis=1).sum()),
                "infinite_cells": int(np.isinf(matrix).sum()),
                "raw_order_recalculation_matches": True,
                "row_map_matches": True,
            }
        audit[representation] = {
            "feature_manifest": str(manifest_path.relative_to(ROOT)),
            "feature_manifest_sha256": sha256(manifest_path),
            "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "partitions": partitions,
        }

    split_checks = {}
    for seed in split_manifest["seeds"]:
        groups = {}
        for name, expected_count in (("train", 637), ("valid", 91)):
            path = PROCESS / f"splits/seed_{seed}/{name}.csv"
            frame = pd.read_csv(path, usecols=["row_id", "source_row"])
            positions = frame["source_row"].tolist()
            require(len(positions) == expected_count, f"Seed {seed} {name}: count")
            require(frame["row_id"].tolist() == [f"caco2_wang:train_val:{i}" for i in positions], f"Seed {seed} {name}: mapping")
            groups[name] = positions
        require(Counter(groups["train"] + groups["valid"]) == Counter(range(728)), f"Seed {seed}: coverage")
        split_checks[str(seed)] = {"train": 637, "valid": 91, "complete_without_duplicates": True}

    report = {
        "scope": "three baseline representations, all source rows and five official splits",
        "representations": audit,
        "split_checks": split_checks,
        "raw_columns_read": ["Drug"],
        "test_y_values_inspected": False,
        "Y_or_identifiers_in_X": False,
        "rows_dropped": 0,
        "model_training": False,
    }
    output = PROCESS / "qc/baseline_feature_integrity.json"
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Existing feature audit differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(content)
    print("All three representations match Drug-only recalculation in source order")
    for name, result in audit.items():
        print(f"{name}: train_val {result['partitions']['train_val']['rows']}x{result['partitions']['train_val']['columns']}, "
              f"test {result['partitions']['test']['rows']}x{result['partitions']['test']['columns']}")
    print("Five splits cover 728 train_val rows without duplicates; no test labels read")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
