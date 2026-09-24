"""Build label-free RDKit2D matrices from the frozen descriptor manifest."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
OUTPUT = PROCESS / "features/rdkit2d"
DATASET = "caco2_wang"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_descriptors() -> tuple[list[str], list, str]:
    path = OUTPUT / "rdkit2d_descriptor_manifest.json"
    checksum_file = OUTPUT / "rdkit2d_descriptor_manifest.sha256"
    expected = checksum_file.read_text().split()[0]
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Frozen RDKit2D descriptor manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text())
    if manifest["rdkit_version"] != rdBase.rdkitVersion:
        raise ValueError("RDKit version differs from frozen descriptor manifest")
    selected = manifest["selected_descriptors"]
    names = manifest["descriptor_names_in_order"]
    if len(selected) != 210 or len(names) != 210 or [r["name"] for r in selected] != names:
        raise ValueError("Frozen RDKit2D descriptor names or order mismatch")
    if [r["feature_order"] for r in selected] != list(range(1, 211)):
        raise ValueError("Frozen RDKit2D feature order mismatch")
    if hashlib.sha256("\n".join(names).encode()).hexdigest() != manifest["descriptor_names_sha256"]:
        raise ValueError("Frozen RDKit2D descriptor names SHA-256 mismatch")
    functions = []
    for record in selected:
        name = record["name"]
        if record["calculation_function"] != f"rdkit.Chem.Descriptors.{name}":
            raise ValueError(f"Unexpected frozen descriptor function: {name}")
        function = getattr(Descriptors, name, None)
        if not callable(function):
            raise ValueError(f"Frozen descriptor unavailable in RDKit: {name}")
        if getattr(function, "__module__", None) != record["implementation_module"]:
            raise ValueError(f"Frozen descriptor implementation changed: {name}")
        functions.append(function)
    return names, functions, actual


def source_inputs(partition: str, raw_manifest: dict) -> tuple[list[str], dict]:
    record = raw_manifest["files"][partition]
    raw_path = ROOT / record["path"]
    if sha256(raw_path) != record["sha256"]:
        raise ValueError(f"Raw {partition} SHA-256 changed")
    # Only Drug is parsed. Neither identifiers nor labels enter this matrix.
    smiles = pd.read_csv(raw_path, usecols=["Drug"])["Drug"]
    row_map_path = PROCESS / "row_ids" / f"{partition}.csv"
    row_map = pd.read_csv(row_map_path)
    expected_ids = [f"{DATASET}:{partition}:{i}" for i in range(record["rows"])]
    if (
        len(smiles) != record["rows"]
        or smiles.isna().any()
        or row_map["row_id"].tolist() != expected_ids
        or row_map["source_partition"].tolist() != [partition] * record["rows"]
        or row_map["source_row"].tolist() != list(range(record["rows"]))
    ):
        raise ValueError(f"Raw {partition} rows or row_id mapping changed")
    return smiles.tolist(), {
        "rows": record["rows"],
        "raw_sha256": record["sha256"],
        "row_map": row_map_path.relative_to(ROOT).as_posix(),
        "row_map_sha256": sha256(row_map_path),
    }


def calculate(smiles: list[str], names: list[str], functions: list, partition: str) -> tuple[np.ndarray, dict]:
    matrix = np.empty((len(smiles), len(names)), dtype=np.float64)
    nonfinite = {name: {"nan": 0, "positive_inf": 0, "negative_inf": 0} for name in names}
    for row_index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"{partition} row {row_index}: unparseable SMILES")
        for feature_index, (name, function) in enumerate(zip(names, functions)):
            try:
                result = float(function(mol))
            except Exception as error:
                raise ValueError(
                    f"{partition} row {row_index}: descriptor calculation failed for {name}"
                ) from error
            if np.isnan(result):
                nonfinite[name]["nan"] += 1
                result = np.nan
            elif np.isposinf(result):
                nonfinite[name]["positive_inf"] += 1
                result = np.nan
            elif np.isneginf(result):
                nonfinite[name]["negative_inf"] += 1
                result = np.nan
            matrix[row_index, feature_index] = result
    return matrix, nonfinite


def write_or_verify_matrix(path: Path, matrix: np.ndarray) -> str:
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.shape != matrix.shape or not np.array_equal(existing, matrix, equal_nan=True):
            raise FileExistsError(f"Existing RDKit2D matrix differs: {path}")
    else:
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
    return sha256(path)


def main() -> None:
    names, functions, descriptor_manifest_sha = frozen_descriptors()
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    if raw_manifest["benchmark_name"] != DATASET:
        raise ValueError("Unexpected benchmark name")

    partitions = {}
    for partition in ("train_val", "test"):
        smiles, source = source_inputs(partition, raw_manifest)
        matrix, nonfinite = calculate(smiles, names, functions, partition)
        if matrix.shape != (source["rows"], len(names)) or np.isinf(matrix).any():
            raise ValueError(f"Unexpected {partition} RDKit2D matrix schema")
        path = OUTPUT / f"{partition}.npy"
        source.update({
            "matrix_path": path.relative_to(ROOT).as_posix(),
            "matrix_sha256": write_or_verify_matrix(path, matrix),
            "matrix_shape": list(matrix.shape),
            "matrix_dtype": str(matrix.dtype),
            "missing_count_by_feature": {
                name: int(np.isnan(matrix[:, i]).sum()) for i, name in enumerate(names)
            },
            "raw_nonfinite_count_by_feature": nonfinite,
            "rows_with_missing": int(np.isnan(matrix).any(axis=1).sum()),
            "no_rows_dropped": True,
        })
        partitions[partition] = source

    feature_manifest = {
        "dataset": DATASET,
        "representation": "RDKit2D",
        "descriptor_manifest": "data/caco_trackA/process/features/rdkit2d/rdkit2d_descriptor_manifest.json",
        "descriptor_manifest_sha256": descriptor_manifest_sha,
        "feature_names_in_order": names,
        "descriptor_count": len(names),
        "input_column": "Drug",
        "input_fields_excluded": ["Y", "Drug_ID", "row_id", "source_partition", "source_row", "seed", "split"],
        "structure_handling": "RDKit Chem.MolFromSmiles on original Drug; no salt removal, neutralization, tautomer or stereochemistry changes",
        "nonfinite_handling": "NaN and +/-Inf descriptor outputs become NaN; rows retained; imputation only in later train-fitted pipeline",
        "calculation_failure_handling": "raise with partition, row index and descriptor; do not delete rows",
        "calculation_script": "caco_trackA_v1/scripts/build_rdkit2d.py",
        "calculation_script_sha256": sha256(Path(__file__)),
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matrix_format": "NumPy .npy, float64, X only",
        "partitions": partitions,
        "fitted_preprocessing": "none",
        "label_transform": "none; Y is not read by this script",
        "model_training": "not started",
        "freeze_policy": "refuse to overwrite a differing matrix or feature manifest",
    }
    manifest_path = OUTPUT / "features_manifest.json"
    content = (json.dumps(feature_manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if manifest_path.exists() and manifest_path.read_bytes() != content:
        raise FileExistsError(f"Existing RDKit2D feature manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(content)
    print("Built/verified frozen RDKit2D matrices:")
    for partition, record in partitions.items():
        print(f"{partition}: {record['matrix_shape']}, rows with missing {record['rows_with_missing']}, SHA-256 {record['matrix_sha256']}")
    print(f"Feature manifest: {manifest_path.relative_to(ROOT)}, SHA-256 {sha256(manifest_path)}")


if __name__ == "__main__":
    main()
