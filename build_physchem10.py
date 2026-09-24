"""Build frozen, label-free PhysChem-10 matrices in official row order."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
OUTPUT = PROCESS / "features/physchem10"
DATASET = "caco2_wang"

# Order and functions are fixed by endpoint_1_caco_trackA_baseline.md, section 5.
FEATURES = (
    ("MolWt", "Descriptors.MolWt", Descriptors.MolWt),
    ("MolLogP", "Crippen.MolLogP", Crippen.MolLogP),
    ("TPSA", "rdMolDescriptors.CalcTPSA", rdMolDescriptors.CalcTPSA),
    ("NumHDonors", "rdMolDescriptors.CalcNumHBD", rdMolDescriptors.CalcNumHBD),
    ("NumHAcceptors", "rdMolDescriptors.CalcNumHBA", rdMolDescriptors.CalcNumHBA),
    ("NumRotatableBonds", "Descriptors.NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", "rdMolDescriptors.CalcNumRings", rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", "rdMolDescriptors.CalcNumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", "rdMolDescriptors.CalcFractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("FormalCharge", "Chem.GetFormalCharge", Chem.GetFormalCharge),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inputs(partition: str, manifest: dict) -> tuple[list[str], dict]:
    record = manifest["files"][partition]
    raw_path = ROOT / record["path"]
    if sha256(raw_path) != record["sha256"]:
        raise ValueError(f"Raw {partition} SHA-256 changed")

    # Only molecular structure is read. Test Y and identifiers never enter X.
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


def matrix_from_smiles(smiles: list[str], partition: str) -> np.ndarray:
    matrix = np.empty((len(smiles), len(FEATURES)), dtype=np.float64)
    for row_index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"{partition} row {row_index}: unparseable SMILES")
        for feature_index, (name, _, function) in enumerate(FEATURES):
            try:
                matrix[row_index, feature_index] = float(function(mol))
            except Exception as error:
                raise ValueError(
                    f"{partition} row {row_index}: failed to calculate {name}"
                ) from error
    return matrix


def write_or_verify_matrix(path: Path, matrix: np.ndarray) -> str:
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.shape != matrix.shape or not np.array_equal(existing, matrix, equal_nan=True):
            raise FileExistsError(f"Existing frozen feature matrix differs: {path}")
    else:
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
    return sha256(path)


def main() -> None:
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    if raw_manifest["benchmark_name"] != DATASET:
        raise ValueError("Unexpected benchmark name")
    if len(FEATURES) != 10 or len({name for name, _, _ in FEATURES}) != 10:
        raise ValueError("PhysChem-10 feature definition is invalid")

    partitions = {}
    matrices = {}
    for partition in ("train_val", "test"):
        smiles, source = source_inputs(partition, raw_manifest)
        matrix = matrix_from_smiles(smiles, partition)
        if matrix.shape != (source["rows"], 10):
            raise ValueError(f"Unexpected {partition} matrix shape")
        if not np.isfinite(matrix).all():
            bad_rows = np.where(~np.isfinite(matrix).all(axis=1))[0].tolist()
            raise ValueError(f"{partition}: non-finite descriptor values at rows {bad_rows}")
        matrices[partition] = matrix
        path = OUTPUT / f"{partition}.npy"
        source["matrix_path"] = path.relative_to(ROOT).as_posix()
        source["matrix_sha256"] = write_or_verify_matrix(path, matrix)
        source["matrix_shape"] = list(matrix.shape)
        source["matrix_dtype"] = str(matrix.dtype)
        source["missing_count_by_feature"] = {
            name: int(np.isnan(matrix[:, index]).sum())
            for index, (name, _, _) in enumerate(FEATURES)
        }
        source["non_finite_count_by_feature"] = {
            name: int((~np.isfinite(matrix[:, index])).sum())
            for index, (name, _, _) in enumerate(FEATURES)
        }
        partitions[partition] = source

    manifest = {
        "dataset": DATASET,
        "representation": "PhysChem-10",
        "definition_source": "endpoint_1_caco_trackA_baseline.md section 5",
        "input_column": "Drug",
        "input_fields_excluded": ["Y", "Drug_ID", "row_id", "source_partition", "source_row", "seed", "split"],
        "structure_handling": "RDKit Chem.MolFromSmiles on original Drug; no salt removal, neutralization, tautomer or stereochemistry changes",
        "feature_names_in_order": [name for name, _, _ in FEATURES],
        "features": [
            {"feature_order": index + 1, "feature_name": name, "calculation_function": function_name}
            for index, (name, function_name, _) in enumerate(FEATURES)
        ],
        "calculation_script": "caco_trackA_v1/scripts/build_physchem10.py",
        "calculation_script_sha256": sha256(Path(__file__)),
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matrix_format": "NumPy .npy, float64, X only",
        "partitions": partitions,
        "fitted_preprocessing": "none",
        "label_transform": "none; Y is not read by this script",
        "model_training": "not started",
        "freeze_policy": "refuse to overwrite a differing matrix or manifest",
    }
    manifest_path = OUTPUT / "features_manifest.json"
    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if manifest_path.exists() and manifest_path.read_bytes() != content:
        raise FileExistsError(f"Existing frozen feature manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(content)
    print("Built/verified frozen PhysChem-10 matrices:")
    for partition, record in partitions.items():
        print(f"{partition}: {record['matrix_shape']}, SHA-256 {record['matrix_sha256']}")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}, SHA-256 {sha256(manifest_path)}")


if __name__ == "__main__":
    main()
