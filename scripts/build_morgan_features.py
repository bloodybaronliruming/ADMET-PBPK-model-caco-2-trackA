"""Build label-free Morgan fingerprint matrices from preserved TDC SMILES."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
OUTPUT = PROCESS / "features/morgan_r2_2048"
DATASET = "caco2_wang"
RADIUS = 2
FP_SIZE = 2048
INCLUDE_CHIRALITY = False
USE_BOND_TYPES = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inputs(partition: str, manifest: dict) -> tuple[list[str], dict]:
    record = manifest["files"][partition]
    path = ROOT / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Raw {partition} SHA-256 changed")

    # Deliberately select Drug alone. Y and Drug_ID never enter this feature builder.
    smiles = pd.read_csv(path, usecols=["Drug"])["Drug"]
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


def matrix_from_smiles(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=INCLUDE_CHIRALITY,
        useBondTypes=USE_BOND_TYPES,
        countSimulation=False,
    )
    matrix = np.empty((len(smiles), FP_SIZE), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Unparseable source SMILES at row {index}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[index])
    return matrix


def write_or_verify_matrix(path: Path, matrix: np.ndarray) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved.files != ["X"] or not np.array_equal(saved["X"], matrix):
                raise FileExistsError(f"Existing feature matrix differs: {path}")
    else:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=matrix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())
    return sha256(path)


def main() -> None:
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    if raw_manifest["benchmark_name"] != DATASET:
        raise ValueError("Unexpected benchmark name")

    sources = {}
    for partition in ("train_val", "test"):
        smiles, source = source_inputs(partition, raw_manifest)
        matrix = matrix_from_smiles(smiles)
        if matrix.shape != (source["rows"], FP_SIZE):
            raise ValueError(f"Unexpected {partition} feature shape")
        path = OUTPUT / f"{partition}_X.npz"
        source["matrix_path"] = path.relative_to(ROOT).as_posix()
        source["matrix_sha256"] = write_or_verify_matrix(path, matrix)
        source["matrix_shape"] = list(matrix.shape)
        sources[partition] = source

    manifest = {
        "dataset": DATASET,
        "purpose": "deterministic molecular input for leakage audit and future baseline",
        "input_column": "Drug",
        "input_fields_excluded": ["Y", "Drug_ID", "row_id", "source_partition", "source_row", "seed", "split"],
        "output_array_key": "X",
        "feature_names_in_order": [f"morgan_{i:04d}" for i in range(FP_SIZE)],
        "fingerprint": {
            "method": "RDKit Morgan bit vector",
            "radius": RADIUS,
            "fp_size": FP_SIZE,
            "include_chirality": INCLUDE_CHIRALITY,
            "use_bond_types": USE_BOND_TYPES,
            "count_simulation": False,
        },
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "partitions": sources,
        "fitted_preprocessing": "none",
        "label_transform": "none; Y is not read by this script",
        "model_training": "not started",
    }
    path = OUTPUT / "features_manifest.json"
    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Existing feature manifest differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    print("Built/verified label-free Morgan matrices:")
    for partition, record in sources.items():
        print(f"{partition}: {record['matrix_shape']}, {record['matrix_sha256']}")
    print(f"Manifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
