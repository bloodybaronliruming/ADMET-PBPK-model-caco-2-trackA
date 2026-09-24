"""Preserve row identity while materializing official Caco2_Wang splits."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import pandas as pd
from tdc.benchmark_group import admet_group


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
DATASET = "caco2_wang"
SEEDS = (1, 2, 3, 4, 5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def main() -> None:
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    source = {}
    for partition in ("train_val", "test"):
        record = raw_manifest["files"][partition]
        path = ROOT / record["path"]
        cache_path = ROOT / record["cache_path"]
        if sha256(path) != record["sha256"] or sha256(cache_path) != record["sha256"]:
            raise ValueError(f"Raw/cache SHA-256 changed: {partition}")
        frame = pd.read_csv(path)
        if len(frame) != record["rows"] or list(frame.columns) != record["columns"]:
            raise ValueError(f"Raw schema changed: {partition}")
        if frame.isna().any().any():
            raise ValueError(f"Raw record has a missing value: {partition}")
        source[partition] = frame

    group = admet_group(path=str(RAW / "tdc_cache"))
    benchmark = group.get("Caco2_Wang")
    if benchmark["name"] != DATASET:
        raise ValueError(f"Unexpected benchmark: {benchmark['name']}")
    for partition in ("train_val", "test"):
        if not benchmark[partition].equals(source[partition]):
            raise ValueError(f"Benchmark differs from preserved CSV: {partition}")

    original = source["train_val"]
    original_tuples = list(original.itertuples(index=False, name=None))
    if len(set(original_tuples)) != len(original_tuples):
        raise ValueError("Full rows are not unique; split row mapping is ambiguous")
    row_lookup = {record: index for index, record in enumerate(original_tuples)}
    pending: dict[Path, bytes] = {}
    split_records = {}

    for partition, frame in source.items():
        row_map = pd.DataFrame({
            "row_id": [f"{DATASET}:{partition}:{i}" for i in range(len(frame))],
            "source_partition": partition,
            "source_row": range(len(frame)),
        })
        pending[PROCESS / "row_ids" / f"{partition}.csv"] = csv_bytes(row_map)

    for seed in SEEDS:
        train, valid = group.get_train_valid_split(
            benchmark=benchmark["name"], split_type="default", seed=seed
        )
        split_frames = {}
        split_ids = {}
        for name, frame in (("train", train), ("valid", valid)):
            if list(frame.columns) != list(original.columns):
                raise ValueError(f"Seed {seed} {name}: columns changed")
            tuples = list(frame.itertuples(index=False, name=None))
            try:
                positions = [row_lookup[record] for record in tuples]
            except KeyError as exc:
                raise ValueError(f"Seed {seed} {name}: record absent from raw") from exc
            split_ids[name] = positions
            result = frame.copy()
            result.insert(0, "row_id", [f"{DATASET}:train_val:{i}" for i in positions])
            result.insert(1, "source_partition", "train_val")
            result.insert(2, "source_row", positions)
            result.insert(3, "seed", seed)
            result.insert(4, "split", name)
            split_frames[name] = result

        combined = split_ids["train"] + split_ids["valid"]
        if Counter(combined) != Counter(range(len(original))):
            raise ValueError(f"Seed {seed}: missing/duplicate train_val row")
        seed_dir = PROCESS / "splits" / f"seed_{seed}"
        for name, frame in split_frames.items():
            pending[seed_dir / f"{name}.csv"] = csv_bytes(frame)
        split_records[str(seed)] = {
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_row_ids_sha256": hashlib.sha256(
                "\n".join(split_frames["train"]["row_id"]).encode()
            ).hexdigest(),
            "valid_row_ids_sha256": hashlib.sha256(
                "\n".join(split_frames["valid"]["row_id"]).encode()
            ).hexdigest(),
        }

    manifest = {
        "dataset": DATASET,
        "source_manifest": "data/caco_trackA/raw/dataset_manifest.json",
        "source_sha256": {part: raw_manifest["files"][part]["sha256"] for part in source},
        "row_id_rule": "caco2_wang:{source_partition}:{zero_based_source_row}",
        "split_method": "PyTDC BenchmarkGroup.get_train_valid_split(default)",
        "seeds": list(SEEDS),
        "python_version": platform.python_version(),
        "pytdc_version": raw_manifest["pytdc_version"],
        "pandas_version": pd.__version__,
        "splits": split_records,
        "fixed_test_rows": len(source["test"]),
    }
    pending[PROCESS / "splits_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()

    for path, content in pending.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite differing derived file: {path}")
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    print(f"Created/verified {len(pending)} derived files; fixed test SHA-256 {sha256(RAW / 'caco2_wang_test.csv')}")
    for seed, record in split_records.items():
        print(f"Seed {seed}: train {record['train_rows']}, valid {record['valid_rows']}, complete source-row coverage")


if __name__ == "__main__":
    main()
