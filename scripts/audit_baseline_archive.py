"""Verify preserved Caco2_Wang files against the official TDC archive."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
EXPECTED_ROWS = {"train_val": 728, "test": 182}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = RAW / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["benchmark_name"] != "caco2_wang":
        raise ValueError("Unexpected benchmark name")
    archive_info = manifest["source_archive"]
    archive_path = ROOT / archive_info["path"]
    if sha256_file(archive_path) != archive_info["sha256"]:
        raise ValueError("Official archive SHA-256 changed")

    partitions = {}
    smiles_by_partition = {}
    with zipfile.ZipFile(archive_path) as archive:
        for partition, expected_rows in EXPECTED_ROWS.items():
            record = manifest["files"][partition]
            raw_path = ROOT / record["path"]
            cache_path = ROOT / record["cache_path"]
            member = f"admet_group/caco2_wang/{partition}.csv"
            zip_sha = hashlib.sha256(archive.read(member)).hexdigest()
            raw_sha = sha256_file(raw_path)
            cache_sha = sha256_file(cache_path)
            if not (raw_sha == cache_sha == zip_sha == record["sha256"]):
                raise ValueError(f"Raw, cache, archive or manifest differ: {partition}")
            with raw_path.open("rb") as stream:
                header = stream.readline().decode("utf-8-sig").strip()
            if header != "Drug_ID,Drug,Y":
                raise ValueError(f"Unexpected raw columns: {partition}")
            # Only the Drug column is loaded; test Y values are not inspected.
            smiles = pd.read_csv(raw_path, usecols=["Drug"])["Drug"]
            if len(smiles) != expected_rows or record["rows"] != expected_rows:
                raise ValueError(f"Unexpected row count: {partition}")
            if smiles.isna().any():
                raise ValueError(f"Missing Drug at {partition}")
            counts = Counter(smiles.tolist())
            smiles_by_partition[partition] = counts
            partitions[partition] = {
                "rows": len(smiles),
                "unique_raw_smiles": len(counts),
                "duplicate_smiles_groups": sum(value > 1 for value in counts.values()),
                "duplicate_extra_rows": sum(value - 1 for value in counts.values()),
                "raw_path": record["path"],
                "raw_sha256": raw_sha,
                "cache_path": record["cache_path"],
                "archive_member": member,
                "sha_matches_cache_and_archive": True,
            }

    total_rows = sum(item["rows"] for item in partitions.values())
    all_smiles = smiles_by_partition["train_val"] + smiles_by_partition["test"]
    unique_smiles = len(all_smiles)
    duplicate_extra_rows = sum(value - 1 for value in all_smiles.values())
    cross_partition_overlap = len(
        smiles_by_partition["train_val"].keys() & smiles_by_partition["test"].keys()
    )
    if (total_rows, unique_smiles, duplicate_extra_rows) != (910, 906, 4):
        raise ValueError("Official row or repeated-SMILES counts changed")

    report = {
        "scope": "baseline checklist 0.1 first item; archive integrity and row count only",
        "benchmark": manifest["benchmark_name"],
        "dataset_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_archive": archive_info,
        "partitions": partitions,
        "total_rows": total_rows,
        "unique_raw_smiles": unique_smiles,
        "duplicate_extra_rows_preserved": duplicate_extra_rows,
        "cross_partition_raw_smiles_overlap": cross_partition_overlap,
        "test_y_values_inspected": False,
        "raw_files_modified": False,
    }
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Verified official raw archive: {partitions['train_val']['rows']} train_val + "
          f"{partitions['test']['rows']} test = {total_rows} rows")
    print(f"Retained {unique_smiles} distinct raw SMILES and {duplicate_extra_rows} repeated rows")
    for partition, item in partitions.items():
        print(f"{partition} SHA-256: {item['raw_sha256']}")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
