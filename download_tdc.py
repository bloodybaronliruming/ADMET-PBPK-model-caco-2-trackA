"""Acquire and register the unchanged TDC Caco2_Wang benchmark CSV files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUESTED_NAME = "Caco2_Wang"
OFFICIAL_NAME = "caco2_wang"
GROUP_NAME = "admet_group"
PARTITIONS = {"train_val": "caco2_wang_train_val.csv", "test": "caco2_wang_test.csv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_without_overwrite(source: Path, target: Path) -> str:
    source_hash = sha256(source)
    if target.exists():
        if sha256(target) != source_hash:
            raise FileExistsError(f"Existing file differs from official cache: {target}")
        return source_hash
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary file already exists: {temporary}")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != source_hash:
        temporary.unlink()
        raise IOError(f"SHA-256 mismatch while copying {source}")
    os.replace(temporary, target)
    return source_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Caco2_Wang through BenchmarkGroup and register its original CSV files."
    )
    parser.parse_args()

    from tdc.benchmark_group import admet_group
    from tdc.metadata import benchmark2id

    root = Path(__file__).resolve().parents[2]
    raw = root / "data" / "caco_trackA" / "raw"
    cache = raw / "tdc_cache"
    cache.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=str(cache))
    if group.name != GROUP_NAME or OFFICIAL_NAME not in group.dataset_names:
        raise ValueError("Installed PyTDC does not expose the expected ADMET benchmark")
    benchmark = group.get(REQUESTED_NAME)
    if benchmark["name"] != OFFICIAL_NAME:
        raise ValueError(
            f"Requested {REQUESTED_NAME}, but BenchmarkGroup returned {benchmark['name']}"
        )

    records = {}
    for partition, filename in PARTITIONS.items():
        source = cache / GROUP_NAME / OFFICIAL_NAME / f"{partition}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Official cache file is missing: {source}")
        frame = benchmark[partition]
        required = {"Drug", "Y"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{partition} lacks columns: {required - set(frame.columns)}")
        target = raw / filename
        file_hash = copy_without_overwrite(source, target)
        records[partition] = {
            "path": target.relative_to(root).as_posix(),
            "cache_path": source.relative_to(root).as_posix(),
            "sha256": file_hash,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "missing_smiles": int(frame["Drug"].isna().sum()),
            "missing_y": int(frame["Y"].isna().sum()),
            "format": "csv",
        }

    package_source = Path(importlib.metadata.distribution("PyTDC").locate_file(
        "tdc/benchmark_group/base_group.py"
    ))
    archive = cache / f"{GROUP_NAME}.zip"
    source_id = benchmark2id.get(GROUP_NAME)
    manifest = {
        "requested_dataset": REQUESTED_NAME,
        "benchmark_name": benchmark["name"],
        "benchmark_group": group.name,
        "task": "regression",
        "official_metric": "MAE",
        "source": "TDC ADMET Benchmark Group",
        "label_scale_status": "待核实",
        "license_status": "待核实",
        "source_url": (
            f"https://dataverse.harvard.edu/api/access/datafile/{source_id}"
            if source_id is not None else None
        ),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytdc_version": importlib.metadata.version("PyTDC"),
        "pytdc_benchmark_source_sha256": sha256(package_source),
        "source_archive": (
            {"path": archive.relative_to(root).as_posix(), "sha256": sha256(archive)}
            if archive.is_file() else None
        ),
        "files": records,
    }
    manifest_path = raw / "dataset_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark_name", "pytdc_version", "pytdc_benchmark_source_sha256",
            "source_archive", "files"
        ):
            if existing.get(key) != manifest[key]:
                raise FileExistsError(f"Existing manifest differs: {manifest_path}")
        print(f"Verified existing {manifest_path.relative_to(root)}")
        return

    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary manifest already exists: {temporary}")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Wrote {manifest_path.relative_to(root)}")
    for partition, record in records.items():
        print(f"{partition}: {record['rows']} rows, SHA-256 {record['sha256']}")


if __name__ == "__main__":
    main()
