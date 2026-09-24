"""Independently verify saved train/valid row identities and official order."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tdc.benchmark_group import admet_group


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
SEEDS = (1, 2, 3, 4, 5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


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
    split_manifest_path = PROCESS / "splits_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text())
    raw_path = ROOT / raw_manifest["files"]["train_val"]["path"]
    require(sha256(raw_path) == raw_manifest["files"]["train_val"]["sha256"], "Raw train_val SHA mismatch")
    require(split_manifest["source_sha256"]["train_val"] == sha256(raw_path), "Split source SHA mismatch")
    require(split_manifest["seeds"] == list(SEEDS), "Split seed list mismatch")
    require(split_manifest["dataset"] == "caco2_wang", "Dataset mismatch")
    original = pd.read_csv(raw_path)
    require(len(original) == 728, "Raw train_val row count mismatch")
    columns = list(original.columns)
    require(columns == ["Drug_ID", "Drug", "Y"], "Raw schema mismatch")
    original_records = list(original.itertuples(index=False, name=None))
    require(len(set(original_records)) == len(original_records), "Ambiguous duplicate full rows")
    positions = {record: i for i, record in enumerate(original_records)}

    row_map_path = PROCESS / "row_ids/train_val.csv"
    row_map = pd.read_csv(row_map_path)
    expected_ids = [f"caco2_wang:train_val:{i}" for i in range(len(original))]
    require(row_map["row_id"].tolist() == expected_ids, "Row map IDs/order mismatch")
    require(row_map["source_row"].tolist() == list(range(len(original))), "Row map source order mismatch")
    require(row_map["source_partition"].tolist() == ["train_val"] * len(original), "Row map partition mismatch")

    group = admet_group(path=str(RAW / "tdc_cache"))
    benchmark = group.get("Caco2_Wang")
    require(benchmark["name"] == "caco2_wang", "Official benchmark name mismatch")
    require(benchmark["train_val"].equals(original), "Official train_val differs from raw")

    audited: dict[str, dict] = {}
    for seed in SEEDS:
        official_train, official_valid = group.get_train_valid_split(
            benchmark=benchmark["name"], split_type="default", seed=seed
        )
        split_positions: dict[str, list[int]] = {}
        partition_report: dict[str, dict] = {}
        for name, official in (("train", official_train), ("valid", official_valid)):
            saved_path = PROCESS / f"splits/seed_{seed}/{name}.csv"
            saved = pd.read_csv(saved_path)
            require(len(saved) == (637 if name == "train" else 91), f"Seed {seed} {name}: row count")
            require(list(official.columns) == columns, f"Seed {seed} {name}: official schema")
            official_records = list(official.itertuples(index=False, name=None))
            try:
                official_positions = [positions[record] for record in official_records]
            except KeyError as error:
                raise ValueError(f"Seed {seed} {name}: official row absent from raw") from error
            split_positions[name] = official_positions
            ids = [expected_ids[i] for i in official_positions]
            require(saved["source_row"].tolist() == official_positions, f"Seed {seed} {name}: source row order")
            require(saved["row_id"].tolist() == ids, f"Seed {seed} {name}: row ID order")
            require(saved["source_partition"].tolist() == ["train_val"] * len(saved), f"Seed {seed} {name}: partition")
            require(saved["seed"].tolist() == [seed] * len(saved), f"Seed {seed} {name}: seed field")
            require(saved["split"].tolist() == [name] * len(saved), f"Seed {seed} {name}: split field")
            require(saved[columns].equals(official.reset_index(drop=True)), f"Seed {seed} {name}: official row content/order")
            manifest_record = split_manifest["splits"][str(seed)]
            require(manifest_record[f"{name}_rows"] == len(saved), f"Seed {seed} {name}: manifest count")
            require(manifest_record[f"{name}_row_ids_sha256"] == ids_hash(ids), f"Seed {seed} {name}: manifest ID hash")
            partition_report[name] = {
                "rows": len(saved),
                "saved_csv_sha256": sha256(saved_path),
                "ordered_row_ids_sha256": ids_hash(ids),
                "official_order_and_content_match": True,
            }
        all_positions = split_positions["train"] + split_positions["valid"]
        require(Counter(all_positions) == Counter(range(len(original))), f"Seed {seed}: incomplete or duplicate coverage")
        require(not (set(split_positions["train"]) & set(split_positions["valid"])), f"Seed {seed}: train/valid overlap")
        audited[str(seed)] = {
            **partition_report,
            "complete_728_row_coverage": True,
            "train_valid_overlap": 0,
        }

    report = {
        "scope": "baseline checklist 0.1 official split row identity and order audit",
        "benchmark": "caco2_wang",
        "split_method": "PyTDC BenchmarkGroup.get_train_valid_split(default)",
        "raw_train_val_sha256": sha256(raw_path),
        "row_map_sha256": sha256(row_map_path),
        "splits_manifest_sha256": sha256(split_manifest_path),
        "seeds": list(SEEDS),
        "splits": audited,
        "test_file_opened": False,
        "test_y_values_inspected": False,
        "raw_or_saved_splits_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Verified five official splits: each 637 train + 91 valid; all 728 source rows covered")
    print("Saved row IDs, source positions, full row content and order match PyTDC for seeds 1..5")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
