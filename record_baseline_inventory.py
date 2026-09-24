"""Record the current baseline data, environment, source and seed inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "PyTDC": "PyTDC",
    "RDKit": "rdkit",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "xgboost": "xgboost",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    raw_manifest_path = ROOT / "data/caco_trackA/raw/dataset_manifest.json"
    split_manifest_path = ROOT / "data/caco_trackA/process/splits_manifest.json"
    baseline_plan_path = ROOT / "endpoint_1_caco_trackA_baseline.md"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    split_manifest = json.loads(split_manifest_path.read_text())
    expected_seeds = [1, 2, 3, 4, 5]
    if split_manifest["seeds"] != expected_seeds:
        raise ValueError("Official split seeds differ from baseline plan")
    if raw_manifest["benchmark_name"] != "caco2_wang":
        raise ValueError("Unexpected benchmark")

    sources = {}
    for partition in ("train_val", "test"):
        record = raw_manifest["files"][partition]
        file = ROOT / record["path"]
        if sha256(file) != record["sha256"]:
            raise ValueError(f"Raw {partition} checksum mismatch")
        sources[partition] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "rows": record["rows"],
        }
    if (sources["train_val"]["rows"], sources["test"]["rows"]) != (728, 182):
        raise ValueError("Unexpected raw row count")

    package_versions = {label: package_version(name) for label, name in PACKAGES.items()}
    status = git("status", "--porcelain=v1")
    report = {
        "scope": "baseline checklist 0.1 data, environment, git and seed registration",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "admetpbpk; captured from active Python interpreter",
        "python_executable": __import__("sys").executable,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "xgboost_installation_status": "installed" if package_versions["xgboost"] else "not installed",
        "dataset": {
            "benchmark_name": raw_manifest["benchmark_name"],
            "benchmark_group": raw_manifest["benchmark_group"],
            "pytdc_version_at_download": raw_manifest["pytdc_version"],
            "manifest_path": str(raw_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(raw_manifest_path),
            "source_archive_sha256": raw_manifest["source_archive"]["sha256"],
            "partitions": sources,
            "splits_manifest_path": str(split_manifest_path.relative_to(ROOT)),
            "splits_manifest_sha256": sha256(split_manifest_path),
        },
        "git": {
            "head_commit_sha": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_clean": status == "",
            "working_tree_status_porcelain_v1": status.splitlines(),
            "note": "HEAD identifies the last commit; uncommitted and untracked files are not part of that commit.",
        },
        "seeds": {
            "split_seeds": expected_seeds,
            "validation_random_state_policy": "split_seed",
            "full_data_model_seeds": expected_seeds,
            "search_seed": 20260924,
            "primary_test_random_state_policy": "split_seed for stochastic models",
        },
        "baseline_plan_path": str(baseline_plan_path.relative_to(ROOT)),
        "baseline_plan_sha256_at_inventory": sha256(baseline_plan_path),
        "models_trained_by_inventory": False,
        "test_y_values_inspected": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded benchmark, package versions, Git HEAD/status and seeds: {output.relative_to(ROOT)}")
    print(f"XGBoost: {report['xgboost_installation_status']}")
    print(f"Git working tree clean: {report['git']['working_tree_clean']}")


if __name__ == "__main__":
    main()
