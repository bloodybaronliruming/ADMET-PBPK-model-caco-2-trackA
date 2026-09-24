"""Freeze the 16 core baseline definitions before validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/baselines"
SEARCH = OUT / "search_space"
FEATURES = ROOT / "data/caco_trackA/process/features"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path: Path, value: dict) -> str:
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Frozen configuration differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_index = json.loads((SEARCH / "candidates_manifest.json").read_text())
    assert candidate_index["search_seed"] == 20260924
    representations = {}
    for key, directory, width in (
        ("physchem10", "physchem10", 10),
        ("rdkit2d", "rdkit2d", 210),
        ("morgan_r2_2048", "morgan_r2_2048", 2048),
    ):
        manifest_path = FEATURES / directory / "features_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["feature_names_in_order"]) == width
        assert manifest["dataset"] == "caco2_wang"
        source = manifest["partitions"]["train_val"]
        assert source["rows"] == 728 and source["matrix_shape"] == [728, width]
        representations[key] = {
            "feature_manifest": str(manifest_path.relative_to(ROOT)),
            "feature_manifest_sha256": digest(manifest_path),
            "train_val_matrix": source["matrix_path"],
            "train_val_matrix_sha256": source["matrix_sha256"],
            "feature_count": width,
            "input_column": "Drug",
            "Y_in_X": False,
        }
    rep_doc = {"dataset": "caco2_wang", "representations": representations}
    rep_sha = write_frozen(OUT / "representations.json", rep_doc)

    algorithms = {}
    for key, record in candidate_index["files"].items():
        path = ROOT / record["path"]
        assert digest(path) == record["sha256"]
        candidates = json.loads(path.read_text())
        assert candidates["candidate_count"] == record["candidate_count"]
        algorithms[key] = {
            "estimator": candidates["estimator"],
            "candidate_file": record["path"],
            "candidate_sha256": record["sha256"],
            "candidate_count": record["candidate_count"],
            "fixed_estimator_params": candidates["fixed_estimator_params"],
            "applicable_representations": candidates["applicable_representations"],
        }
    alg_doc = {"dataset": "caco2_wang", "algorithms": algorithms}
    alg_sha = write_frozen(OUT / "algorithms.json", alg_doc)

    experiments = []
    for exp_id, statistic in (("B00_NULL_MEAN", "mean"), ("B01_NULL_MEDIAN", "median")):
        experiments.append({
            "experiment_id": exp_id, "representation": None,
            "algorithm": "null", "statistic": statistic,
            "candidate_file": None, "preprocessing": [],
        })
    for prefix, representation, algorithms_here in (
        ("B1", "physchem10", [("0", "ridge"), ("1", "svr"), ("2", "rf"), ("3", "et"), ("4", "xgb")]),
        ("B2", "rdkit2d", [("0", "ridge"), ("1", "svr"), ("2", "rf"), ("3", "et"), ("4", "xgb")]),
        ("B3", "morgan_r2_2048", [("0", "svr"), ("1", "rf"), ("2", "et"), ("3", "xgb")]),
    ):
        for suffix, algorithm in algorithms_here:
            steps = (["median_imputer"] if representation != "morgan_r2_2048" else []) + ["variance_filter"]
            if algorithm in ("ridge", "svr") and representation != "morgan_r2_2048":
                steps.append("standard_scaler")
            assert representation in algorithms[algorithm]["applicable_representations"]
            experiments.append({
                "experiment_id": f"{prefix}{suffix}_{'PC10' if prefix == 'B1' else 'RDKIT2D' if prefix == 'B2' else 'MORGAN'}_{algorithm.upper()}",
                "representation": representation,
                "algorithm": algorithm,
                "candidate_file": algorithms[algorithm]["candidate_file"],
                "candidate_sha256": algorithms[algorithm]["candidate_sha256"],
                "preprocessing": steps,
            })
    expected_ids = ["B00_NULL_MEAN", "B01_NULL_MEDIAN", "B10_PC10_RIDGE", "B11_PC10_SVR",
                    "B12_PC10_RF", "B13_PC10_ET", "B14_PC10_XGB", "B20_RDKIT2D_RIDGE",
                    "B21_RDKIT2D_SVR", "B22_RDKIT2D_RF", "B23_RDKIT2D_ET", "B24_RDKIT2D_XGB",
                    "B30_MORGAN_SVR", "B31_MORGAN_RF", "B32_MORGAN_ET", "B33_MORGAN_XGB"]
    assert [item["experiment_id"] for item in experiments] == expected_ids
    exp_doc = {
        "dataset": "caco2_wang", "model_version": "caco_trackA_v1",
        "source_plan": "endpoint_1_caco_trackA_baseline.md sections 2-16, 30",
        "label": "original TDC Y; no transform",
        "primary_metric": "MAE on original Y scale",
        "split_seeds": [1, 2, 3, 4, 5],
        "validation_random_state_policy": "split_seed for RF, ET, XGB",
        "preprocessing_fit_scope": "each official train partition only",
        "candidate_selection": "minimum unrounded mean validation MAE",
        "early_stopping": False,
        "experiments": experiments,
    }
    exp_sha = write_frozen(OUT / "experiments.json", exp_doc)
    index = {"dataset": "caco2_wang", "experiment_count": 16,
             "files": {"representations.json": rep_sha,
                       "algorithms.json": alg_sha, "experiments.json": exp_sha},
             "test_labels_read": False, "model_training": False}
    write_frozen(OUT / "experiments_manifest.json", index)
    print(f"Frozen 3 representations, 5 algorithms, 16 experiments in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
