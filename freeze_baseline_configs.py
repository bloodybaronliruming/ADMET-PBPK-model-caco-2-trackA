"""Freeze all 16 final configurations only after complete validation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn

from validate_baselines import (CONFIG, RESULTS, ROOT, complexity_key,
                                load_candidates, load_inputs, load_json,
                                load_setup, sha256)


def verified_choice(experiment: dict, algorithms: dict, candidate_index: dict) -> tuple[dict, float, float]:
    exp_id = experiment["experiment_id"]
    folder = RESULTS / exp_id / "validation"
    if not all((folder / name).is_file() for name in
               ("candidate_results.csv", "best_config.json", "validation_summary.json")):
        raise FileNotFoundError(f"{exp_id}: incomplete validation")
    candidates = load_candidates(experiment, algorithms, candidate_index)
    # Pandas treats the literal candidate ID "NULL" as NA unless disabled.
    frame = pd.read_csv(folder / "candidate_results.csv", keep_default_na=False)
    summary = load_json(folder / "validation_summary.json")
    best_file = load_json(folder / "best_config.json")
    expected_pairs = {(c["candidate_id"], s) for c in candidates for s in (1, 2, 3, 4, 5)}
    actual_pairs = list(zip(frame["candidate_id"], frame["split_seed"]))
    if (len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs
            or frame["status"].tolist() != ["ok"] * len(frame)
            or frame["train_n"].tolist() != [637] * len(frame)
            or frame["valid_n"].tolist() != [91] * len(frame)
            or frame["experiment_id"].tolist() != [exp_id] * len(frame)
            or not np.isfinite(frame["MAE"]).all()):
        raise ValueError(f"{exp_id}: validation records incomplete or altered")
    ranked = []
    for candidate in candidates:
        values = frame.loc[frame["candidate_id"] == candidate["candidate_id"], "MAE"]
        ranked.append((float(values.mean()), float(values.std(ddof=1)),
                       complexity_key(experiment["algorithm"], candidate["params"]),
                       candidate["candidate_id"], candidate))
    best = min(ranked, key=lambda item: item[:4])
    close = lambda actual, expected: bool(np.isclose(actual, expected, rtol=0, atol=1e-12))
    if (best_file["candidate_id"] != best[3] or best_file["params"] != best[4]["params"]
            or not close(best_file["mean_valid_mae"], best[0])
            or not close(best_file["sd_valid_mae"], best[1])
            or summary["best_candidate"] != best[3]
            or not close(summary["mean_valid_mae"], best[0])
            or not close(summary["sd_valid_mae"], best[1]) or summary["n_splits"] != 5):
        raise ValueError(f"{exp_id}: saved selection differs from unrounded MAE")
    return best[4], best[0], best[1]


def main() -> None:
    reps, algorithms, config, candidate_index = load_setup()
    _, _, _, input_audit = load_inputs(reps)
    if input_audit["test_partition_opened"] or input_audit["test_y_read"]:
        raise ValueError("Development input audit included test")
    raw = load_json(ROOT / "data/caco_trackA/raw/dataset_manifest.json")
    if raw["benchmark_name"] != "caco2_wang" or raw["files"]["test"]["rows"] != 182:
        raise ValueError("Unexpected raw archive")
    test_path = ROOT / raw["files"]["test"]["path"]
    if sha256(test_path) != raw["files"]["test"]["sha256"]:
        raise ValueError("Fixed test file changed")
    selected = {}
    for exp in config["experiments"]:
        selected[exp["experiment_id"]] = verified_choice(exp, algorithms, candidate_index)
    if len(selected) != 16:
        raise ValueError("All 16 validations required")
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError("XGBoost must be installed before configuration freeze") from exc
    versions = {"python": platform.python_version(), "PyTDC": version("PyTDC"),
                "RDKit": rdkit.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__}
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    common = {"dataset": "caco2_wang", "model_version": "caco_trackA_v1",
              "raw_sha256": {name: raw["files"][name]["sha256"] for name in ("train_val", "test")},
              "software_versions": versions, "git_commit": git_head,
              "git_worktree_status": git_status, "frozen_at_utc": timestamp,
              "selection_rule": "minimum unrounded mean validation MAE; then sample SD; then prespecified complexity; then candidate ID",
              "label_scale": "original TDC Y", "early_stopping": False}
    pending = []
    for exp in config["experiments"]:
        exp_id = exp["experiment_id"]
        candidate, mean_mae, sd_mae = selected[exp_id]
        representation = exp["representation"]
        frozen = {**common, "experiment_id": exp_id, "representation": representation,
                  "feature_manifest_sha256": reps[representation]["feature_manifest_sha256"] if representation else None,
                  "algorithm": exp["algorithm"], "candidate_id": candidate["candidate_id"],
                  "hyperparameters": candidate["params"], "preprocessing": exp["preprocessing"],
                  "validation_mean_MAE": mean_mae, "validation_SD_MAE": sd_mae,
                  "validation_candidate_results_sha256": sha256(RESULTS / exp_id / "validation/candidate_results.csv"),
                  "benchmark_protocols": ["tdc_compatible_5split", "full_train_refit"],
                  "status": "CONFIG FROZEN"}
        if exp["algorithm"] == "null":
            frozen["null_statistic"] = exp["statistic"]
        content = (json.dumps(frozen, ensure_ascii=False, indent=2) + "\n").encode()
        destination = RESULTS / exp_id / "frozen_config.json"
        if destination.exists():
            raise FileExistsError(f"Freeze output exists: {destination}")
        pending.append((destination, content))
    gate = {"status": "PASS", "validation_experiments": 16,
            "fixed_test_sha256_verified_without_label_interpretation": True,
            "test_y_used_for_selection": False, "frozen_at_utc": timestamp,
            "frozen_configs": {str(path.relative_to(ROOT)): hashlib.sha256(content).hexdigest()
                               for path, content in pending}}
    gate_path = RESULTS / "validation_gate.json"
    if gate_path.exists():
        raise FileExistsError(f"Validation Gate output exists: {gate_path}")
    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n")
    print("Frozen all 16 final configurations; Validation Gate = PASS")


if __name__ == "__main__":
    main()
