"""Check frozen XGBoost candidate compatibility without reading benchmark labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xgboost

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from admet_pbpk.baselines import make_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-fit", action="store_true", help="Check parameters only when CUDA is unavailable")
    args = parser.parse_args()
    candidate_path = ROOT / "configs/baselines/search_space/xgb_candidates.json"
    algorithm_path = ROOT / "configs/baselines/algorithms.json"
    experiment_path = ROOT / "configs/baselines/experiments.json"
    candidate_bytes = candidate_path.read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    candidates = json.loads(candidate_bytes)["candidates"]
    algorithms = json.loads(algorithm_path.read_text())["algorithms"]
    experiments = [e for e in json.loads(experiment_path.read_text())["experiments"]
                   if e["algorithm"] == "xgb"]
    if len(candidates) != 24 or len(experiments) != 3:
        raise ValueError("Expected 24 frozen candidates and three XGBoost experiments")
    if candidate_sha != algorithms["xgb"]["candidate_sha256"]:
        raise ValueError("Frozen XGBoost candidate SHA-256 mismatch")
    expected_ids = {"B14_PC10_XGB", "B24_RDKIT2D_XGB", "B33_MORGAN_XGB"}
    if {e["experiment_id"] for e in experiments} != expected_ids:
        raise ValueError("Unexpected XGBoost experiment set")
    checked = 0
    for experiment in experiments:
        for candidate in candidates:
            for seed in range(1, 6):
                pipeline = make_pipeline(experiment, algorithms["xgb"], candidate["params"], seed)
                params = pipeline.named_steps["model"].get_params()
                for key, value in candidate["params"].items():
                    if key not in params or params[key] != value:
                        raise ValueError(f"Unsupported or changed parameter: {key}")
                if params["objective"] != "reg:squarederror" or params["random_state"] != seed:
                    raise ValueError("Fixed objective or random state changed")
                if params["device"] != "cuda" or params["tree_method"] != "hist":
                    raise ValueError("Frozen baseline requires CUDA histogram training")
                checked += 1
    actual_device = None
    if not args.skip_fit:
        # One representative fit catches interface/runtime failures; no benchmark data are opened.
        sample = np.arange(100, dtype=np.float32).reshape(10, 10)
        target = np.arange(10, dtype=np.float32)
        smoke = make_pipeline(experiments[0], algorithms["xgb"], candidates[0]["params"], 1)
        smoke.fit(sample, target)
        predictions = smoke.predict(sample)
        if predictions.shape != (10,) or not np.isfinite(predictions).all():
            raise ValueError("Representative XGBoost fit/predict failed")
        config = json.loads(smoke.named_steps["model"].get_booster().save_config())
        actual_device = config["learner"]["generic_param"]["device"]
        if not actual_device.startswith("cuda"):
            raise ValueError(f"XGBoost silently fell back to {actual_device}")
    report = {
        "xgboost_version": xgboost.__version__,
        "candidate_file": str(candidate_path.relative_to(ROOT)),
        "candidate_sha256": candidate_sha,
        "candidate_count": len(candidates),
        "experiment_ids": [e["experiment_id"] for e in experiments],
        "split_seeds": [1, 2, 3, 4, 5],
        "pipeline_parameter_combinations_checked": checked,
        "representative_fit_predict": "skipped" if args.skip_fit else "passed",
        "device_for_frozen_baseline": "cuda",
        "actual_fit_device": actual_device,
        "benchmark_data_opened": False,
        "formal_validation_completed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
