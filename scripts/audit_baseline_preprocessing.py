"""Check train-only fitted preprocessing on official split 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from validate_baselines import ROOT, load_candidates, load_inputs, load_setup
from admet_pbpk.baselines import make_pipeline


def main() -> None:
    reps, algorithms, config, candidate_index = load_setup()
    _, matrices, splits, input_audit = load_inputs(reps)
    train_idx = splits[1]["train"]
    valid_idx = splits[1]["valid"]
    results = {}
    variants = ["B10_PC10_RIDGE", "B12_PC10_RF", "B20_RDKIT2D_RIDGE",
                "B22_RDKIT2D_RF", "B30_MORGAN_SVR", "B31_MORGAN_RF"]
    by_id = {item["experiment_id"]: item for item in config["experiments"]}
    if len(by_id) != 16:
        raise ValueError("Expected 16 distinct experiments")
    for exp in by_id.values():
        if exp["algorithm"] == "null":
            expected_steps = []
        elif exp["representation"] == "morgan_r2_2048":
            expected_steps = ["variance_filter"]
        else:
            expected_steps = ["median_imputer", "variance_filter"]
            if exp["algorithm"] in ("ridge", "svr"):
                expected_steps.append("standard_scaler")
        if exp["preprocessing"] != expected_steps:
            raise ValueError(f"{exp['experiment_id']}: preprocessing differs from baseline plan")
    for exp_id in variants:
        exp = by_id[exp_id]
        x_train = matrices[exp["representation"]][train_idx]
        x_valid = matrices[exp["representation"]][valid_idx]
        candidate = load_candidates(exp, algorithms, candidate_index)[0]
        pipeline = make_pipeline(exp, algorithms[exp["algorithm"]], candidate["params"], 1)[:-1]
        train_out = pipeline.fit_transform(x_train)
        valid_out = pipeline.transform(x_valid)
        if len(train_out) != 637 or len(valid_out) != 91:
            raise ValueError(f"{exp_id}: transformed row counts changed")
        if not np.isfinite(train_out).all() or not np.isfinite(valid_out).all():
            raise ValueError(f"{exp_id}: nonfinite transformed value")
        if "median_imputer" in pipeline.named_steps:
            actual = pipeline.named_steps["median_imputer"].statistics_
            expected = np.nanmedian(x_train, axis=0)
            if not np.allclose(actual, expected, equal_nan=True):
                raise ValueError(f"{exp_id}: imputer not fitted on train rows")
            before_filter = pipeline.named_steps["median_imputer"].transform(x_train)
        else:
            before_filter = x_train
        variance = pipeline.named_steps["variance_filter"]
        if not np.array_equal(variance.get_support(), np.ptp(before_filter, axis=0) > 0):
            raise ValueError(f"{exp_id}: variance filter not fitted on train rows")
        if "standard_scaler" in pipeline.named_steps:
            expected_mean = before_filter[:, variance.get_support()].mean(axis=0)
            if not np.allclose(pipeline.named_steps["standard_scaler"].mean_, expected_mean):
                raise ValueError(f"{exp_id}: scaler not fitted on train rows")
        results[exp_id] = {"train_rows": 637, "valid_rows": 91,
                           "input_features": x_train.shape[1], "output_features": train_out.shape[1],
                           "steps": exp["preprocessing"], "fit_scope": "official seed 1 train only",
                           "valid_transform_only": True}
    report = {"dataset": "caco2_wang", "split_seed": 1, "variants": results,
              "all_16_preprocessing_specs_checked": True,
              "test_partition_opened": input_audit["test_partition_opened"],
              "test_y_read": input_audit["test_y_read"], "model_training": False}
    output = ROOT / "data/caco_trackA/process/qc/baseline_preprocessing_fit.json"
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Existing preprocessing audit differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(content)
    print(f"Train-only preprocessing audit passed for {len(results)} pipeline variants; test not opened")


if __name__ == "__main__":
    main()
