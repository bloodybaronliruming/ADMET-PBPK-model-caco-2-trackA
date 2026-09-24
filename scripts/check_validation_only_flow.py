"""Exercise train-only selection and validation-only tuning without test data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error

from build_morgan_features import ROOT, sha256


RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
FEATURES = PROCESS / "features/morgan_r2_2048"
OUTPUT = ROOT / "caco_trackA_v1/results/split_leakage_smoke"
CONFIGS = (
    {"n_estimators": 16, "max_depth": 12, "min_samples_leaf": 1},
    {"n_estimators": 32, "max_depth": 12, "min_samples_leaf": 2},
)


def main() -> None:
    feature_manifest = json.loads((FEATURES / "features_manifest.json").read_text())
    train_record = feature_manifest["partitions"]["train_val"]
    x_path = ROOT / train_record["matrix_path"]
    if sha256(x_path) != train_record["matrix_sha256"]:
        raise ValueError("train_val feature matrix SHA-256 changed")
    with np.load(x_path, allow_pickle=False) as saved:
        if saved.files != ["X"]:
            raise ValueError("Feature matrix must contain only X")
        x = saved["X"]

    # The fixed test file and its labels are not opened anywhere in this flow.
    train_path = RAW / "caco2_wang_train_val.csv"
    if sha256(train_path) != train_record["raw_sha256"]:
        raise ValueError("train_val source SHA-256 changed")
    y = pd.read_csv(train_path, usecols=["Y"])["Y"].to_numpy(dtype=float)
    if x.shape[0] != len(y) or not np.isfinite(y).all():
        raise ValueError("X/y alignment or labels invalid")

    split_manifest = json.loads((PROCESS / "splits_manifest.json").read_text())
    by_config = [[] for _ in CONFIGS]
    split_details = {}
    for seed in split_manifest["seeds"]:
        groups = {}
        for name in ("train", "valid"):
            path = PROCESS / "splits" / f"seed_{seed}" / f"{name}.csv"
            frame = pd.read_csv(path, usecols=["row_id", "source_row"])
            positions = frame["source_row"].to_numpy(dtype=int)
            if frame["row_id"].tolist() != [
                f"caco2_wang:train_val:{index}" for index in positions
            ]:
                raise ValueError(f"Seed {seed} {name} row identity mismatch")
            groups[name] = positions
        train_idx, valid_idx = groups["train"], groups["valid"]
        if (
            len(set(train_idx) & set(valid_idx)) != 0
            or sorted(np.concatenate((train_idx, valid_idx)).tolist()) != list(range(len(x)))
        ):
            raise ValueError(f"Seed {seed} split coverage or overlap failure")

        selector = VarianceThreshold(threshold=0.0)
        x_train = selector.fit_transform(x[train_idx])
        x_valid = selector.transform(x[valid_idx])
        if x_train.shape[1] == 0 or x_valid.shape[1] != x_train.shape[1]:
            raise ValueError(f"Seed {seed} feature selection failed")
        scores = []
        for config_idx, config in enumerate(CONFIGS):
            model = RandomForestRegressor(
                **config, max_features="sqrt", random_state=seed, n_jobs=1
            )
            model.fit(x_train, y[train_idx])
            prediction = model.predict(x_valid)
            mae = float(mean_absolute_error(y[valid_idx], prediction))
            scores.append(mae)
            by_config[config_idx].append(mae)
        split_details[str(seed)] = {
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "selected_features": x_train.shape[1],
            "validation_mae_by_config": scores,
        }

    mean_mae = [float(np.mean(scores)) for scores in by_config]
    result = {
        "scope": "validation-only leakage workflow check; not a final benchmark result",
        "inputs_opened": [
            train_record["matrix_path"],
            "data/caco_trackA/raw/caco2_wang_train_val.csv",
            "data/caco_trackA/process/splits/seed_{1..5}/{train,valid}.csv",
        ],
        "test_features_used": False,
        "test_labels_read": False,
        "feature_selection_fit_partition": "current seed train only",
        "model_fit_partition": "current seed train only",
        "configuration_selection_metric": "mean validation MAE across five official split seeds",
        "imputer_or_scaler_used": False,
        "early_stopping_used": False,
        "external_features_used": False,
        "model": "scikit-learn RandomForestRegressor, small workflow check",
        "configs": list(CONFIGS),
        "split_details": split_details,
        "mean_validation_mae_by_config": mean_mae,
        "chosen_config_index_for_workflow_check": int(np.argmin(mean_mae)),
        "sklearn_version": sklearn.__version__,
        "final_test_evaluation": "not performed",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "validation_only_flow.json"
    content = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Existing workflow report differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    print(f"Validation-only flow completed for {len(split_details)} splits and {len(CONFIGS)} fixed configurations")
    print("Feature selection fit: current train only; model fit: current train only")
    print("No test data, imputer/scaler, early stopping, or external features used")
    print(f"Report: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
