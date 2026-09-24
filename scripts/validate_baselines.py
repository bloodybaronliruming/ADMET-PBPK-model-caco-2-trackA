"""Run official-split baseline validation without opening the fixed test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from admet_pbpk.baselines import make_pipeline, require_xgb_cuda  # noqa: E402

CONFIG = ROOT / "configs/baselines"
PROCESS = ROOT / "data/caco_trackA/process"
TRAIN_VAL = ROOT / "data/caco_trackA/raw/caco2_wang_train_val.csv"
RESULTS = ROOT / "caco_trackA_v1/results/baselines"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_setup() -> tuple[dict, dict, dict, dict]:
    index = load_json(CONFIG / "experiments_manifest.json")
    assert index["experiment_count"] == 16
    for name, expected in index["files"].items():
        if sha256(CONFIG / name) != expected:
            raise ValueError(f"Frozen config changed: {name}")
    reps = load_json(CONFIG / "representations.json")["representations"]
    algorithms = load_json(CONFIG / "algorithms.json")["algorithms"]
    experiments = load_json(CONFIG / "experiments.json")
    candidates_index = load_json(CONFIG / "search_space/candidates_manifest.json")
    if [e["experiment_id"] for e in experiments["experiments"]] != [
        "B00_NULL_MEAN", "B01_NULL_MEDIAN", "B10_PC10_RIDGE", "B11_PC10_SVR",
        "B12_PC10_RF", "B13_PC10_ET", "B14_PC10_XGB", "B20_RDKIT2D_RIDGE",
        "B21_RDKIT2D_SVR", "B22_RDKIT2D_RF", "B23_RDKIT2D_ET", "B24_RDKIT2D_XGB",
        "B30_MORGAN_SVR", "B31_MORGAN_RF", "B32_MORGAN_ET", "B33_MORGAN_XGB",
    ]:
        raise ValueError("Experiment set/order changed")
    return reps, algorithms, experiments, candidates_index


def load_inputs(representations: dict) -> tuple[np.ndarray, dict, dict, dict]:
    split_manifest = load_json(PROCESS / "splits_manifest.json")
    if split_manifest["seeds"] != [1, 2, 3, 4, 5]:
        raise ValueError("Official split seeds changed")
    if sha256(TRAIN_VAL) != split_manifest["source_sha256"]["train_val"]:
        raise ValueError("train_val raw file changed")
    labels = pd.read_csv(TRAIN_VAL, usecols=["Y"])["Y"].to_numpy(dtype=np.float64)
    if labels.shape != (728,) or not np.isfinite(labels).all():
        raise ValueError("Invalid train_val labels")
    row_map = pd.read_csv(PROCESS / "row_ids/train_val.csv", usecols=["row_id", "source_row"])
    expected_ids = [f"caco2_wang:train_val:{i}" for i in range(728)]
    if row_map["row_id"].tolist() != expected_ids or row_map["source_row"].tolist() != list(range(728)):
        raise ValueError("train_val row mapping changed")
    matrices = {}
    input_files = [str(TRAIN_VAL.relative_to(ROOT)),
                   "data/caco_trackA/process/row_ids/train_val.csv",
                   "data/caco_trackA/process/splits_manifest.json"]
    for key, rep in representations.items():
        manifest_path = ROOT / rep["feature_manifest"]
        if sha256(manifest_path) != rep["feature_manifest_sha256"]:
            raise ValueError(f"{key}: feature manifest changed")
        manifest = load_json(manifest_path)
        source = manifest["partitions"]["train_val"]
        if source["matrix_sha256"] != rep["train_val_matrix_sha256"]:
            raise ValueError(f"{key}: matrix manifest disagrees")
        path = ROOT / rep["train_val_matrix"]
        if sha256(path) != rep["train_val_matrix_sha256"]:
            raise ValueError(f"{key}: matrix changed")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != ["X"]:
                    raise ValueError(f"{key}: unexpected matrix keys")
                matrix = archive["X"]
        else:
            matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (728, rep["feature_count"]) or np.isinf(matrix).any():
            raise ValueError(f"{key}: matrix shape/values")
        matrices[key] = matrix
        input_files.extend([rep["feature_manifest"], rep["train_val_matrix"]])
    splits = {}
    for seed in split_manifest["seeds"]:
        groups = {}
        for part, count in (("train", 637), ("valid", 91)):
            path = PROCESS / f"splits/seed_{seed}/{part}.csv"
            frame = pd.read_csv(path, usecols=["row_id", "source_row"])
            positions = frame["source_row"].to_numpy(dtype=np.int64)
            if len(positions) != count or frame["row_id"].tolist() != [expected_ids[i] for i in positions]:
                raise ValueError(f"Seed {seed} {part}: row mapping")
            groups[part] = positions
            input_files.append(str(path.relative_to(ROOT)))
        if sorted(np.concatenate([groups["train"], groups["valid"]]).tolist()) != list(range(728)):
            raise ValueError(f"Seed {seed}: split coverage")
        splits[seed] = groups
    return labels, matrices, splits, {"files_opened": input_files,
                                       "test_partition_opened": False,
                                       "test_y_read": False,
                                       "train_val_rows": 728,
                                       "split_counts": {str(s): {p: len(v) for p, v in groups.items()}
                                                        for s, groups in splits.items()}}


def load_candidates(experiment: dict, algorithms: dict, index: dict) -> list[dict]:
    if experiment["algorithm"] == "null":
        return [{"candidate_id": "NULL", "params": {}}]
    key = experiment["algorithm"]
    record = index["files"][key]
    path = ROOT / record["path"]
    if record["path"] != experiment["candidate_file"] or sha256(path) != record["sha256"]:
        raise ValueError("Frozen candidate file mismatch")
    if algorithms[key]["candidate_sha256"] != record["sha256"]:
        raise ValueError("Algorithm candidate SHA mismatch")
    doc = load_json(path)
    if len(doc["candidates"]) != doc["candidate_count"] or not doc["candidates"]:
        raise ValueError("Candidate count mismatch")
    return doc["candidates"]


def complexity_key(key: str, params: dict) -> tuple:
    """Prespecified exact-tie ordering after validation MAE and SD."""
    if key == "ridge":
        return (-params["alpha"],)
    if key == "svr":
        gamma = params["gamma"]
        return (params["C"], -params["epsilon"], 0 if gamma == "scale" else 1, str(gamma))
    if key in ("rf", "et"):
        depth = params["max_depth"]
        return (float("inf") if depth is None else depth,
                -params["min_samples_leaf"], params["n_estimators"], str(params["max_features"]))
    if key == "xgb":
        return (params["max_depth"], -params["min_child_weight"], params["n_estimators"],
                params["learning_rate"], params["subsample"], params["colsample_bytree"], params["reg_lambda"])
    return ()


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Validation output already exists: {path}")
    path.write_bytes(content)


def run_experiment(experiment: dict, algorithms: dict, candidate_index: dict,
                   labels: np.ndarray, matrices: dict, splits: dict) -> None:
    exp_id = experiment["experiment_id"]
    output = RESULTS / exp_id / "validation"
    if output.exists():
        raise FileExistsError(f"Validation directory already exists: {output}")
    candidates = load_candidates(experiment, algorithms, candidate_index)
    matrix = matrices.get(experiment["representation"])
    records = []
    for candidate in candidates:
        for seed, groups in splits.items():
            train_idx, valid_idx = groups["train"], groups["valid"]
            start = time.monotonic()
            if experiment["algorithm"] == "null":
                value = float(np.mean(labels[train_idx])) if experiment["statistic"] == "mean" else float(np.median(labels[train_idx]))
                predicted = np.full(len(valid_idx), value)
            else:
                x_train = matrix[train_idx]
                if experiment["representation"] != "morgan_r2_2048" and np.isnan(x_train).all(axis=0).any():
                    raise ValueError(f"{exp_id} seed {seed}: all-missing training descriptor")
                pipeline = make_pipeline(experiment, algorithms[experiment["algorithm"]], candidate["params"], seed)
                pipeline.fit(x_train, labels[train_idx])
                if experiment["algorithm"] == "xgb":
                    require_xgb_cuda(pipeline)
                predicted = pipeline.predict(matrix[valid_idx])
            truth = labels[valid_idx]
            records.append({
                "experiment_id": exp_id, "representation": experiment["representation"] or "none",
                "algorithm": experiment["algorithm"], "candidate_id": candidate["candidate_id"],
                "split_seed": seed, "train_n": len(train_idx), "valid_n": len(valid_idx),
                "MAE": float(mean_absolute_error(truth, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, predicted))),
                "R2": float(r2_score(truth, predicted)),
                "runtime": time.monotonic() - start, "status": "ok",
            })
        if experiment["algorithm"] == "xgb":
            print(f"{exp_id}: completed {candidate['candidate_id']} on all five splits", flush=True)
    frame = pd.DataFrame.from_records(records)
    summaries = []
    for candidate in candidates:
        rows = frame.loc[frame["candidate_id"] == candidate["candidate_id"], "MAE"]
        if len(rows) != 5:
            raise ValueError("Candidate did not complete all five splits")
        summaries.append((float(rows.mean()), float(rows.std(ddof=1)),
                          complexity_key(experiment["algorithm"], candidate["params"]),
                          candidate["candidate_id"], candidate))
    best = min(summaries, key=lambda item: item[:4])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "candidate_results.csv", frame.to_csv(index=False).encode())
    best_config = {"experiment_id": exp_id, "candidate_id": best[3], "params": best[4]["params"],
                   "selection_rule": "min unrounded mean MAE, then sample SD, then prespecified complexity, then candidate ID",
                   "mean_valid_mae": best[0], "sd_valid_mae": best[1]}
    summary = {"experiment_id": exp_id, "primary_metric": "MAE", "n_splits": 5,
               "best_candidate": best[3], "mean_valid_mae": best[0], "sd_valid_mae": best[1],
               "candidate_count": len(candidates), "validation_rows": len(frame),
               "test_y_read": False, "model_training": True}
    write_new(output / "best_config.json", (json.dumps(best_config, indent=2) + "\n").encode())
    write_new(output / "validation_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(f"{exp_id}: {len(candidates)} candidates x 5 splits; best {best[3]}, mean MAE {best[0]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-inputs", action="store_true")
    group.add_argument("--experiment")
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    reps, algorithms, experiment_doc, candidate_index = load_setup()
    labels, matrices, splits, input_audit = load_inputs(reps)
    if args.audit_inputs:
        for exp in experiment_doc["experiments"]:
            load_candidates(exp, algorithms, candidate_index)
        input_audit["candidate_files_checked"] = len(candidate_index["files"])
        input_audit["experiment_configs_checked"] = 16
        input_audit["model_training"] = False
        if args.audit_report:
            write_new(args.audit_report, (json.dumps(input_audit, indent=2) + "\n").encode())
        print("Input audit passed: 728 train_val rows, five 637/91 splits, 16 configs; fixed test not opened")
        return
    matches = [exp for exp in experiment_doc["experiments"] if exp["experiment_id"] == args.experiment]
    if len(matches) != 1:
        raise ValueError("Unknown experiment ID")
    run_experiment(matches[0], algorithms, candidate_index, labels, matrices, splits)


if __name__ == "__main__":
    main()
