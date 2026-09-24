"""Summarize all 16 frozen-candidate validations after every run completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/baselines"
RESULTS = ROOT / "caco_trackA_v1/results/baselines"
SEEDS = (1, 2, 3, 4, 5)
FIELDS = ["experiment_id", "representation", "algorithm", "best_candidate_id",
          "candidate_count", "MAE_valid_1", "MAE_valid_2", "MAE_valid_3",
          "MAE_valid_4", "MAE_valid_5", "MAE_mean", "MAE_SD",
          "label_scale", "candidate_results_sha256"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def setup() -> tuple[list[dict], dict, dict]:
    manifest = read_json(CONFIG / "experiments_manifest.json")
    require(manifest["experiment_count"] == 16, "Unexpected experiment count")
    for name, expected_hash in manifest["files"].items():
        require(sha256(CONFIG / name) == expected_hash, f"Frozen config changed: {name}")
    experiments_doc = read_json(CONFIG / "experiments.json")
    require(experiments_doc["label"] == "original TDC Y; no transform", "Label definition changed")
    experiments = experiments_doc["experiments"]
    require(len(experiments) == 16 and len({e["experiment_id"] for e in experiments}) == 16,
            "Expected 16 distinct experiments")
    return experiments, read_json(CONFIG / "search_space/candidates_manifest.json"), manifest


def candidate_ids(experiment: dict, candidate_manifest: dict) -> set[str]:
    if experiment["algorithm"] == "null":
        return {"NULL"}
    record = candidate_manifest["files"][experiment["algorithm"]]
    path = ROOT / record["path"]
    require(record["path"] == experiment["candidate_file"]
            and record["sha256"] == experiment["candidate_sha256"]
            and sha256(path) == record["sha256"],
            f"{experiment['experiment_id']}: candidate file changed")
    document = read_json(path)
    identifiers = [entry["candidate_id"] for entry in document["candidates"]]
    require(len(identifiers) == len(set(identifiers)) == record["candidate_count"],
            f"{experiment['experiment_id']}: candidate list/count changed")
    return set(identifiers)


def summarize_one(experiment: dict, candidates: set[str]) -> dict:
    exp_id = experiment["experiment_id"]
    directory = RESULTS / exp_id / "validation"
    csv_path = directory / "candidate_results.csv"
    best_path = directory / "best_config.json"
    summary_path = directory / "validation_summary.json"
    require(all(path.is_file() for path in (csv_path, best_path, summary_path)),
            f"{exp_id}: incomplete validation output")
    best_saved = read_json(best_path)
    summary_saved = read_json(summary_path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 5 * len(candidates), f"{exp_id}: unexpected validation row count")
    by_candidate: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        candidate = row["candidate_id"]
        seed = int(row["split_seed"])
        require(candidate in candidates and seed in SEEDS and seed not in by_candidate[candidate],
                f"{exp_id}: candidate/split missing or repeated")
        require(row["experiment_id"] == exp_id
                and row["representation"] == (experiment["representation"] or "none")
                and row["algorithm"] == experiment["algorithm"]
                and row["status"] == "ok"
                and int(row["train_n"]) == 637 and int(row["valid_n"]) == 91,
                f"{exp_id}: identity, row count or status changed")
        value = float(row["MAE"])
        require(math.isfinite(value) and value >= 0, f"{exp_id}: invalid MAE")
        by_candidate[candidate][seed] = value
    require(set(by_candidate) == candidates
            and all(set(seed_values) == set(SEEDS) for seed_values in by_candidate.values()),
            f"{exp_id}: incomplete candidate coverage")
    means = {candidate: statistics.mean(values.values())
             for candidate, values in by_candidate.items()}
    minimum = min(means.values())
    winners = [candidate for candidate, value in means.items() if value == minimum]
    require(len(winners) == 1, f"{exp_id}: exact MAE tie requires separate tie-break review")
    winner = winners[0]
    selected = by_candidate[winner]
    mean = statistics.mean(selected.values())
    sd = statistics.stdev(selected.values())
    close = lambda actual, expected: math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
    require(best_saved["candidate_id"] == winner
            and close(best_saved["mean_valid_mae"], mean)
            and close(best_saved["sd_valid_mae"], sd)
            and summary_saved["best_candidate"] == winner
            and summary_saved["candidate_count"] == len(candidates)
            and summary_saved["n_splits"] == 5
            and summary_saved["test_y_read"] is False
            and close(summary_saved["mean_valid_mae"], mean)
            and close(summary_saved["sd_valid_mae"], sd),
            f"{exp_id}: saved selection/summary differs from source rows")
    return {"experiment_id": exp_id,
            "representation": experiment["representation"] or "none",
            "algorithm": experiment["algorithm"],
            "best_candidate_id": winner, "candidate_count": len(candidates),
            **{f"MAE_valid_{seed}": selected[seed] for seed in SEEDS},
            "MAE_mean": mean, "MAE_SD": sd,
            "label_scale": "original TDC Y",
            "candidate_results_sha256": sha256(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-readiness", action="store_true",
                      help="Audit existing runs and report missing experiments without writing a matrix")
    mode.add_argument("--write-validation-matrix", action="store_true",
                      help="Write the official matrix only after all 16 validations exist")
    parser.add_argument("--readiness-report", type=Path)
    args = parser.parse_args()
    experiments, candidate_manifest, config_manifest = setup()
    rows = []
    pending = []
    for experiment in experiments:
        exp_id = experiment["experiment_id"]
        if not (RESULTS / exp_id / "validation").exists():
            pending.append(exp_id)
            continue
        rows.append(summarize_one(experiment, candidate_ids(experiment, candidate_manifest)))
    readiness = {"planned_experiments": 16, "completed_and_audited": len(rows),
                 "pending_experiments": pending, "test_y_read": False,
                 "matrix_written": False}
    if args.check_readiness:
        if args.readiness_report:
            path = args.readiness_report
            content = (json.dumps(readiness, ensure_ascii=False, indent=2) + "\n").encode()
            if path.exists():
                raise FileExistsError(f"Readiness output exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(f"Readiness: {len(rows)}/16 audited; pending: {', '.join(pending) if pending else 'none'}")
        return
    require(len(rows) == 16 and not pending, "All 16 validations are required before writing the matrix")
    matrix_path = RESULTS / "baseline_validation_matrix.csv"
    manifest_path = RESULTS / "baseline_validation_matrix_manifest.json"
    if matrix_path.exists() or manifest_path.exists():
        raise FileExistsError("Official validation matrix output already exists")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode()
    manifest = {"dataset": "caco2_wang", "model_version": "caco_trackA_v1",
                "scope": "all 16 core baseline validations", "rows": 16,
                "split_seeds": list(SEEDS), "primary_metric": "MAE",
                "label_scale": "original TDC Y", "sd_definition": "sample SD, ddof=1",
                "matrix_sha256": hashlib.sha256(content).hexdigest(),
                "frozen_experiments_manifest_sha256": sha256(CONFIG / "experiments_manifest.json"),
                "frozen_config_hashes": config_manifest["files"],
                "test_y_read": False, "final_test_results_included": False}
    matrix_path.write_bytes(content)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote complete 16-row validation matrix: {matrix_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
