"""Independently audit saved core baseline validation results; no data labels read."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/baselines"
RESULTS = ROOT / "caco_trackA_v1/results/baselines"
RUNS = ROOT / "progress/runs"
SEEDS = {1, 2, 3, 4, 5}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def matching_successful_runs(experiment_id: str) -> list[str]:
    matches = []
    for folder in RUNS.glob("caco_trackA_baseline_validation_*"):
        command = folder / "command.txt"
        status = folder / "exit_status.txt"
        if not command.is_file() or not status.is_file():
            continue
        if f"--experiment {experiment_id}" in command.read_text() and status.read_text().strip() == "0":
            matches.append(str(folder.relative_to(ROOT)))
    require(bool(matches), f"{experiment_id}: no successful run log")
    return sorted(matches)


def main() -> None:
    frozen_index = read_json(CONFIG / "experiments_manifest.json")
    for name, expected_hash in frozen_index["files"].items():
        require(sha256(CONFIG / name) == expected_hash, f"Frozen config changed: {name}")
    experiments_doc = read_json(CONFIG / "experiments.json")
    candidate_index = read_json(CONFIG / "search_space/candidates_manifest.json")
    experiments = experiments_doc["experiments"]
    require(len(experiments) == 16 and len({x["experiment_id"] for x in experiments}) == 16,
            "Expected 16 distinct planned experiments")
    audited = {}
    missing = []
    for experiment in experiments:
        exp_id = experiment["experiment_id"]
        directory = RESULTS / exp_id / "validation"
        if not directory.exists():
            missing.append(exp_id)
            continue
        if experiment["algorithm"] == "null":
            candidates = {"NULL": {}}
            candidate_hash = None
        else:
            record = candidate_index["files"][experiment["algorithm"]]
            candidate_path = ROOT / record["path"]
            require(record["path"] == experiment["candidate_file"]
                    and sha256(candidate_path) == record["sha256"]
                    and record["sha256"] == experiment["candidate_sha256"],
                    f"{exp_id}: candidate file changed")
            document = read_json(candidate_path)
            candidates = {entry["candidate_id"]: entry["params"] for entry in document["candidates"]}
            require(len(candidates) == document["candidate_count"] == record["candidate_count"],
                    f"{exp_id}: frozen candidate count")
            candidate_hash = record["sha256"]
        files = {name: directory / name for name in
                 ("candidate_results.csv", "best_config.json", "validation_summary.json")}
        require(all(path.is_file() for path in files.values()), f"{exp_id}: missing validation file")
        best_saved = read_json(files["best_config.json"])
        summary = read_json(files["validation_summary.json"])
        with files["candidate_results.csv"].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 5 * len(candidates), f"{exp_id}: row count")
        grouped: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            candidate_id = row["candidate_id"]
            seed = int(row["split_seed"])
            require(row["experiment_id"] == exp_id
                    and row["algorithm"] == experiment["algorithm"]
                    and row["representation"] == (experiment["representation"] or "none"),
                    f"{exp_id}: result identity changed")
            require(candidate_id in candidates and seed in SEEDS and seed not in grouped[candidate_id],
                    f"{exp_id}: unknown or repeated candidate/split")
            require(int(row["train_n"]) == 637 and int(row["valid_n"]) == 91
                    and row["status"] == "ok", f"{exp_id}: split size/status")
            for metric in ("MAE", "RMSE", "R2", "runtime"):
                require(math.isfinite(float(row[metric])), f"{exp_id}: nonfinite {metric}")
            require(float(row["MAE"]) >= 0 and float(row["RMSE"]) >= 0
                    and float(row["runtime"]) >= 0, f"{exp_id}: invalid metric/runtime")
            grouped[candidate_id][seed] = float(row["MAE"])
        require(set(grouped) == set(candidates)
                and all(set(values) == SEEDS for values in grouped.values()),
                f"{exp_id}: incomplete candidate/split coverage")
        recomputed = {key: {"mean": statistics.mean(values.values()),
                            "sd": statistics.stdev(values.values())}
                      for key, values in grouped.items()}
        smallest_mean = min(item["mean"] for item in recomputed.values())
        winners = [key for key, item in recomputed.items() if item["mean"] == smallest_mean]
        require(len(winners) == 1, f"{exp_id}: exact MAE tie requires separate tie-break audit")
        winner = winners[0]
        mean = recomputed[winner]["mean"]
        sd = recomputed[winner]["sd"]
        close = lambda actual, expected: math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
        require(best_saved["experiment_id"] == exp_id
                and best_saved["candidate_id"] == winner
                and best_saved["params"] == candidates[winner]
                and close(best_saved["mean_valid_mae"], mean)
                and close(best_saved["sd_valid_mae"], sd),
                f"{exp_id}: saved best configuration disagrees with independent calculation")
        require(summary["experiment_id"] == exp_id and summary["primary_metric"] == "MAE"
                and summary["n_splits"] == 5 and summary["best_candidate"] == winner
                and summary["candidate_count"] == len(candidates)
                and summary["validation_rows"] == len(rows)
                and summary["test_y_read"] is False
                and close(summary["mean_valid_mae"], mean)
                and close(summary["sd_valid_mae"], sd),
                f"{exp_id}: summary disagrees with independent calculation")
        audited[exp_id] = {
            "candidate_count": len(candidates), "validation_rows": len(rows),
            "seeds_per_candidate": sorted(SEEDS), "independently_selected_candidate": winner,
            "mean_valid_MAE": mean, "sd_valid_MAE": sd,
            "candidate_file_sha256": candidate_hash,
            "validation_file_sha256": {name: sha256(path) for name, path in files.items()},
            "successful_run_logs": matching_successful_runs(exp_id),
        }
    require((len(audited), len(missing)) in {(13, 3), (16, 0)},
            f"Expected 13/16 or 16/16 completed experiments; got {len(audited)}, {len(missing)}")
    if missing:
        require(set(missing) == {"B14_PC10_XGB", "B24_RDKIT2D_XGB", "B33_MORGAN_XGB"},
                f"Unexpected pending experiments: {missing}")
    report = {
        "scope": "saved validation results only; no model refit or label file read",
        "completed_experiments": len(audited), "planned_experiments": 16,
        "pending_experiments": missing, "audited": audited,
        "test_y_read": False, "test_partition_opened": False,
        "can_mark_full_validation_matrix_complete": not missing,
    }
    output = RESULTS / f"validation_results_audit_{len(audited)}of16.json"
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Existing validation audit differs: {output}")
    if not output.exists():
        output.write_bytes(content)
    print(f"Validation audit passed: {len(audited)} complete experiments; {len(missing)} pending")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
