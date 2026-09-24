"""Audit the 16 frozen final baselines against the Final Benchmark Gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from audit_predictions import audit_evaluated


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "caco_trackA_v1/results/baselines"
PROTOCOLS = ("tdc_compatible_5split", "full_train_refit")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def train_ids_digest(indices: list[int]) -> str:
    payload = "".join(f"caco2_wang:train_val:{i}\n" for i in indices)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_logs() -> dict[tuple[str, str], list[Path]]:
    found: dict[tuple[str, str], dict[str, Path]] = {}
    valid_stages = ("train", "predict", "audit-blind", "evaluate", "audit-evaluated")
    for folder in sorted((ROOT / "progress/runs").glob("caco_trackA_baseline_final_*")):
        command_path = folder / "command.txt"
        if not command_path.is_file():
            continue
        parts = command_path.read_text().strip().split()
        if len(parts) != 8 or parts[:2] != ["bash", "caco_trackA_v1/scripts/run_baseline_final.sh"]:
            continue
        if parts[2] != "--experiment" or parts[4] != "--protocol" or parts[6] != "--stage":
            continue
        key = (parts[3], parts[5])
        stage = parts[7]
        check(stage == "all" or stage in valid_stages, f"Unknown final stage: {key}, {stage}")
        found.setdefault(key, {})
        check(stage not in found[key], f"Duplicate final stage log: {key}, {stage}")
        check((folder / "exit_status.txt").read_text().strip() == "0", f"Failed final run: {key}")
        for line in (folder / "input_config_hashes.txt").read_text().splitlines():
            expected, source = line.split(maxsplit=1)
            check(digest(ROOT / source) == expected, f"Run input changed: {key}, {source}")
        found[key][stage] = folder
    complete = {}
    for key, stages in found.items():
        if "all" in stages:
            check(len(stages) == 1, f"Mixed all/segmented logs: {key}")
            complete[key] = [stages["all"]]
        else:
            check(set(stages) == set(valid_stages), f"Incomplete segmented logs: {key}")
            ordered = [stages[stage] for stage in valid_stages]
            started = [(path / "started_at_utc.txt").read_text().strip() for path in ordered]
            check(started == sorted(started), f"Segmented logs out of order: {key}")
            complete[key] = ordered
    return complete


def main() -> None:
    report_path = RESULTS / "final_benchmark_gate.json"
    check(not report_path.exists(), "Final Benchmark Gate already exists; refusing to overwrite")
    experiments = document(ROOT / "configs/baselines/experiments.json")["experiments"]
    ids = [e["experiment_id"] for e in experiments]
    check(len(ids) == len(set(ids)) == 16, "Expected 16 frozen experiments")
    gate_path = RESULTS / "validation_gate.json"
    gate = document(gate_path)
    check(gate["status"] == "PASS" and len(gate["frozen_configs"]) == 16,
          "Validation Gate incomplete")
    matrix_path = RESULTS / "baseline_final_matrix.csv"
    matrix_manifest_path = RESULTS / "baseline_final_matrix_manifest.json"
    matrix_manifest = document(matrix_manifest_path)
    check(matrix_manifest["matrix_sha256"] == digest(matrix_path)
          and matrix_manifest["validation_gate_sha256"] == digest(gate_path),
          "Final matrix or Validation Gate changed")
    matrix = rows(matrix_path)
    check(len(matrix) == 32, "Final matrix must have 32 rows")
    registry = rows(ROOT / "progress/experiments.csv")
    registry_keys = {(r["experiment_id"], r["benchmark_protocol"]) for r in registry}
    check(len(registry) == len(registry_keys) == 32, "Experiment registry incomplete")
    logs = run_logs()
    check(set(logs) == {(e, p) for e in ids for p in PROTOCOLS},
          "Expected exactly 32 successful logged final protocols")
    splits = {
        seed: [int(r["source_row"]) for r in rows(
            ROOT / f"data/caco_trackA/process/splits/seed_{seed}/train.csv")]
        for seed in range(1, 6)
    }
    check(all(len(v) == len(set(v)) == 637 for v in splits.values()),
          "Official train split count changed")
    split_hash = digest(ROOT / "data/caco_trackA/process/splits_manifest.json")
    raw_hash = digest(ROOT / "data/caco_trackA/raw/caco2_wang_train_val.csv")
    expected_test_ids = [f"caco2_wang:test:{i}" for i in range(182)]
    test_map = rows(ROOT / "data/caco_trackA/process/row_ids/test.csv")
    check([r["row_id"] for r in test_map] == expected_test_ids
          and [int(r["source_row"]) for r in test_map] == list(range(182)),
          "Fixed test row identity/order changed")

    audited = []
    model_artifacts = 0
    null_runs = 0
    run_count = 0
    for index, experiment in enumerate(experiments):
        exp = experiment["experiment_id"]
        frozen_path = RESULTS / exp / "frozen_config.json"
        frozen = document(frozen_path)
        frozen_hash = digest(frozen_path)
        check(gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) == frozen_hash,
              f"{exp}: frozen config changed")
        check(frozen["algorithm"] == experiment["algorithm"]
              and frozen["representation"] == experiment["representation"],
              f"{exp}: frozen algorithm/representation changed")
        for offset, protocol in enumerate(PROTOCOLS):
            key = (exp, protocol)
            # This re-evaluates saved blind/evaluated files against the fixed test
            # without training or changing predictions. Existing audits are write-once.
            audit_evaluated(exp, protocol)
            folder = RESULTS / exp / "final" / protocol
            blind = document(folder / "blind_audit.json")
            evaluation = document(folder / "evaluation_audit.json")
            summary = document(folder / "summary.json")
            check(blind["status"] == evaluation["status"] == "PASS"
                  and blind["test_y_read"] is False
                  and evaluation["test_y_read_only_after_blind_audit"] is True
                  and evaluation["summary"] == summary,
                  f"{key}: audit failed")
            matrix_row = matrix[2 * index + offset]
            check((matrix_row["experiment_id"], matrix_row["benchmark_protocol"]) == key
                  and matrix_row["frozen_config_sha256"] == frozen_hash,
                  f"{key}: final matrix row differs")
            expected_names = ([f"split_seed_{seed}" for seed in range(1, 6)]
                              if protocol == PROTOCOLS[0] else
                              [f"model_seed_{seed}" for seed in range(1, 6)]
                              if experiment["algorithm"] in {"rf", "et", "xgb"} else ["single"])
            check([r["run_name"] for r in evaluation["runs"]] == expected_names
                  and matrix_row["individual_run_ids"].split(";") == expected_names
                  and int(matrix_row["n_runs"]) == len(expected_names),
                  f"{key}: seed/run coverage differs")
            values = []
            for run in evaluation["runs"]:
                name = run["run_name"]
                run_dir = folder / name
                metadata = document(run_dir / "model_metadata.json")
                indices = (splits[int(name.rsplit("_", 1)[1])]
                           if protocol == PROTOCOLS[0] else list(range(728)))
                check(metadata["train_n"] == len(indices)
                      and metadata["train_row_ids_sha256"] == train_ids_digest(indices)
                      and metadata["train_raw_sha256"] == raw_hash
                      and metadata["splits_manifest_sha256"] == split_hash
                      and metadata["frozen_config_sha256"] == frozen_hash
                      and metadata["candidate_id"] == frozen["candidate_id"]
                      and metadata["preprocessing"] == frozen["preprocessing"]
                      and metadata["test_partition_opened"] is False
                      and metadata["test_y_read"] is False,
                      f"{key}/{name}: training boundary mismatch")
                if experiment["algorithm"] == "null":
                    check(metadata["artifact_format"] == "null_constant"
                          and metadata["artifact_path"] is None
                          and metadata["artifact_sha256"] is None
                          and math.isfinite(float(metadata["constant_prediction"])),
                          f"{key}/{name}: null artifact mismatch")
                    null_runs += 1
                else:
                    artifact_path = ROOT / metadata["artifact_path"]
                    check(metadata["artifact_format"] == "joblib"
                          and artifact_path.is_file()
                          and digest(artifact_path) == metadata["artifact_sha256"],
                          f"{key}/{name}: model artifact mismatch")
                    model_artifacts += 1
                if experiment["algorithm"] == "xgb":
                    check(metadata.get("actual_training_device") == "cuda:0",
                          f"{key}/{name}: GPU training not confirmed")
                blind_rows = rows(run_dir / "blind_predictions.csv")
                evaluated_rows = rows(run_dir / "evaluated_predictions.csv")
                check(len(blind_rows) == len(evaluated_rows) == 182
                      and [r["row_id"] for r in blind_rows] == expected_test_ids
                      and [r["row_id"] for r in evaluated_rows] == expected_test_ids
                      and "Y_true" not in blind_rows[0] and "residual" not in blind_rows[0]
                      and "Y_true" in evaluated_rows[0] and "residual" in evaluated_rows[0],
                      f"{key}/{name}: prediction coverage/schema mismatch")
                metrics = document(run_dir / "metrics.json")
                check(metrics["rows"] == 182
                      and math.isclose(float(metrics["MAE"]), float(run["MAE"]), abs_tol=1e-12)
                      and math.isfinite(float(metrics["RMSE"]))
                      and math.isfinite(float(metrics["R2"])),
                      f"{key}/{name}: saved metrics mismatch")
                values.append(float(run["MAE"]))
                run_count += 1
            expected_sd = statistics.stdev(values) if len(values) > 1 else None
            check(math.isclose(statistics.mean(values), float(matrix_row["MAE_mean"]), abs_tol=1e-12)
                  and (matrix_row["MAE_SD"] == "N/A" if expected_sd is None
                       else math.isclose(expected_sd, float(matrix_row["MAE_SD"]), abs_tol=1e-12))
                  and matrix_row["MAE_SD_definition"] == summary["MAE_SD_definition"]
                  and matrix_row["test_rows_per_run"] == "182",
                  f"{key}: matrix aggregate/SD mismatch")
            audited.append({"experiment_id": exp, "benchmark_protocol": protocol,
                            "run_count": len(expected_names),
                            "evaluation_audit_sha256": digest(folder / "evaluation_audit.json"),
                            "run_logs": [str(path.relative_to(ROOT)) for path in logs[key]]})
    check(run_count == model_artifacts + null_runs, "Artifact accounting mismatch")
    report = {
        "status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "caco2_wang", "model_version": "caco_trackA_v1",
        "experiments": 16, "protocols": 32, "runs": run_count,
        "model_artifacts_verified": model_artifacts,
        "null_runs_with_explicit_constant_instead_of_artifact": null_runs,
        "test_rows_per_run": 182,
        "validation_gate_sha256": digest(gate_path),
        "validation_matrix_sha256": digest(RESULTS / "baseline_validation_matrix.csv"),
        "final_matrix_sha256": digest(matrix_path),
        "training_boundary_basis": (
            "train_full.py uses audited official split train indices or all 728 train_val rows; "
            "one pipeline.fit uses only the selected training rows. Per-run ordered row-ID hashes "
            "match those inputs. predict_test.py reads test identifiers/features without Y; "
            "evaluate_test.py reads test Y only after the complete blind audit. "
            "All 32 original run logs have exit code 0 and matching input/code hashes."
        ),
        "selection_changed_after_test": False,
        "checklist": {
            "primary_frozen_config": True,
            "primary_train_only_fit": True,
            "primary_five_official_splits_same_test": True,
            "primary_predict_without_test_y": True,
            "primary_182_rows_official_order": True,
            "primary_blind_schema": True,
            "primary_independent_evaluation_alignment": True,
            "primary_run_mae_mean_sd": True,
            "primary_optional_rmse_r2_saved": True,
            "primary_config_hash": True,
            "primary_model_artifact_or_explicit_null_reason": True,
            "secondary_full_train_preprocessing": True,
            "secondary_seed_policy": True,
            "secondary_inference_without_test_y": True,
            "secondary_blind_and_evaluated_files": True,
            "secondary_run_mae_mean_sd_protocol": True,
            "secondary_config_hash_and_artifact_or_reason": True,
        },
        "audited_protocols": audited,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Final Benchmark Gate PASS: 16 experiments, 32 protocols, {run_count} runs")
    print(f"Saved {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
