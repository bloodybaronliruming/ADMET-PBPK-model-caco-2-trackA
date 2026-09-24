"""Build the frozen 16 × 2 final baseline matrix from audited results."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "caco_trackA_v1/results/baselines"
PROTOCOLS = ("tdc_compatible_5split", "full_train_refit")
FIELDS = (
    "experiment_id", "representation", "algorithm", "candidate_id",
    "benchmark_protocol", "n_runs", "individual_run_ids", "run_level_MAE",
    "MAE_mean", "MAE_SD", "MAE_SD_definition", "test_rows_per_run",
    "label_scale", "frozen_config_sha256", "blind_audit_sha256",
    "evaluation_audit_sha256", "summary_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def main() -> None:
    experiments_path = ROOT / "configs/baselines/experiments.json"
    gate_path = RESULTS / "validation_gate.json"
    registry_path = ROOT / "progress/experiments.csv"
    validation_path = RESULTS / "baseline_validation_matrix.csv"
    output_path = RESULTS / "baseline_final_matrix.csv"
    manifest_path = RESULTS / "baseline_final_matrix_manifest.json"
    require(not output_path.exists() and not manifest_path.exists(),
            "Final matrix already exists; refusing to overwrite")
    plan = read_json(experiments_path)
    experiments = plan["experiments"]
    require(len(experiments) == 16 and len({e["experiment_id"] for e in experiments}) == 16,
            "Expected 16 unique frozen experiments")
    gate = read_json(gate_path)
    require(gate["status"] == "PASS" and len(gate["frozen_configs"]) == 16,
            "Validation Gate incomplete")
    with registry_path.open(newline="") as stream:
        registered = list(csv.DictReader(stream))
    registry = {(r["experiment_id"], r["benchmark_protocol"]): r for r in registered}
    require(len(registered) == len(registry) == 32, "Expected 32 distinct registry records")
    with validation_path.open(newline="") as stream:
        validated = {r["experiment_id"]: r for r in csv.DictReader(stream)}
    require(len(validated) == 16, "Validation matrix incomplete")

    rows = []
    sources = {}
    for experiment in experiments:
        exp_id = experiment["experiment_id"]
        frozen_path = RESULTS / exp_id / "frozen_config.json"
        frozen = read_json(frozen_path)
        frozen_hash = sha256(frozen_path)
        require(gate["frozen_configs"].get(str(frozen_path.relative_to(ROOT))) == frozen_hash,
                f"{exp_id}: frozen config differs from gate")
        require(frozen["experiment_id"] == exp_id
                and frozen["representation"] == experiment["representation"]
                and frozen["algorithm"] == experiment["algorithm"]
                and frozen["candidate_id"] == validated[exp_id]["best_candidate_id"],
                f"{exp_id}: frozen identity differs from validation")
        sources[str(frozen_path.relative_to(ROOT))] = frozen_hash
        for protocol in PROTOCOLS:
            directory = RESULTS / exp_id / "final" / protocol
            blind_path = directory / "blind_audit.json"
            evaluation_path = directory / "evaluation_audit.json"
            summary_path = directory / "summary.json"
            blind, evaluated, summary = map(read_json, (blind_path, evaluation_path, summary_path))
            require(blind["status"] == evaluated["status"] == "PASS"
                    and blind["test_y_read"] is False
                    and evaluated["test_y_read_only_after_blind_audit"] is True
                    and evaluated["summary"] == summary,
                    f"{exp_id}/{protocol}: audit incomplete")
            require(summary["experiment_id"] == exp_id
                    and summary["benchmark_protocol"] == protocol
                    and summary["frozen_config_sha256"] == frozen_hash
                    and summary["blind_audit_sha256"] == sha256(blind_path)
                    and summary["test_rows_per_run"] == 182
                    and summary["label_scale"] == "original TDC Y"
                    and summary["selection_changed_after_test"] is False,
                    f"{exp_id}/{protocol}: summary identity or boundary mismatch")
            runs = evaluated["runs"]
            expected_names = ([f"split_seed_{seed}" for seed in range(1, 6)]
                              if protocol == PROTOCOLS[0] else
                              [f"model_seed_{seed}" for seed in range(1, 6)]
                              if experiment["algorithm"] in {"rf", "et", "xgb"} else ["single"])
            require([r["run_name"] for r in runs] == expected_names
                    and blind["run_count"] == summary["run_count"] == len(runs)
                    and [r["run_name"] for r in summary["run_level_MAE"]] == expected_names,
                    f"{exp_id}/{protocol}: run identity/count mismatch")
            values = []
            for audit_run, summary_run in zip(runs, summary["run_level_MAE"]):
                run_dir = directory / audit_run["run_name"]
                metadata = read_json(run_dir / "model_metadata.json")
                require(metadata["frozen_config_sha256"] == frozen_hash
                        and metadata["train_n"] == (637 if protocol == PROTOCOLS[0] else 728)
                        and metadata["test_y_read"] is False
                        and audit_run["test_n"] == 182,
                        f"{exp_id}/{protocol}/{audit_run['run_name']}: model boundary mismatch")
                for key, filename in (
                    ("model_metadata_sha256", "model_metadata.json"),
                    ("blind_predictions_sha256", "blind_predictions.csv"),
                    ("evaluated_predictions_sha256", "evaluated_predictions.csv"),
                    ("metrics_sha256", "metrics.json"),
                ):
                    require(audit_run[key] == sha256(run_dir / filename),
                            f"{exp_id}/{protocol}/{audit_run['run_name']}: {filename} changed")
                require(close(audit_run["MAE"], summary_run["MAE"]),
                        f"{exp_id}/{protocol}: run MAE mismatch")
                values.append(float(audit_run["MAE"]))
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            require(close(mean, summary["MAE_mean"])
                    and (sd is None and summary["MAE_SD"] is None
                         or sd is not None and close(sd, summary["MAE_SD"])),
                    f"{exp_id}/{protocol}: aggregate MAE mismatch")
            expected_sd = ("sample SD across official split runs" if protocol == PROTOCOLS[0]
                           else "sample SD across model seeds" if sd is not None
                           else "N/A: one deterministic full-train run")
            require(summary["MAE_SD_definition"] == expected_sd,
                    f"{exp_id}/{protocol}: SD definition mismatch")
            record = registry.get((exp_id, protocol))
            require(record is not None and record["status"] == "evaluated_and_audited"
                    and record["summary_path"] == str(summary_path.relative_to(ROOT))
                    and record["evaluation_audit_path"] == str(evaluation_path.relative_to(ROOT))
                    and record["run_count"] == str(len(runs))
                    and close(record["MAE_mean"], mean)
                    and (record["MAE_SD"] == "N/A" if sd is None
                         else close(record["MAE_SD"], sd)),
                    f"{exp_id}/{protocol}: experiment registry mismatch")
            for path in (blind_path, evaluation_path, summary_path):
                sources[str(path.relative_to(ROOT))] = sha256(path)
            rows.append({
                "experiment_id": exp_id,
                "representation": experiment["representation"] or "none",
                "algorithm": experiment["algorithm"],
                "candidate_id": frozen["candidate_id"],
                "benchmark_protocol": protocol,
                "n_runs": len(runs),
                "individual_run_ids": ";".join(expected_names),
                "run_level_MAE": ";".join(repr(v) for v in values),
                "MAE_mean": mean,
                "MAE_SD": sd if sd is not None else "N/A",
                "MAE_SD_definition": expected_sd,
                "test_rows_per_run": 182,
                "label_scale": "original TDC Y",
                "frozen_config_sha256": frozen_hash,
                "blind_audit_sha256": sha256(blind_path),
                "evaluation_audit_sha256": sha256(evaluation_path),
                "summary_sha256": sha256(summary_path),
            })
    require(len(rows) == 32, "Expected 32 matrix rows")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode()
    manifest = {
        "dataset": "caco2_wang", "model_version": "caco_trackA_v1",
        "scope": "16 frozen core baselines × two final protocols",
        "rows": len(rows), "experiments": len(experiments),
        "primary_metric": "MAE", "label_scale": "original TDC Y",
        "matrix_sha256": hashlib.sha256(content).hexdigest(),
        "validation_gate_sha256": sha256(gate_path),
        "validation_matrix_sha256": sha256(validation_path),
        "experiments_config_sha256": sha256(experiments_path),
        "experiments_registry_sha256": sha256(registry_path),
        "source_sha256": sources,
        "test_y_read_by_summary": False,
        "selection_changed_after_test": False,
    }
    output_path.write_bytes(content)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} audited rows: {output_path.relative_to(ROOT)}")
    print(f"SHA-256: {manifest['matrix_sha256']}")


if __name__ == "__main__":
    main()
