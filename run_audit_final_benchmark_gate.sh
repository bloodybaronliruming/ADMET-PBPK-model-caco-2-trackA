#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
run_id="caco_trackA_final_benchmark_gate_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="progress/runs/$run_id"
mkdir -p "$run_dir"
printf '%s\n' 'python -u caco_trackA_v1/scripts/audit_final_benchmark_gate.py' > "$run_dir/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
python -c 'import sys, platform; print(sys.executable); print(platform.platform())' > "$run_dir/environment.txt"
git rev-parse HEAD > "$run_dir/git_head.txt"
git status --short > "$run_dir/git_status.txt"
sha256sum \
    caco_trackA_v1/scripts/audit_final_benchmark_gate.py \
    caco_trackA_v1/scripts/train_full.py \
    caco_trackA_v1/scripts/predict_test.py \
    caco_trackA_v1/scripts/evaluate_test.py \
    caco_trackA_v1/scripts/audit_predictions.py \
    caco_trackA_v1/results/baselines/baseline_final_matrix.csv \
    caco_trackA_v1/results/baselines/validation_gate.json > "$run_dir/input_hashes.txt"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit_status.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_at_utc.txt"' EXIT
python -u caco_trackA_v1/scripts/audit_final_benchmark_gate.py \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
