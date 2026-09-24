#!/usr/bin/env bash
set -u

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root" || exit 1

run_id="caco_trackA_baseline_validation_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="progress/runs/$run_id"
mkdir -p "$run_dir"
if [[ "${1:-}" == "--audit-inputs" ]]; then
    arguments=(--audit-inputs --audit-report "$run_dir/input_audit.json")
else
    arguments=("$@")
fi
printf '%q ' python -u caco_trackA_v1/scripts/validate_baselines.py "${arguments[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
python -c 'import sys, platform, sklearn, rdkit, numpy, pandas; from importlib.metadata import version; print(sys.executable); print(platform.platform()); print("Python", platform.python_version()); print("PyTDC", version("PyTDC")); print("RDKit", rdkit.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("sklearn", sklearn.__version__); print("xgboost", version("xgboost"))' > "$run_dir/environment.txt"
git rev-parse HEAD > "$run_dir/git_head.txt"
git status --short > "$run_dir/git_status.txt"
hostname > "$run_dir/hostname.txt"
sha256sum \
    data/caco_trackA/raw/caco2_wang_train_val.csv \
    data/caco_trackA/process/splits_manifest.json \
    configs/baselines/representations.json \
    configs/baselines/algorithms.json \
    configs/baselines/experiments.json \
    configs/baselines/search_space/candidates_manifest.json \
    caco_trackA_v1/scripts/validate_baselines.py \
    src/admet_pbpk/baselines.py \
    > "$run_dir/input_config_hashes.txt"
python -u caco_trackA_v1/scripts/validate_baselines.py "${arguments[@]}" \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
status=$?
wait
printf '%s\n' "$status" > "$run_dir/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_at_utc.txt"
exit "$status"
