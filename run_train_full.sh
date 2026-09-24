#!/usr/bin/env bash
set -u

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root" || exit 1
run_id="caco_trackA_final_training_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="progress/runs/$run_id"
mkdir -p "$run_dir"
if [[ "${1:-}" == "--audit-inputs" ]]; then
    arguments=(--audit-inputs --audit-report "$run_dir/input_audit.json")
else
    arguments=("$@")
fi
printf '%q ' python -u caco_trackA_v1/scripts/train_full.py "${arguments[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
python -c 'import sys, platform; print(sys.executable); print(platform.platform())' > "$run_dir/environment.txt"
git rev-parse HEAD > "$run_dir/git_head.txt"
git status --short > "$run_dir/git_status.txt"
hostname > "$run_dir/hostname.txt"
sha256sum \
    configs/baselines/experiments_manifest.json \
    configs/baselines/search_space/candidates_manifest.json \
    data/caco_trackA/raw/caco2_wang_train_val.csv \
    data/caco_trackA/process/splits_manifest.json \
    caco_trackA_v1/scripts/train_full.py \
    src/admet_pbpk/baselines.py \
    > "$run_dir/input_config_hashes.txt"
python -u caco_trackA_v1/scripts/train_full.py "${arguments[@]}" \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
status=$?
wait
printf '%s\n' "$status" > "$run_dir/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_at_utc.txt"
exit "$status"
