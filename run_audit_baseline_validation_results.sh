#!/usr/bin/env bash
set -u

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root" || exit 1
run_id="caco_trackA_validation_results_audit_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="progress/runs/$run_id"
mkdir -p "$run_dir"
printf '%s\n' "python -u caco_trackA_v1/scripts/audit_baseline_validation_results.py" > "$run_dir/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
python -c 'import sys, platform; print(sys.executable); print(platform.platform())' > "$run_dir/environment.txt"
python -u caco_trackA_v1/scripts/audit_baseline_validation_results.py \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
status=$?
wait
printf '%s\n' "$status" > "$run_dir/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_at_utc.txt"
exit "$status"
