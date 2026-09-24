#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

experiment=""
protocol=""
stage="all"
while (($#)); do
    case "$1" in
        --experiment) experiment="${2:?Missing experiment ID}"; shift 2 ;;
        --protocol) protocol="${2:?Missing protocol}"; shift 2 ;;
        --stage) stage="${2:?Missing stage}"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$experiment" && -n "$protocol" ]] || { printf 'Required: --experiment ID --protocol NAME\n' >&2; exit 2; }
case "$protocol" in
    tdc_compatible_5split|full_train_refit) ;;
    *) printf 'Unknown protocol: %s\n' "$protocol" >&2; exit 2 ;;
esac
case "$stage" in
    all|train|predict|audit-blind|evaluate|audit-evaluated) ;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac

# Verify the frozen experiment and gate before creating a run log.
algorithm="$(python - "$experiment" "$protocol" <<'PY'
import hashlib, json, sys
from pathlib import Path
exp, protocol = sys.argv[1:]
root = Path('caco_trackA_v1/results/baselines')
gate = json.loads((root / 'validation_gate.json').read_text())
path = root / exp / 'frozen_config.json'
if exp not in {item['experiment_id'] for item in json.loads(Path('configs/baselines/experiments.json').read_text())['experiments']}:
    raise SystemExit('Experiment ID is not in the frozen plan')
content = path.read_bytes()
record = json.loads(content)
if (gate.get('status') != 'PASS' or len(gate.get('frozen_configs', {})) != 16
        or gate['frozen_configs'].get(str(path)) != hashlib.sha256(content).hexdigest()
        or record['experiment_id'] != exp or protocol not in record['benchmark_protocols']):
    raise SystemExit('Validation Gate or frozen configuration mismatch')
print(record['algorithm'])
PY
)"

run_id="caco_trackA_baseline_final_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="progress/runs/$run_id"
mkdir -p "$run_dir"
printf '%q ' bash caco_trackA_v1/scripts/run_baseline_final.sh \
    --experiment "$experiment" --protocol "$protocol" --stage "$stage" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
python -c 'import sys, platform; from importlib.metadata import version; print(sys.executable); print(platform.platform()); print("Python", platform.python_version()); print("PyTDC", version("PyTDC")); print("scikit-learn", version("scikit-learn")); print("xgboost", version("xgboost"))' > "$run_dir/environment.txt"
git rev-parse HEAD > "$run_dir/git_head.txt"
git status --short > "$run_dir/git_status.txt"
hostname > "$run_dir/hostname.txt"
sha256sum \
    caco_trackA_v1/results/baselines/validation_gate.json \
    "caco_trackA_v1/results/baselines/$experiment/frozen_config.json" \
    caco_trackA_v1/scripts/run_baseline_final.sh \
    caco_trackA_v1/scripts/train_full.py \
    caco_trackA_v1/scripts/predict_test.py \
    caco_trackA_v1/scripts/audit_predictions.py \
    caco_trackA_v1/scripts/evaluate_test.py \
    src/admet_pbpk/baselines.py > "$run_dir/input_config_hashes.txt"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit_status.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_at_utc.txt"' EXIT

run_names=()
if [[ "$protocol" == "tdc_compatible_5split" ]]; then
    for seed in 1 2 3 4 5; do run_names+=("split_seed_$seed"); done
elif [[ "$algorithm" == "rf" || "$algorithm" == "et" || "$algorithm" == "xgb" ]]; then
    for seed in 1 2 3 4 5; do run_names+=("model_seed_$seed"); done
else
    run_names+=("single")
fi

run_args() {
    local name="$1"
    case "$name" in
        split_seed_*) run_options=(--split-seed "${name#split_seed_}") ;;
        model_seed_*) run_options=(--model-seed "${name#model_seed_}") ;;
        single) run_options=() ;;
    esac
}

do_train() {
    for name in "${run_names[@]}"; do
        run_args "$name"
        python -u caco_trackA_v1/scripts/train_full.py \
            --experiment "$experiment" --protocol "$protocol" "${run_options[@]}"
    done
}
do_predict() {
    for name in "${run_names[@]}"; do
        run_args "$name"
        metadata="caco_trackA_v1/results/baselines/$experiment/final/$protocol/$name/model_metadata.json"
        python -u caco_trackA_v1/scripts/predict_test.py \
            --experiment "$experiment" --protocol "$protocol" \
            "${run_options[@]}" --model-metadata "$metadata"
    done
}
do_audit_blind() {
    python -u caco_trackA_v1/scripts/audit_predictions.py \
        --experiment "$experiment" --protocol "$protocol" --mode blind
}
do_evaluate() {
    do_audit_blind
    for name in "${run_names[@]}"; do
        blind="caco_trackA_v1/results/baselines/$experiment/final/$protocol/$name/blind_predictions.csv"
        python -u caco_trackA_v1/scripts/evaluate_test.py --blind-predictions "$blind"
    done
}
do_audit_evaluated() {
    python -u caco_trackA_v1/scripts/audit_predictions.py \
        --experiment "$experiment" --protocol "$protocol" --mode evaluated
}

case "$stage" in
    all)
        do_train
        do_predict
        do_audit_blind
        do_evaluate
        do_audit_evaluated
        ;;
    train) do_train ;;
    predict) do_predict ;;
    audit-blind) do_audit_blind ;;
    evaluate) do_evaluate ;;
    audit-evaluated) do_audit_evaluated ;;
esac
printf 'Final stage complete: %s, %s, %s\n' "$experiment" "$protocol" "$stage"
