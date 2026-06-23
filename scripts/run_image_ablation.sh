#!/usr/bin/env bash
# Image ablation pipeline for a rented GPU box (RunPod).
#
# Trains the new wave-operator improvements *cumulatively* on CIFAR-10 so each
# feature's contribution to FID is isolated, plus a class-conditional + CFG
# "go-for-SOTA" run and a matched conditional softmax baseline. Optionally runs
# the same idea on MNIST. Mirrors scripts/run_sc09_ablation.sh:
#   - master log mirrored into outputs/run_master.log (the pod is ephemeral)
#   - AUTOPUSH=1 force-pushes small artifacts (config/metrics/logs/pngs) to GitHub
#     after each run (checkpoints + fid_gen_tmp stay on the pod — too big for git)
#   - SHUTDOWN=1 self-terminates the pod via the RunPod API when done (stops billing),
#     but only after a successful final push, so results can never be lost
#   - resumable: any run whose metrics.json already exists is skipped (so the
#     existing unconditional baselines are REUSED for free, not retrained)
#   - per-run failure isolation: one OOM/crash never aborts the whole sweep
#
# Usage:
#   bash scripts/run_image_ablation.sh                                  # full CIFAR sweep
#   SMOKE=1 bash scripts/run_image_ablation.sh                          # 2-epoch smoke test, tiny eval
#   AUTOPUSH=1 bash scripts/run_image_ablation.sh                       # push results after each run
#   AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_image_ablation.sh            # overnight: push + self-terminate
#   DATASETS="cifar mnist" bash scripts/run_image_ablation.sh           # also run the MNIST sweep
#   EPOCHS=200 bash scripts/run_image_ablation.sh                       # override CIFAR epoch count
#   FAST=1 bash scripts/run_image_ablation.sh                           # big-GPU preset (bs=256)
#   N_EVAL_SAMPLES=10000 bash scripts/run_image_ablation.sh             # FID sample budget
#   GUIDANCE=1.5 bash scripts/run_image_ablation.sh                     # CFG scale for conditional runs

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs
: > outputs/run_master.log
exec > >(tee -a outputs/run_master.log) 2>&1

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10000}"
EPOCHS="${EPOCHS:-200}"
MNIST_EPOCHS="${MNIST_EPOCHS:-100}"
FAST="${FAST:-0}"
AUTOPUSH="${AUTOPUSH:-0}"
SHUTDOWN="${SHUTDOWN:-0}"
DATASETS="${DATASETS:-cifar}"
GUIDANCE="${GUIDANCE:-1.5}"      # CFG scale for the conditional runs

if [ "$SMOKE" = "1" ]; then
    EPOCHS=2; MNIST_EPOCHS=2; N_EVAL_SAMPLES=256
fi

# FAST preset: fewer, larger gradient steps. Applied uniformly to every run so
# the ablation stays matched. Pair with a higher EPOCHS if undertrained.
COMMON_FAST=()
[ "$FAST" = "1" ] && COMMON_FAST=(--batch_size 256 --lr 2e-4)

# -----------------------------------------------------------------------------
# Run matrices. Each entry: "save_dir | train_script | extra flags"
# CIFAR is a CUMULATIVE ablation (each row adds one new feature) so the FID
# delta attributable to each piece is visible, then the conditional+CFG SOTA
# attempt and a matched conditional softmax baseline.
# -----------------------------------------------------------------------------
WAVE_BASE="--attn wave --conditioning physics --kernel 2d"
COND_FLAGS="--num_classes 10 --sampler dpmpp --guidance_scale ${GUIDANCE}"

CIFAR_RUNS=(
  "outputs/cifar_standard_adaln_sc            | train_cifar.py | --attn standard --conditioning adaln"
  "outputs/cifar_wave_physics_2d_sc           | train_cifar.py | ${WAVE_BASE}"
  "outputs/cifar_wave_dyn_sc                  | train_cifar.py | ${WAVE_BASE} --dynamic_filter"
  "outputs/cifar_wave_dyn_hyena_sc            | train_cifar.py | ${WAVE_BASE} --dynamic_filter --gating hyena"
  "outputs/cifar_wave_full_sc                 | train_cifar.py | ${WAVE_BASE} --dynamic_filter --gating hyena --aniso_kernel"
  "outputs/cifar_wave_full_sc_cond            | train_cifar.py | ${WAVE_BASE} --dynamic_filter --gating hyena --aniso_kernel ${COND_FLAGS}"
  "outputs/cifar_standard_adaln_sc_cond       | train_cifar.py | --attn standard --conditioning adaln ${COND_FLAGS}"
)

MNIST_RUNS=(
  "outputs/mnist_standard_adaln            | train_mnist.py | --attn standard --conditioning adaln"
  "outputs/mnist_wave_full                 | train_mnist.py | --attn wave --conditioning physics --kernel 2d --dynamic_filter --gating hyena --aniso_kernel"
  "outputs/mnist_wave_full_cond            | train_mnist.py | --attn wave --conditioning physics --kernel 2d --dynamic_filter --gating hyena --aniso_kernel ${COND_FLAGS}"
)

# -----------------------------------------------------------------------------
# Auto-push: force-add only the small artifacts (outputs/ is gitignored; the big
# checkpoints + fid_gen_tmp stay on the pod). Every git call guarded so a push
# failure never aborts the sweep under `set -e`.
# -----------------------------------------------------------------------------
autopush_results() {
    [ "$AUTOPUSH" = "1" ] || return 0
    local dir="$1"
    echo; echo "=== Auto-push results for ${dir} ==="
    git add -f "${dir}/config.json" "${dir}/metrics.json" \
               "${dir}/metric_history.json" "${dir}/training.log" \
               "${dir}"/*.png outputs/run_master.log 2>/dev/null || true
    if git diff --cached --quiet 2>/dev/null; then echo "  (nothing new)"; return 0; fi
    git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit -m "Image ablation: $(basename "$dir") [auto]" >/dev/null 2>&1 || true
    if git push 2>&1 | tail -2; then echo "  pushed ✓"
    else echo "  PUSH FAILED — results still on pod; check token/remote"; fi
}

# -----------------------------------------------------------------------------
# Self-terminate trap (stops billing on an unattended run). Final safety push of
# ALL artifacts first; only removes the pod if that push succeeds — a broken
# token can never delete the box with unsaved results.
# -----------------------------------------------------------------------------
shutdown_trap() {
    local code=$?
    echo; echo "=== Run ended (exit ${code}) ==="
    if [ "$AUTOPUSH" = "1" ]; then
        echo "Final safety push of all results + logs before terminating…"
        git add -f outputs/cifar_*/config.json outputs/cifar_*/metrics.json \
                   outputs/cifar_*/metric_history.json outputs/cifar_*/training.log outputs/cifar_*/*.png \
                   outputs/mnist_*/config.json outputs/mnist_*/metrics.json \
                   outputs/mnist_*/metric_history.json outputs/mnist_*/training.log outputs/mnist_*/*.png \
                   outputs/run_master.log 2>/dev/null || true
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "Image ablation: final [auto]" >/dev/null 2>&1 || true
        if ! git push 2>&1 | tail -2; then
            echo "FINAL PUSH FAILED — NOT terminating pod so results aren't lost."
            echo "Fix the token/remote, push manually, then terminate from the web console."
            return
        fi
    fi
    echo "Terminating pod ${RUNPOD_POD_ID} via RunPod API to stop billing…"
    local payload
    payload=$(printf '{"query":"mutation{podTerminate(input:{podId:\\"%s\\"})}"}' "$RUNPOD_POD_ID")
    curl -s -H "Content-Type: application/json" \
        "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -d "$payload"; echo
}

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo "=== Pre-flight ==="
python -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU visible. Aborting — this script is for rented GPUs.')
p = torch.cuda.get_device_properties(0)
print(f'CUDA {torch.version.cuda}  {p.name}  {p.total_memory/1e9:.1f} GB')
"
echo "Plan: DATASETS='${DATASETS}'  EPOCHS=${EPOCHS}  FAST=${FAST}  "\
"eval_samples=${N_EVAL_SAMPLES}  guidance=${GUIDANCE}  AUTOPUSH=${AUTOPUSH}  SHUTDOWN=${SHUTDOWN}"

# Arm the shutdown trap AFTER the smoke early-exit so a smoke test never kills the box.
if [ "$SMOKE" != "1" ] && [ "$SHUTDOWN" = "1" ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "WARNING: SHUTDOWN=1 but RUNPOD_POD_ID / RUNPOD_API_KEY unset — pod will NOT self-terminate."
    else
        [ "$AUTOPUSH" = "1" ] || echo "WARNING: SHUTDOWN=1 without AUTOPUSH=1 — results NOT saved off-pod before termination."
        trap shutdown_trap EXIT
        echo "Self-terminate armed: pod ${RUNPOD_POD_ID} terminates when this script exits."
    fi
fi

# -----------------------------------------------------------------------------
# One run: train (resumable, failure-isolated) → eval → push
# -----------------------------------------------------------------------------
run_one() {
    local save_dir train_script flags
    IFS='|' read -r save_dir train_script flags <<< "$1"
    # trim surrounding whitespace
    save_dir="$(echo "$save_dir" | xargs)"; train_script="$(echo "$train_script" | xargs)"
    flags="$(echo "$flags" | xargs)"

    if [ -f "${save_dir}/metrics.json" ]; then
        echo; echo "=== Skip ${save_dir} (metrics.json exists — delete to re-run) ==="
        return 0
    fi

    local ep="$EPOCHS"
    [ "$train_script" = "train_mnist.py" ] && ep="$MNIST_EPOCHS"

    echo; echo "=== Train ${save_dir} ==="; echo "    ${train_script} ${flags}  (epochs=${ep})"
    mkdir -p "$save_dir"
    # shellcheck disable=SC2086  (flags intentionally word-split)
    if ! python "$train_script" ${flags} --save_dir "$save_dir" --epochs "$ep" \
            "${COMMON_FAST[@]}" 2>&1 | tee "${save_dir}/training.log"; then
        echo "!!! Training FAILED for ${save_dir} — see training.log. Continuing."
        autopush_results "$save_dir"; return 0
    fi

    if [ "$SKIP_EVAL" = "1" ]; then
        echo "    (skipping eval — SKIP_EVAL=1)"
    elif ! python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES"; then
        echo "!!! Eval FAILED for ${save_dir} — continuing."
    fi
    autopush_results "$save_dir"
}

# -----------------------------------------------------------------------------
# Execute selected sweeps
# -----------------------------------------------------------------------------
RUNS=()
for d in $DATASETS; do
    case "$d" in
        cifar) RUNS+=("${CIFAR_RUNS[@]}") ;;
        mnist) RUNS+=("${MNIST_RUNS[@]}") ;;
        *) echo "Unknown dataset '$d' (expected: cifar mnist)"; ;;
    esac
done

for entry in "${RUNS[@]}"; do run_one "$entry"; done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo; echo "=== Image ablation summary ==="
printf "  %-40s  %12s\n" "run" "metric"
printf "  %-40s  %12s\n" "----------------------------------------" "------------"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r save_dir _ _ <<< "$entry"; save_dir="$(echo "$save_dir" | xargs)"
    if [ -f "${save_dir}/metrics.json" ]; then
        python - "$save_dir" <<'PY'
import json, sys
d = sys.argv[1]; m = json.load(open(f"{d}/metrics.json"))
key = next((k for k in ("fid_clean_cifar10_train","frechet_mnist_distance",
                        "frechet_sc09_distance") if k in m), None)
val = f"{m[key]:.3f}" if key else "n/a"
print(f"  {d.replace('outputs/',''):<40}  {val:>12}")
PY
    else
        printf "  %-40s  %12s\n" "${save_dir#outputs/}" "MISSING"
    fi
done
echo; echo "Done. (Lower FID/FMD = better.) Conditional rows use CFG scale ${GUIDANCE}."
