#!/usr/bin/env bash
# CFG guidance-scale sweep for the class-conditional CIFAR runs (RunPod).
#
# Why: the conditional ablation runs were only evaluated at CFG=1.5 and came out
# WORSE than their unconditional counterparts (wave: 57.9 vs 55.6 FID). FID vs
# guidance scale is a U-curve, so a single point says nothing about where the
# optimum is. This sweeps eval-time guidance over the SAME checkpoint — sampling
# is cheap, training is not.
#
# The original conditional checkpoints were lost with the pod (autopush only
# saves small artifacts), so each run is retrained iff no checkpoint_epoch*.pt
# is present — an existing metrics.json does NOT skip training here, unlike
# run_image_ablation.sh. After training, an EMA-only checkpoint (~32 MB) is
# extracted and INCLUDED in the autopush so the weights survive pod termination.
#
# Mirrors run_image_ablation.sh conventions:
#   - master log mirrored into outputs/run_master.log
#   - resumable: per-scale metrics_g<w>.json skipped if it already exists
#   - per-run failure isolation, AUTOPUSH, SHUTDOWN self-terminate
#
# Usage:
#   bash scripts/run_guidance_sweep.sh                         # full sweep
#   SMOKE=1 bash scripts/run_guidance_sweep.sh                 # 2-epoch dry run
#   AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_guidance_sweep.sh   # overnight
#   SCALES="1.0 1.5 2.0" bash scripts/run_guidance_sweep.sh    # custom scales
#   EPOCHS=200 FAST=1 bash scripts/run_guidance_sweep.sh       # big-GPU preset

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs
: > outputs/run_master.log
exec > >(tee -a outputs/run_master.log) 2>&1

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-200}"
FAST="${FAST:-0}"
AUTOPUSH="${AUTOPUSH:-0}"
SHUTDOWN="${SHUTDOWN:-0}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10000}"
SCALES="${SCALES:-1.0 1.25 1.5 1.75 2.0 3.0}"

if [ "$SMOKE" = "1" ]; then
    EPOCHS=2; N_EVAL_SAMPLES=256; SCALES="1.0 2.0"
fi

COMMON_FAST=()
[ "$FAST" = "1" ] && COMMON_FAST=(--batch_size 256 --lr 2e-4)

# Same configs as the _cond rows of run_image_ablation.sh. Training guidance
# flag only sets the config default; eval overrides it per sweep point.
COND_FLAGS="--num_classes 10 --sampler dpmpp --guidance_scale 1.5"
RUNS=(
  "outputs/cifar_wave_full_sc_cond      | --attn wave --conditioning physics --kernel 2d --dynamic_filter --gating hyena --aniso_kernel ${COND_FLAGS}"
  "outputs/cifar_standard_adaln_sc_cond | --attn standard --conditioning adaln ${COND_FLAGS}"
)

# -----------------------------------------------------------------------------
# Auto-push (same extension-glob approach as run_image_ablation.sh, plus the
# small EMA-only *_ema.pt checkpoints so trained weights survive the pod)
# -----------------------------------------------------------------------------
autopush_results() {
    [ "$AUTOPUSH" = "1" ] || return 0
    local dir="$1"
    echo; echo "=== Auto-push results for ${dir} ==="
    shopt -s nullglob
    git add -f "${dir}"/*.json "${dir}"/*.log "${dir}"/*.png \
               "${dir}"/*_ema.pt outputs/run_master.log 2>/dev/null || true
    shopt -u nullglob
    if git diff --cached --quiet 2>/dev/null; then echo "  (nothing new)"; return 0; fi
    git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit -m "Guidance sweep: $(basename "$dir") [auto]" >/dev/null 2>&1 || true
    local out rc
    out="$(git push 2>&1)"; rc=$?
    echo "$out" | tail -2
    [ "$rc" -eq 0 ] && echo "  pushed ✓" || echo "  PUSH FAILED — results still on pod; check token/remote"
}

# -----------------------------------------------------------------------------
# Self-terminate trap — final safety push first, terminate only if it succeeds
# -----------------------------------------------------------------------------
shutdown_trap() {
    local code=$?
    echo; echo "=== Run ended (exit ${code}) ==="
    if [ "$AUTOPUSH" = "1" ]; then
        echo "Final safety push of all results + logs before terminating…"
        shopt -s nullglob
        git add -f outputs/cifar_*/*.json outputs/cifar_*/*.log outputs/cifar_*/*.png \
                   outputs/cifar_*/*_ema.pt outputs/run_master.log 2>/dev/null || true
        shopt -u nullglob
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "Guidance sweep: final [auto]" >/dev/null 2>&1 || true
        local out rc
        out="$(git push 2>&1)"; rc=$?
        echo "$out" | tail -3
        if [ "$rc" -ne 0 ]; then
            echo "FINAL PUSH FAILED — NOT terminating pod so results aren't lost."
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
echo "Plan: EPOCHS=${EPOCHS}  FAST=${FAST}  eval_samples=${N_EVAL_SAMPLES}  "\
"scales='${SCALES}'  AUTOPUSH=${AUTOPUSH}  SHUTDOWN=${SHUTDOWN}"

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
# Extract a small EMA-only checkpoint (loadable by metrics.evaluate: it sorts
# after the full checkpoint of the same epoch and carries ema_state_dict)
# -----------------------------------------------------------------------------
extract_ema() {
    python - "$1" <<'PY'
import glob, sys, torch
d = sys.argv[1]
full = [c for c in sorted(glob.glob(f"{d}/checkpoint_epoch*.pt")) if not c.endswith("_ema.pt")]
if not full:
    raise SystemExit(f"no full checkpoint in {d}")
ck = full[-1]
c = torch.load(ck, map_location="cpu", weights_only=False)
state = c.get("ema_state_dict") or c["model_state_dict"]
out = ck.replace(".pt", "_ema.pt")
torch.save({"ema_state_dict": state}, out)
print(f"EMA-only checkpoint → {out}")
PY
}

# -----------------------------------------------------------------------------
# One run: train iff no checkpoint, then eval at each guidance scale
# -----------------------------------------------------------------------------
run_one() {
    local save_dir flags
    IFS='|' read -r save_dir flags <<< "$1"
    save_dir="$(echo "$save_dir" | xargs)"; flags="$(echo "$flags" | xargs)"

    shopt -s nullglob
    local ckpts=("${save_dir}"/checkpoint_epoch*.pt)
    shopt -u nullglob
    if [ ${#ckpts[@]} -gt 0 ]; then
        echo; echo "=== Skip training ${save_dir} (checkpoint exists) ==="
    else
        echo; echo "=== Train ${save_dir} ==="; echo "    train_cifar.py ${flags}  (epochs=${EPOCHS})"
        mkdir -p "$save_dir"
        # shellcheck disable=SC2086  (flags intentionally word-split)
        if ! python train_cifar.py ${flags} --save_dir "$save_dir" --epochs "$EPOCHS" \
                "${COMMON_FAST[@]}" 2>&1 | tee "${save_dir}/training.log"; then
            echo "!!! Training FAILED for ${save_dir} — see training.log. Continuing."
            autopush_results "$save_dir"; return 0
        fi
        extract_ema "$save_dir" || echo "!!! EMA extraction failed for ${save_dir}"
        autopush_results "$save_dir"
    fi

    for w in $SCALES; do
        local out_name="metrics_g${w}.json"
        if [ -f "${save_dir}/${out_name}" ]; then
            echo "  --- skip guidance ${w} (${out_name} exists)"
            continue
        fi
        echo; echo "  --- Eval ${save_dir} @ guidance ${w} ---"
        if ! python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES" \
                --guidance_scale "$w" --out_name "$out_name"; then
            echo "!!! Eval FAILED for ${save_dir} @ guidance ${w} — continuing."
        fi
        autopush_results "$save_dir"
    done
}

for entry in "${RUNS[@]}"; do run_one "$entry"; done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo; echo "=== Guidance sweep summary (FID, lower = better) ==="
for entry in "${RUNS[@]}"; do
    IFS='|' read -r save_dir _ <<< "$entry"; save_dir="$(echo "$save_dir" | xargs)"
    echo "  ${save_dir#outputs/}:"
    for w in $SCALES; do
        f="${save_dir}/metrics_g${w}.json"
        if [ -f "$f" ]; then
            python - "$f" "$w" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"    guidance {sys.argv[2]:>5}  FID {m['fid_clean_cifar10_train']:.3f}")
PY
        else
            printf "    guidance %5s  MISSING\n" "$w"
        fi
    done
done
echo; echo "Done. Compare against unconditional: wave_full 55.63, standard_adaln 57.40."
