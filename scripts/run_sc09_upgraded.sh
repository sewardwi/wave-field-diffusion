#!/usr/bin/env bash
# SC09 with the upgraded wave operator + class-conditional CFG (RunPod).
#
# Answers the two open questions from the 2026-07-18 run
# (see README.md "SC09 audio — first clean run" + results/sc09_fsd_table.json):
#   1. The base wave kernel lost badly to softmax on audio (FSD 43.6 vs 30.0)
#      and mode-collapsed onto two digits. On CIFAR the same content-independent-
#      kernel weakness was fixed by --dynamic_filter --gating hyena (FID 85.9 ->
#      55.6, beating softmax). Never tested on audio until now.
#   2. Does class-conditioning + CFG break the mode collapse directly (the
#      standard fix, independent of the kernel upgrade)?
#
# 2x2 matrix, all wave + upgraded operator, all bs=64 (matched steps; softmax
# is NOT in this matrix so the OOM lesson doesn't apply, but bs=64 keeps this
# comparable to the corrected run_sc09_ablation.sh baseline):
#   - physics,   unconditional
#   - adaln,     unconditional
#   - physics,   class-conditional + CFG
#   - adaln,     class-conditional + CFG
#
# Resumable: any run whose metrics.json already exists is skipped.
# Estimated wall-clock on a single RTX 4090: ~3-4 hours (4 runs, bs=64, 100 ep).
#
# Usage:
#   bash scripts/run_sc09_upgraded.sh                                # full matrix
#   SMOKE=1 bash scripts/run_sc09_upgraded.sh                        # 3-epoch smoke test
#   AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_sc09_upgraded.sh          # overnight: push + self-terminate
#   EPOCHS=150 bash scripts/run_sc09_upgraded.sh                     # override epoch count
#   GUIDANCE=2.0 bash scripts/run_sc09_upgraded.sh                   # CFG scale (image sweep found optimum >=3; unknown for audio)
#   N_EVAL_SAMPLES=5000 bash scripts/run_sc09_upgraded.sh            # cheaper final FSD
#   SKIP_EVAL=1 bash scripts/run_sc09_upgraded.sh                    # train only, no 10k eval

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
EPOCHS="${EPOCHS:-100}"
AUTOPUSH="${AUTOPUSH:-0}"
SHUTDOWN="${SHUTDOWN:-0}"
GUIDANCE="${GUIDANCE:-2.0}"     # CFG scale for the conditional runs' periodic sampling + final eval

if [ "$SMOKE" = "1" ]; then
    EPOCHS=3; N_EVAL_SAMPLES=64
fi

# attn/gating are fixed (that's the point of this run); only conditioning and
# class-conditioning vary.
UPGRADE_FLAGS="--attn wave --dynamic_filter --gating hyena"
COND_FLAGS="--num_classes 10 --guidance_scale ${GUIDANCE}"

RUNS=(
    "outputs/sc09_wave_dyn_hyena_physics       | --conditioning physics"
    "outputs/sc09_wave_dyn_hyena_adaln         | --conditioning adaln"
    "outputs/sc09_wave_dyn_hyena_physics_cond  | --conditioning physics ${COND_FLAGS}"
    "outputs/sc09_wave_dyn_hyena_adaln_cond    | --conditioning adaln ${COND_FLAGS}"
)

# -----------------------------------------------------------------------------
# Auto-push (extension globs under nullglob — see run_sc09_ablation.sh; a
# specific missing filename aborts `git add` entirely and stages nothing).
# -----------------------------------------------------------------------------
autopush_results() {
    [ "$AUTOPUSH" = "1" ] || return 0
    local dir="$1"
    echo; echo "=== Auto-push results for ${dir} ==="
    shopt -s nullglob
    git add -f "${dir}"/*.json "${dir}"/*.log "${dir}"/*.png "${dir}"/*_ema.pt \
               outputs/run_master.log 2>/dev/null || true
    shopt -u nullglob
    if git diff --cached --quiet 2>/dev/null; then echo "  (nothing new)"; return 0; fi
    git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit -m "SC09 upgraded-operator: $(basename "$dir") [auto]" >/dev/null 2>&1 || true
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
        git add -f outputs/sc09_wave_dyn_hyena_*/*.json outputs/sc09_wave_dyn_hyena_*/*.log \
                   outputs/sc09_wave_dyn_hyena_*/*.png outputs/sc09_wave_dyn_hyena_*/*_ema.pt \
                   outputs/run_master.log 2>/dev/null || true
        shopt -u nullglob
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "SC09 upgraded-operator: final [auto]" >/dev/null 2>&1 || true
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
# Pre-flight (same CUDA + AMP-gradient check as the overnight orchestrator —
# this script also runs p_losses under autocast, so the same bug class applies)
# -----------------------------------------------------------------------------
echo "=== Pre-flight ==="
python scripts/preflight_amp_check.py || { echo "PREFLIGHT FAILED — aborting."; exit 1; }

if [ ! -d "data/SpeechCommands/speech_commands_v0.02" ]; then
    echo; echo "=== Pre-fetching SC09 (~2.3 GB) ==="
    python -c "from datasets.sc09 import SC09; SC09(root='./data', subset='training', cache=False)"
fi
if [ ! -f "metrics/weights/sc09_classifier.pt" ]; then
    echo; echo "=== Training SC09 classifier (FSD feature extractor) ==="
    python -m metrics.train_classifier --task sc09 --epochs 10
fi

echo "Plan: EPOCHS=${EPOCHS}  bs=64 (fixed — matched steps, see run_sc09_ablation.sh)  "\
"guidance=${GUIDANCE}  eval_samples=${N_EVAL_SAMPLES}  AUTOPUSH=${AUTOPUSH}  SHUTDOWN=${SHUTDOWN}"

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
# Extract a small EMA-only checkpoint (~5 MB) so autopush can afford weights
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
# One run: train (resumable, failure-isolated) -> eval -> EMA extract -> push
# -----------------------------------------------------------------------------
run_one() {
    local save_dir flags
    IFS='|' read -r save_dir flags <<< "$1"
    save_dir="$(echo "$save_dir" | xargs)"; flags="$(echo "$flags" | xargs)"

    if [ -f "${save_dir}/metrics.json" ]; then
        echo; echo "=== Skip ${save_dir} (metrics.json exists — delete to re-run) ==="
        return 0
    fi

    echo; echo "=== Train ${save_dir} ==="
    echo "    ${UPGRADE_FLAGS} ${flags}  (epochs=${EPOCHS}, bs=64)"
    mkdir -p "$save_dir"
    # shellcheck disable=SC2086
    if ! python train_audio.py ${UPGRADE_FLAGS} ${flags} \
            --save_dir "$save_dir" --epochs "$EPOCHS" --batch_size 64 \
            2>&1 | tee "${save_dir}/training.log"; then
        echo "!!! Training FAILED for ${save_dir} — see training.log. Continuing."
        autopush_results "$save_dir"; return 0
    fi

    extract_ema "$save_dir" || echo "!!! EMA extraction failed for ${save_dir}"

    if [ "$SKIP_EVAL" = "1" ]; then
        echo "    (skipping eval — SKIP_EVAL=1)"
    # Explicit --guidance_scale: don't rely on config.json fallback in
    # metrics.evaluate for conditional runs; belt-and-suspenders.
    elif ! python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES" \
            --guidance_scale "$GUIDANCE"; then
        echo "!!! Eval FAILED for ${save_dir} — continuing."
    fi
    autopush_results "$save_dir"
}

for entry in "${RUNS[@]}"; do run_one "$entry"; done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo; echo "=== SC09 upgraded-operator summary ==="
printf "  %-38s  %8s  %8s  %8s\n" "run" "FSD" "conf_acc" "entropy"
printf "  %-38s  %8s  %8s  %8s\n" "--------------------------------------" "--------" "--------" "--------"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r save_dir _ <<< "$entry"; save_dir="$(echo "$save_dir" | xargs)"
    if [ -f "${save_dir}/metrics.json" ]; then
        python - "$save_dir" <<'PY'
import json, sys
d = sys.argv[1]; m = json.load(open(f"{d}/metrics.json"))
print(f"  {d.replace('outputs/',''):<38}  "
      f"{m.get('frechet_sc09_distance', float('nan')):8.3f}  "
      f"{m.get('confident_accuracy', float('nan')):8.3f}  "
      f"{m.get('class_entropy', float('nan')):8.3f}")
PY
    else
        printf "  %-38s  %8s  %8s  %8s\n" "${save_dir#outputs/}" "MISSING" "-" "-"
    fi
done
echo; echo "Done. Compare against base-operator 2x2: wave+physics 43.6/0.83, "\
"wave+adaln 66.5/0.72, softmax+physics 30.0/1.53 (see results/sc09_fsd_table.json)."
