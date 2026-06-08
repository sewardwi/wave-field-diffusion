#!/usr/bin/env bash
# Full SC09 ablation pipeline for a rented GPU box.
#
# What it does:
#   1. Verifies a CUDA-capable GPU is visible.
#   2. Pre-fetches the SC09 (Speech Commands digit subset, ~2.3 GB) if missing.
#   3. Confirms the SC09 classifier weights used as the FSD feature extractor.
#   4. Trains 4 ablation configurations sequentially:
#        - wave     + physics
#        - standard + physics
#        - wave     + adaln
#        - standard + adaln
#   5. Runs 10k-sample reference FSD on each.
#   6. Prints a summary table of FSDs, confident-accuracies, and class entropies.
#
# Resumable: any run whose metrics.json already exists is skipped.
# Total wall-clock on a single RTX 4090: ~2 hours.
#
# Usage:
#   bash scripts/run_sc09_ablation.sh                          # full pipeline
#   FAST=1 bash scripts/run_sc09_ablation.sh                   # big-GPU preset (bs=256, lr=4e-4)
#   AUTOPUSH=1 FAST=1 bash scripts/run_sc09_ablation.sh        # git-push results after each config
#   AUTOPUSH=1 SHUTDOWN=1 FAST=1 bash scripts/run_sc09_ablation.sh   # overnight: push + self-terminate pod
#   EPOCHS=150 bash scripts/run_sc09_ablation.sh               # override epoch count
#   SMOKE=1 bash scripts/run_sc09_ablation.sh                  # 3-epoch smoke test only
#   N_EVAL_SAMPLES=5000 bash scripts/run_sc09_ablation.sh      # cheaper final FSD
#   SKIP_EVAL=1 bash scripts/run_sc09_ablation.sh              # train only, no 10k eval
#
# Speed: the dataset is cached in RAM and training/sampling use bf16 autocast
# on CUDA automatically. FAST=1 adds a batch-size/LR preset for further speedup
# (fewer, larger gradient steps — pair with a higher EPOCHS if undertrained).

set -euo pipefail

# cd to repo root regardless of where this script is invoked from
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10000}"
EPOCHS="${EPOCHS:-100}"
FAST="${FAST:-0}"
AUTOPUSH="${AUTOPUSH:-0}"     # git-commit+push small result files after each config
SHUTDOWN="${SHUTDOWN:-0}"     # terminate the pod when the run ends (stops billing)

# -----------------------------------------------------------------------------
# Auto-push results off the pod (so an overnight run is safe without rsync).
# Force-adds only the small artifacts — outputs/ is gitignored and the big
# checkpoints/.wav files stay on the pod. Every git call is guarded so a push
# failure (e.g. bad token) never aborts the run under `set -e`.
# -----------------------------------------------------------------------------
autopush_results() {
    [ "$AUTOPUSH" = "1" ] || return 0
    local dir="$1"
    echo
    echo "=== Auto-push results for ${dir} ==="
    git add -f "${dir}/metrics.json" "${dir}/metric_history.json" \
               "${dir}/config.json" "${dir}"/*.png 2>/dev/null || true
    if git diff --cached --quiet 2>/dev/null; then
        echo "  (nothing new to commit)"
        return 0
    fi
    git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit -m "SC09 results: $(basename "$dir") [auto]" >/dev/null 2>&1 || true
    if git push 2>&1 | tail -2; then
        echo "  pushed ✓"
    else
        echo "  PUSH FAILED — results are still on the pod; check token/remote"
    fi
}

# attention:conditioning pairs to run
RUNS=(
    "wave:physics"
    "standard:physics"
    "wave:adaln"
    "standard:adaln"
)

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo "=== Pre-flight ==="
python -c "
import torch
ok = torch.cuda.is_available()
if not ok:
    raise SystemExit('No CUDA GPU visible. Aborting — this script is for rented GPUs.')
print(f'CUDA: {torch.version.cuda}  device: {torch.cuda.get_device_name(0)}  '
      f'mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Pre-fetch SC09 if missing (the download takes 1-3 min on a datacenter line;
# better to fail here than mid-training)
if [ ! -d "data/SpeechCommands/speech_commands_v0.02" ]; then
    echo
    echo "=== Pre-fetching SC09 (~2.3 GB) ==="
    python -c "from datasets.sc09 import SC09; SC09(root='./data', subset='training', cache=False)"
fi

# Classifier weights ship with the repo, but verify and re-train if missing
if [ ! -f "metrics/weights/sc09_classifier.pt" ]; then
    echo
    echo "=== Training SC09 classifier (FSD feature extractor) ==="
    python -m metrics.train_classifier --task sc09 --epochs 10
fi

# -----------------------------------------------------------------------------
# Smoke test (optional)
# -----------------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
    echo
    echo "=== Smoke test (3 epochs, tiny eval) ==="
    python train_audio.py \
        --epochs 3 --sample_every 3 \
        --n_metric_samples 64 --n_real_features 200 \
        --batch_size 16 --ddim_steps 20 \
        --save_dir outputs/sc09_smoketest
    echo "Smoke test complete. Unset SMOKE and re-run for the full ablation."
    exit 0
fi

# -----------------------------------------------------------------------------
# Self-terminate the pod when the run ends, so an unattended overnight run does
# not keep billing after it finishes. Armed HERE (after the smoke early-exit) so
# a smoke test never shuts the pod down. The EXIT trap fires on success AND on
# failure — combined with AUTOPUSH (per-config), completed results are already
# pushed before the pod goes away. Disarm with Ctrl-C if you want to keep the box.
# -----------------------------------------------------------------------------
# The EXIT trap below does a final safety push of ALL result artifacts and only
# removes the pod if that push succeeds — so a broken token can never delete the
# box with unsaved results (worst case: pod stays up, you pay, nothing is lost).
shutdown_trap() {
    local code=$?
    echo
    echo "=== Run ended (exit ${code}) ==="
    if [ "$AUTOPUSH" = "1" ]; then
        echo "Final safety push of all results before terminating…"
        git add -f outputs/sc09_*/metrics.json outputs/sc09_*/metric_history.json \
                   outputs/sc09_*/config.json outputs/sc09_*/*.png 2>/dev/null || true
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "SC09 results: final [auto]" >/dev/null 2>&1 || true
        if ! git push 2>&1 | tail -2; then
            echo "FINAL PUSH FAILED — NOT terminating pod so results aren't lost."
            echo "Fix the token / rsync manually, then 'runpodctl remove pod ${RUNPOD_POD_ID}'."
            return
        fi
    fi
    echo "Terminating pod ${RUNPOD_POD_ID} to stop billing…"
    runpodctl remove pod "$RUNPOD_POD_ID"
}

if [ "$SHUTDOWN" = "1" ]; then
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "WARNING: SHUTDOWN=1 but runpodctl not found — pod will NOT self-terminate."
    elif [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "WARNING: SHUTDOWN=1 but \$RUNPOD_POD_ID is unset — pod will NOT self-terminate."
    else
        [ "$AUTOPUSH" = "1" ] || echo "WARNING: SHUTDOWN=1 without AUTOPUSH=1 — results are NOT saved off-pod before termination."
        trap shutdown_trap EXIT
        echo "Self-terminate armed: pod ${RUNPOD_POD_ID} will be removed when this script exits."
    fi
fi

# -----------------------------------------------------------------------------
# Run all four ablations
# -----------------------------------------------------------------------------
for entry in "${RUNS[@]}"; do
    attn="${entry%%:*}"
    cond="${entry##*:}"
    save_dir="outputs/sc09_${attn}_${cond}"

    if [ -f "$save_dir/metrics.json" ]; then
        echo
        echo "=== Skip ${save_dir} (metrics.json exists — delete to re-run) ==="
        continue
    fi

    echo
    echo "=== Training: attn=${attn}  conditioning=${cond} ==="
    echo "    → ${save_dir}"
    mkdir -p "$save_dir"

    # FAST preset lets train_audio.py pick batch_size/lr; otherwise pin bs=64.
    train_args=(--attn "$attn" --conditioning "$cond" --save_dir "$save_dir" --epochs "$EPOCHS")
    if [ "$FAST" = "1" ]; then
        train_args+=(--fast)
    else
        train_args+=(--batch_size 64)
    fi
    python train_audio.py "${train_args[@]}" 2>&1 | tee "${save_dir}/training.log"

    if [ "$SKIP_EVAL" = "1" ]; then
        echo "    (skipping 10k-sample eval — SKIP_EVAL=1)"
    else
        echo
        echo "=== Final ${N_EVAL_SAMPLES}-sample FSD eval: ${save_dir} ==="
        python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES"
    fi

    autopush_results "$save_dir"
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "=== SC09 ablation summary ==="
printf "  %-32s  %8s  %8s  %8s\n" "run" "FSD" "conf_acc" "entropy"
printf "  %-32s  %8s  %8s  %8s\n" "-------------------------------" "--------" "--------" "--------"
for entry in "${RUNS[@]}"; do
    attn="${entry%%:*}"
    cond="${entry##*:}"
    save_dir="outputs/sc09_${attn}_${cond}"
    if [ -f "$save_dir/metrics.json" ]; then
        python - <<PY
import json, pathlib
m = json.load(open("${save_dir}/metrics.json"))
print(f"  {'sc09_${attn}_${cond}':<32}  "
      f"{m.get('frechet_sc09_distance', float('nan')):8.3f}  "
      f"{m.get('confident_accuracy', float('nan')):8.3f}  "
      f"{m.get('class_entropy', float('nan')):8.3f}")
PY
    else
        printf "  %-32s  %8s  %8s  %8s\n" "sc09_${attn}_${cond}" "MISSING" "-" "-"
    fi
done

echo
echo "Done. To pull results back to your laptop, from your *local* machine:"
echo "  rsync -av -e 'ssh -p <port>' root@<host>:/path/to/wave-field-diffusion/outputs/sc09_*/ \\"
echo "        --exclude='checkpoint_epoch*.pt' --exclude='fid_gen_tmp' \\"
echo "        ~/Documents/coding-projects/wave-field-diffusion/outputs/"
