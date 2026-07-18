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
#   DIAG=1 bash scripts/run_sc09_ablation.sh                   # mode-collapse diagnostic (~1h)
#   DIAG=1 AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_sc09_ablation.sh   # diagnostic, unattended
#   N_EVAL_SAMPLES=5000 bash scripts/run_sc09_ablation.sh      # cheaper final FSD
#   SKIP_EVAL=1 bash scripts/run_sc09_ablation.sh              # train only, no 10k eval
#
# Speed: the dataset is cached in RAM and training/sampling use bf16 autocast
# on CUDA automatically. FAST=1 adds a batch-size/LR preset for further speedup
# (fewer, larger gradient steps — pair with a higher EPOCHS if undertrained).

set -euo pipefail

# cd to repo root regardless of where this script is invoked from
cd "$(dirname "$0")/.."

# Mirror ALL output (this script's orchestration + every Python subprocess +
# any crash traceback) into a master log under outputs/, so AUTOPUSH can ship it
# to GitHub. The pod has no persistent volume — nothing survives SHUTDOWN — so
# the log must live in the repo, not on the pod. Fresh file per run.
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
FAST="${FAST:-0}"
AUTOPUSH="${AUTOPUSH:-0}"     # git-commit+push small result files after each config
SHUTDOWN="${SHUTDOWN:-0}"     # terminate the pod when the run ends (stops billing)
DIAG="${DIAG:-0}"            # run the mode-collapse diagnostic instead of the ablation
DIAG_EPOCHS="${DIAG_EPOCHS:-40}"   # short training per diagnostic cell
DIAG_EVAL="${DIAG_EVAL:-2000}"     # samples per diagnostic eval (cheaper than 10k)

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
    # Push the full logs (per-config training.log + the master run log) to GitHub
    # so nothing is lost when the ephemeral pod is torn down. Full checkpoints and
    # .wav samples are too big for git and stay on the pod; the small *_ema.pt
    # (EMA weights only, ~5 MB) IS pushed so the trained model survives.
    # Critical: stage by EXTENSION under nullglob — a specific filename that
    # doesn't exist (e.g. metrics.json after a crashed run) makes `git add`
    # abort and stage NOTHING. Extension globs simply vanish when absent.
    shopt -s nullglob
    git add -f "${dir}"/*.json "${dir}"/*.log "${dir}"/*.png "${dir}"/*_ema.pt \
               outputs/run_master.log 2>/dev/null || true
    shopt -u nullglob
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
#
# Terminates via the RunPod GraphQL API with curl (runpodctl's auth is flaky;
# curl with $RUNPOD_API_KEY is the reliable path). The EXIT trap does a final
# safety push of ALL result artifacts and only removes the pod if that push
# succeeds — so a broken token can never delete the box with unsaved results
# (worst case: pod stays up, you pay, nothing is lost).
# -----------------------------------------------------------------------------
shutdown_trap() {
    local code=$?
    echo
    echo "=== Run ended (exit ${code}) ==="
    if [ "$AUTOPUSH" = "1" ]; then
        echo "Final safety push of all results + full logs before terminating…"
        # Extension globs under nullglob (see autopush_results): a single
        # non-matching specific path (e.g. diag_* when no diagnostic ran) makes
        # `git add` abort and stage NOTHING.
        shopt -s nullglob
        git add -f outputs/sc09_*/*.json outputs/sc09_*/*.log outputs/sc09_*/*.png \
                   outputs/sc09_*/*_ema.pt \
                   outputs/diag_*/*.json outputs/diag_*/*.log outputs/diag_*/*.png \
                   outputs/run_master.log 2>/dev/null || true
        shopt -u nullglob
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "SC09 results: final [auto]" >/dev/null 2>&1 || true
        if ! git push 2>&1 | tail -2; then
            echo "FINAL PUSH FAILED — NOT terminating pod so results aren't lost."
            echo "Fix the token / rsync manually, then terminate the pod from the web console."
            return
        fi
    fi
    echo "Terminating pod ${RUNPOD_POD_ID} via RunPod API to stop billing…"
    local payload
    payload=$(printf '{"query":"mutation{podTerminate(input:{podId:\\"%s\\"})}"}' "$RUNPOD_POD_ID")
    curl -s -H "Content-Type: application/json" \
        "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -d "$payload"
    echo
}

if [ "$SHUTDOWN" = "1" ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "WARNING: SHUTDOWN=1 but \$RUNPOD_POD_ID is unset — pod will NOT self-terminate."
    elif [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "WARNING: SHUTDOWN=1 but \$RUNPOD_API_KEY is unset — pod will NOT self-terminate."
    else
        [ "$AUTOPUSH" = "1" ] || echo "WARNING: SHUTDOWN=1 without AUTOPUSH=1 — results are NOT saved off-pod before termination."
        trap shutdown_trap EXIT
        echo "Self-terminate armed: pod ${RUNPOD_POD_ID} will be terminated via RunPod API when this script exits."
    fi
fi

# -----------------------------------------------------------------------------
# Collapse diagnostic (DIAG=1): isolate the cause of SC09 mode collapse cheaply
# (~1h) before any full retrain. Runs a 2x2 on wave+physics (the hardest
# collapser): self_cond {on,off} x eval eta {0,1}, reporting class entropy
# (diversity) per cell. Hypothesis: self-conditioning + deterministic (eta=0)
# DDIM forms a collapse attractor — so self_cond=off and/or eta=1 should lift
# entropy. Reuses AUTOPUSH + the shutdown trap; safe to run unattended.
# -----------------------------------------------------------------------------
if [ "$DIAG" = "1" ]; then
    echo
    echo "=== COLLAPSE DIAGNOSTIC: wave+physics  self_cond{on,off} x eta{0,1} ==="
    echo "    DIAG_EPOCHS=${DIAG_EPOCHS}  eval_samples=${DIAG_EVAL}"

    for sc in on off; do
        [ "$sc" = "on" ] && sc_flag="--self_cond" || sc_flag="--no-self_cond"
        dir="outputs/diag_wave_physics_sc${sc}"
        if [ -f "${dir}/metrics_eta1.json" ]; then
            echo; echo "=== Skip ${dir} (already evaluated — delete to re-run) ==="
            continue
        fi
        echo
        echo "=== Train wave+physics self_cond=${sc} (${DIAG_EPOCHS} ep) -> ${dir} ==="
        mkdir -p "$dir"
        if ! python train_audio.py --attn wave --conditioning physics ${sc_flag} \
                --epochs "${DIAG_EPOCHS}" --batch_size 64 \
                --sample_every "${DIAG_EPOCHS}" --n_metric_samples 500 \
                --n_real_features 1000 --save_dir "${dir}" 2>&1 | tee "${dir}/training.log"; then
            echo "!!! Diagnostic training FAILED for self_cond=${sc} — continuing."
            autopush_results "$dir"; continue
        fi
        # A/B the same checkpoint at deterministic (eta=0) and stochastic (eta=1)
        for eta in 0 1; do
            echo; echo "=== Eval ${dir} at eta=${eta} (${DIAG_EVAL} samples) ==="
            if python -m metrics.evaluate "${dir}" --n_samples "${DIAG_EVAL}" --eta "${eta}"; then
                cp "${dir}/metrics.json" "${dir}/metrics_eta${eta}.json"
            else
                echo "!!! Eval FAILED for ${dir} eta=${eta} — continuing."
            fi
        done
        autopush_results "$dir"
    done

    echo
    echo "=== COLLAPSE DIAGNOSTIC SUMMARY ==="
    echo "    class entropy (uniform=2.303; higher = more diverse = less collapse)"
    printf "  %-22s  %8s  %8s\n" "config" "eta=0" "eta=1"
    printf "  %-22s  %8s  %8s\n" "---------------------" "--------" "--------"
    for sc in on off; do
        dir="outputs/diag_wave_physics_sc${sc}"
        python - "$dir" "$sc" <<'PY'
import json, sys
d, sc = sys.argv[1], sys.argv[2]
def ent(p):
    try:    return f"{json.load(open(p))['class_entropy']:.3f}"
    except Exception: return "    -"
print(f"  {'self_cond=' + sc:<22}  {ent(d + '/metrics_eta0.json'):>8}  {ent(d + '/metrics_eta1.json'):>8}")
PY
    done
    echo
    echo "Read: whichever knob (self_cond=off or eta=1) lifts entropy is the collapse driver."
    exit 0   # triggers the EXIT trap: final push of all logs/results + pod self-terminate
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
    # A single config's crash (e.g. OOM) must NOT abort the whole ablation. Catch
    # the failure, push its log so the error is visible off-pod, and continue.
    if ! python train_audio.py "${train_args[@]}" 2>&1 | tee "${save_dir}/training.log"; then
        echo "!!! Training FAILED for ${attn}_${cond} — see ${save_dir}/training.log. Continuing to next config."
        autopush_results "$save_dir"
        continue
    fi

    # Extract a small EMA-only checkpoint (~5 MB) that autopush can afford to
    # commit — the full checkpoints stay on the pod and die with it.
    python - "$save_dir" <<'PY' || echo "!!! EMA extraction failed for ${save_dir}"
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

    if [ "$SKIP_EVAL" = "1" ]; then
        echo "    (skipping 10k-sample eval — SKIP_EVAL=1)"
    elif ! python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES"; then
        echo "!!! Eval FAILED for ${attn}_${cond} — continuing to next config."
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
