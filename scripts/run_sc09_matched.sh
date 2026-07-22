#!/usr/bin/env bash
# Matched batch-64 audio head-to-head + efficiency benchmark (RunPod).
#
# Closes the two gaps left by the 2026-07-22 upgraded-operator run
# (see README "SC09 audio" sections + results/sc09_fsd_table.json):
#   1. There is no clean softmax baseline at L=1024 — the only softmax audio
#      runs were bs=256 (OOM'd / 1k-eval-only). And the base-vs-upgraded wave
#      comparison is confounded by a ~4x training-budget difference.
#   2. The O(n log n) efficiency claim rests on synthetic benchmark_attn.py, not
#      the real models.
#
# So this trains the FOUR missing UNCONDITIONAL cells at bs=64 (matched to the
# already-done upgraded-wave bs=64 runs), completing a fully matched L=1024
# table across softmax / base-wave / upgraded-wave:
#     - standard + physics       - standard + adaln
#     - wave(base) + physics     - wave(base) + adaln
# and runs scripts/benchmark_audio_models.py to log per-step wall-clock + peak
# GPU memory for softmax vs wave(base) vs wave(upgraded) on the real denoisers.
#
# All runs: bs=64, lr=1e-4, 100 epochs, 10k-sample FSD — identical to the
# upgraded-wave runs, so the whole table is one matched comparison. No --fast
# (see run_sc09_ablation.sh: FAST cuts optimizer steps and softmax OOMs at
# bs=256). softmax at bs=64 fits in 24 GB.
#
# Usage:
#   AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_sc09_matched.sh    # overnight (recommended)
#   bash scripts/run_sc09_matched.sh                          # attended, no push/terminate
#   SMOKE=1 bash scripts/run_sc09_matched.sh                  # 3-epoch smoke + tiny bench
#   EPOCHS=150 bash scripts/run_sc09_matched.sh               # override epoch count
#   BENCH_ONLY=1 bash scripts/run_sc09_matched.sh             # just the efficiency benchmark
#   SKIP_EVAL=1 bash scripts/run_sc09_matched.sh              # train only, no 10k eval

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
BENCH_ONLY="${BENCH_ONLY:-0}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10000}"
EPOCHS="${EPOCHS:-100}"
AUTOPUSH="${AUTOPUSH:-0}"
SHUTDOWN="${SHUTDOWN:-0}"

BENCH_BS="${BENCH_BS:-64}"
BENCH_ITERS="${BENCH_ITERS:-30}"
if [ "$SMOKE" = "1" ]; then
    EPOCHS=3; N_EVAL_SAMPLES=64; BENCH_ITERS=3
fi

RUNS=(
    "outputs/sc09_matched_standard_physics | --attn standard --conditioning physics"
    "outputs/sc09_matched_standard_adaln   | --attn standard --conditioning adaln"
    "outputs/sc09_matched_wave_physics     | --attn wave --conditioning physics"
    "outputs/sc09_matched_wave_adaln       | --attn wave --conditioning adaln"
)

# -----------------------------------------------------------------------------
# Auto-push (extension globs under nullglob — see run_sc09_ablation.sh)
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
        commit -m "SC09 matched: $(basename "$dir") [auto]" >/dev/null 2>&1 || true
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
        git add -f outputs/sc09_matched_*/*.json outputs/sc09_matched_*/*.log \
                   outputs/sc09_matched_*/*.png outputs/sc09_matched_*/*_ema.pt \
                   outputs/audio_bench/*.json outputs/run_master.log 2>/dev/null || true
        shopt -u nullglob
        git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
            commit -m "SC09 matched: final [auto]" >/dev/null 2>&1 || true
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
# Pre-flight (CUDA + AMP self-cond gradient fix — this path also runs p_losses
# under autocast) + dataset/classifier prefetch
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

echo "Plan: EPOCHS=${EPOCHS}  bs=64  eval_samples=${N_EVAL_SAMPLES}  "\
"BENCH_ONLY=${BENCH_ONLY}  AUTOPUSH=${AUTOPUSH}  SHUTDOWN=${SHUTDOWN}"

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
# Efficiency benchmark FIRST — it's quick and gets the wall-clock/memory numbers
# pushed off-pod before the multi-hour training runs.
# -----------------------------------------------------------------------------
echo; echo "=== Efficiency benchmark: softmax vs wave(base) vs wave(upgraded) @ 1024 tokens ==="
mkdir -p outputs/audio_bench
if ! python scripts/benchmark_audio_models.py --out_dir outputs/audio_bench \
        --batch_size "$BENCH_BS" --iters "$BENCH_ITERS" 2>&1 | tee outputs/audio_bench/bench.log; then
    echo "!!! Benchmark FAILED — continuing to training runs."
fi
autopush_results outputs/audio_bench

if [ "$BENCH_ONLY" = "1" ]; then
    echo; echo "BENCH_ONLY=1 — skipping training runs."
    exit 0
fi

# -----------------------------------------------------------------------------
# Small EMA-only checkpoint (~5 MB) so autopush can afford weights
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
# One run: train (resumable, failure-isolated) -> EMA extract -> eval -> push
# -----------------------------------------------------------------------------
run_one() {
    local save_dir flags
    IFS='|' read -r save_dir flags <<< "$1"
    save_dir="$(echo "$save_dir" | xargs)"; flags="$(echo "$flags" | xargs)"

    if [ -f "${save_dir}/metrics.json" ]; then
        echo; echo "=== Skip ${save_dir} (metrics.json exists — delete to re-run) ==="
        return 0
    fi

    echo; echo "=== Train ${save_dir} ==="; echo "    ${flags}  (epochs=${EPOCHS}, bs=64)"
    mkdir -p "$save_dir"
    # shellcheck disable=SC2086
    if ! python train_audio.py ${flags} --save_dir "$save_dir" --epochs "$EPOCHS" \
            --batch_size 64 2>&1 | tee "${save_dir}/training.log"; then
        echo "!!! Training FAILED for ${save_dir} — see training.log. Continuing."
        autopush_results "$save_dir"; return 0
    fi

    extract_ema "$save_dir" || echo "!!! EMA extraction failed for ${save_dir}"

    if [ "$SKIP_EVAL" = "1" ]; then
        echo "    (skipping eval — SKIP_EVAL=1)"
    elif ! python -m metrics.evaluate "$save_dir" --n_samples "$N_EVAL_SAMPLES"; then
        echo "!!! Eval FAILED for ${save_dir} — continuing."
    fi
    autopush_results "$save_dir"
}

for entry in "${RUNS[@]}"; do run_one "$entry"; done

# -----------------------------------------------------------------------------
# Summary (matched bs=64 table; upgraded-wave points shown for reference)
# -----------------------------------------------------------------------------
echo; echo "=== Matched bs=64 SC09 summary (FSD ↓, entropy ↑; uniform=2.303) ==="
printf "  %-38s  %8s  %8s\n" "run" "FSD" "entropy"
printf "  %-38s  %8s  %8s\n" "--------------------------------------" "--------" "--------"
print_row() {  # $1 = dir
    if [ -f "$1/metrics.json" ]; then
        python - "$1" <<'PY'
import json, sys
d = sys.argv[1]; m = json.load(open(f"{d}/metrics.json"))
print(f"  {d.replace('outputs/',''):<38}  "
      f"{m.get('frechet_sc09_distance', float('nan')):8.3f}  "
      f"{m.get('class_entropy', float('nan')):8.3f}")
PY
    else
        printf "  %-38s  %8s  %8s\n" "${1#outputs/}" "MISSING" "-"
    fi
}
for entry in "${RUNS[@]}"; do
    IFS='|' read -r save_dir _ <<< "$entry"; print_row "$(echo "$save_dir" | xargs)"
done
echo "  --- upgraded-wave bs=64 (from 2026-07-22, for reference) ---"
print_row outputs/sc09_wave_dyn_hyena_physics
print_row outputs/sc09_wave_dyn_hyena_adaln
echo; echo "Efficiency numbers: outputs/audio_bench/audio_bench.json"
