#!/usr/bin/env bash
# Efficiency-only pod run — NO training, NO dataset. Runs in ~10 min, so it's
# the cheapest way to answer "does the efficiency claim survive a fair baseline?"
#
# Two benchmarks, both on FlashAttention (F.scaled_dot_product_attention), the
# real modern softmax — not the O(L²) strawman the earlier numbers used:
#   1. benchmark_crossover.py  — operator-level sweep 1k..64k tokens: naive vs
#      flash softmax vs wave (± upgraded). Finds the crossover length and writes
#      a scaling plot.
#   2. benchmark_audio_models.py — full WaveFieldAudioDenoiser at the production
#      1024-token config, now with the fair SDPA softmax baseline.
#
# Usage:
#   AUTOPUSH=1 SHUTDOWN=1 bash scripts/run_benchmarks.sh    # walk-away (recommended)
#   bash scripts/run_benchmarks.sh                          # attended
#   LENGTHS="1024 4096 16384 65536" bash scripts/run_benchmarks.sh
#   CROSS_BS=8 CROSS_DIM=256 CROSS_HEADS=8 bash scripts/run_benchmarks.sh

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs
: > outputs/run_master.log
exec > >(tee -a outputs/run_master.log) 2>&1

AUTOPUSH="${AUTOPUSH:-0}"
SHUTDOWN="${SHUTDOWN:-0}"
LENGTHS="${LENGTHS:-1024 2048 4096 8192 16384 32768 65536}"
CROSS_BS="${CROSS_BS:-8}"
CROSS_DIM="${CROSS_DIM:-256}"
CROSS_HEADS="${CROSS_HEADS:-8}"
AUDIO_BS="${AUDIO_BS:-64}"

autopush() {
    [ "$AUTOPUSH" = "1" ] || return 0
    echo; echo "=== Auto-push ==="
    shopt -s nullglob
    git add -f outputs/crossover/*.json outputs/crossover/*.png \
               outputs/audio_bench/*.json outputs/audio_bench/*.log \
               outputs/run_master.log 2>/dev/null || true
    shopt -u nullglob
    if git diff --cached --quiet 2>/dev/null; then echo "  (nothing new)"; return 0; fi
    git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit -m "Efficiency benchmarks [auto]" >/dev/null 2>&1 || true
    local out rc; out="$(git push 2>&1)"; rc=$?
    echo "$out" | tail -2
    [ "$rc" -eq 0 ] && echo "  pushed ✓" || echo "  PUSH FAILED — results still on pod"
}

terminate_pod() {
    [ "$SHUTDOWN" = "1" ] || return 0
    if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "SHUTDOWN=1 but RUNPOD_POD_ID/RUNPOD_API_KEY unset — NOT terminating; stop the pod manually."
        return 0
    fi
    echo "Terminating pod ${RUNPOD_POD_ID}…"
    local payload
    payload=$(printf '{"query":"mutation{podTerminate(input:{podId:\\"%s\\"})}"}' "$RUNPOD_POD_ID")
    curl -s -H "Content-Type: application/json" \
        "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -d "$payload"; echo
}

shutdown_trap() {
    echo; echo "=== Run ended ==="
    autopush
    # Only terminate if the final push succeeded (or AUTOPUSH off) — never strand results.
    if [ "$AUTOPUSH" = "1" ] && [ "$SHUTDOWN" = "1" ]; then
        if git push >/dev/null 2>&1 || git diff --quiet HEAD origin/"$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null; then
            terminate_pod
        else
            echo "FINAL PUSH not confirmed — NOT terminating pod so results aren't lost."
        fi
    fi
}
[ "$SHUTDOWN" = "1" ] && trap shutdown_trap EXIT

echo "=== Pre-flight ==="
python -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU visible. Aborting — this benchmark needs CUDA for the memory numbers.')
p = torch.cuda.get_device_properties(0)
print(f'CUDA {torch.version.cuda}  {p.name}  {p.total_memory/1e9:.1f} GB  torch {torch.__version__}')
" || exit 1

echo; echo "=== 1/2: crossover sweep (wave vs naive/flash softmax, 1k..64k) ==="
mkdir -p outputs/crossover
python scripts/benchmark_crossover.py --lengths $LENGTHS \
    --batch_size "$CROSS_BS" --dim "$CROSS_DIM" --heads "$CROSS_HEADS" --upgraded \
    2>&1 | tee outputs/crossover/crossover.log || echo "!!! crossover benchmark failed — continuing."
autopush

echo; echo "=== 2/2: full-model audio benchmark @ 1024 tokens (fair SDPA softmax) ==="
mkdir -p outputs/audio_bench
python scripts/benchmark_audio_models.py --out_dir outputs/audio_bench --batch_size "$AUDIO_BS" \
    2>&1 | tee outputs/audio_bench/bench.log || echo "!!! audio benchmark failed — continuing."
autopush

echo; echo "Done. Crossover: outputs/crossover/{crossover.json,crossover.png}; "\
"full-model: outputs/audio_bench/audio_bench.json"
