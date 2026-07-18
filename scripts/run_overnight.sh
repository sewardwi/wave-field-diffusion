#!/usr/bin/env bash
# One-command unattended overnight run for a RunPod box.
#
# Chain:
#   0. Preflight — CUDA visible, CUDA-side check of the AMP self-cond gradient
#      fix (the bug that killed sc09_standard_adaln), and a git-push auth
#      dry-run. Any failure => pod terminates IMMEDIATELY (nothing has run yet;
#      an idle pod would otherwise bill all night).
#   1. SC09 2x2 ablation rerun on the fixed code       (AUTOPUSH after each run)
#   2. CIFAR conditional CFG guidance sweep            (AUTOPUSH after each eval)
#   3. Pod self-terminates via the sweep's SHUTDOWN trap — only after a
#      successful final push, so results can never be lost.
#
# Usage on a fresh pod (from the repo root, cloned with the PAT in the URL):
#   export RUNPOD_API_KEY=<runpod termination key>    # docs/API_KEYS.md (not committed)
#   nohup bash scripts/run_overnight.sh > overnight.log 2>&1 &
#   tail -f overnight.log        # watch the preflight pass, then walk away
#
# Knobs: FAST=0 disables the big-GPU presets; EPOCHS/SCALES etc. pass through
# to the underlying scripts.

set -uo pipefail
cd "$(dirname "$0")/.."

terminate_pod() {
    if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "(RUNPOD_POD_ID / RUNPOD_API_KEY unset — cannot self-terminate; stop the pod manually!)"
        return 0
    fi
    echo "Terminating pod ${RUNPOD_POD_ID} via RunPod API…"
    local payload
    payload=$(printf '{"query":"mutation{podTerminate(input:{podId:\\"%s\\"})}"}' "$RUNPOD_POD_ID")
    curl -s -H "Content-Type: application/json" \
        "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -d "$payload"; echo
}

echo "=== Overnight run preflight — $(date -u +%FT%TZ) ==="

if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "WARNING: RUNPOD_POD_ID / RUNPOD_API_KEY unset — pod will NOT self-terminate when done."
fi

# CUDA + the AMP/self-cond gradient fix, exercised on the actual device.
if ! python scripts/preflight_amp_check.py; then
    echo "PREFLIGHT FAILED — nothing has run; terminating pod to stop billing."
    terminate_pod
    exit 1
fi

# Push auth — discover a bad token NOW, not at 6am with results stranded.
if ! git push --dry-run; then
    echo "PREFLIGHT FAILED: git push auth. Re-clone with the PAT in the remote URL:"
    echo "  git remote set-url origin https://<PAT>@github.com/sewardwi/wave-field-diffusion.git"
    echo "Terminating pod."
    terminate_pod
    exit 1
fi
echo "Preflight complete — starting the overnight chain."

echo; echo "=== Stage 1/2: SC09 2x2 ablation rerun (fixed code) ==="
AUTOPUSH=1 FAST="${FAST:-1}" bash scripts/run_sc09_ablation.sh \
    || echo "!!! SC09 ablation exited nonzero — its per-run results were still pushed; continuing."

echo; echo "=== Stage 2/2: CIFAR guidance sweep (+ final push + self-terminate) ==="
AUTOPUSH=1 SHUTDOWN=1 FAST="${FAST:-1}" bash scripts/run_guidance_sweep.sh
