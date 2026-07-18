#!/usr/bin/env bash
# Push-pipeline test: creates an EMPTY commit and pushes it, using the same
# git invocations as the autopush functions in the ablation/sweep scripts.
# Run it on a fresh pod right after cloning (and exporting keys) to prove the
# overnight results will actually reach GitHub. Touches no files.
#
# Usage:
#   bash scripts/test_push.sh
#
# On success the empty commit stays in history as a harmless marker of when
# the pod checked in. On failure it is rolled back so retries start clean.

set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== Push pipeline test ==="
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "remote: $(git remote get-url origin | sed -E 's#(https?://)[^@/]*@#\1<token>@#')"

if ! git -c user.name="${GIT_NAME:-runpod}" -c user.email="${GIT_EMAIL:-runpod@pod}" \
        commit --allow-empty -m "Push test: $(date -u +%FT%TZ) [auto] (safe to ignore)"; then
    echo "COMMIT FAILED ✗ — git identity/repo problem; fix before the overnight run."
    exit 1
fi

if out="$(git push 2>&1)"; then
    echo "$out" | tail -2
    echo "PUSH OK ✓ — the overnight autopush will work with this remote/token."
else
    echo "$out" | tail -3
    echo "PUSH FAILED ✗ — fix before starting the overnight run, e.g.:"
    echo "  git remote set-url origin https://<PAT>@github.com/sewardwi/wave-field-diffusion.git"
    echo "then re-run: bash scripts/test_push.sh"
    git reset --quiet HEAD~1   # roll back the empty commit so retries start clean
    exit 1
fi
