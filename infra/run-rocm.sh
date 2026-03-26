#!/usr/bin/env bash
# Run TurboQuant dev container with AMD ROCm GPU access.
#
# Requires: Podman, /dev/kfd, /dev/dri
# Tested on: Bazzite 43 (Fedora), Radeon 890M (gfx1150)
#
# Usage:
#   ./infra/run-rocm.sh              # interactive shell
#   ./infra/run-rocm.sh pytest tests/ # run a command

set -euo pipefail

IMAGE="turboquant-rocm"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Build if image doesn't exist
if ! podman image exists "$IMAGE" 2>/dev/null; then
    echo "Building $IMAGE from infra/Containerfile.rocm..."
    podman build -t "$IMAGE" -f "$REPO_ROOT/infra/Containerfile.rocm" "$REPO_ROOT"
fi

TTY_FLAG=""
if [ -t 0 ]; then
    TTY_FLAG="-it"
fi

HF_MOUNT=""
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    HF_MOUNT="-v $HF_CACHE:/root/.cache/huggingface:z"
fi

# shellcheck disable=SC2086
exec podman run --rm $TTY_FLAG \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --security-opt=label=disable \
    -v "$REPO_ROOT:/workspace:z" \
    $HF_MOUNT \
    -w /workspace \
    "$IMAGE" \
    "${@:-bash}"
