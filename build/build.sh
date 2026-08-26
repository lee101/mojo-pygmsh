#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/dist"
mojo build --emit shared-lib "$root/src/capi.mojo" -o "$root/dist/libmojo-pygmsh.so"
if mojo build --emit shared-lib "$root/src/gpu.mojo" \
     -o "$root/dist/libmojo-pygmsh-gpu.so" 2>"$root/dist/gpu-build.log"; then
    echo "built dist/libmojo-pygmsh-gpu.so"
else
    echo "GPU kernels not built (see dist/gpu-build.log); CPU path is unaffected" >&2
fi
