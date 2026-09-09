#!/usr/bin/env bash
# Fetch the third-party task data this project builds on (CL-ALFRED splits, ALFRED data).
# The VirtualHome python package is bundled under externals/virtualhome.
# Run from the repository root. See externals/README.md for what each piece is.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

CL_ALFRED_REPO="https://github.com/snumprlab/cl-alfred.git"

# ------------------------------------------------------------------ CL-ALFRED
# GPL-3.0: fetched, never redistributed in this repository.
if [ -d externals/cl-alfred/.git ]; then
    echo "[skip] externals/cl-alfred already present"
else
    echo "[fetch] CL-ALFRED continual splits (sparse: embodied_split/ only, the full repo is ~10GB)"
    git clone --depth 1 --filter=blob:none --sparse "$CL_ALFRED_REPO" externals/cl-alfred
    git -C externals/cl-alfred sparse-checkout set embodied_split
fi

# ------------------------------------------------------------- ALFRED / ALFWorld
# Set SKIP_ALFRED=1 to stop here (VirtualHome-only setup; the ALFRED data is ~2GB).
if [ "${SKIP_ALFRED:-0}" = "1" ]; then
    echo "[skip] ALFRED task data (SKIP_ALFRED=1)"
elif [ -d externals/alfworld/json_2.1.1 ]; then
    echo "[skip] externals/alfworld/json_2.1.1 already present"
else
    echo "[fetch] ALFRED task data (json_2.1.1)"
    if ! command -v alfworld-download >/dev/null 2>&1; then
        echo "  alfworld-download not on PATH. Install ALFWorld first:" >&2
        echo "    pip install alfworld" >&2
        exit 1
    fi
    ALFWORLD_DATA="$ROOT/externals/alfworld" alfworld-download
fi

echo
echo "Done. Next: download the VirtualHome Unity executable (see externals/README.md)."
