#!/usr/bin/env bash
# Shuttle heatmap checkpoint -> weights/shuttle/heatmap.pt (MIT, see
# rallyiq/perception/shuttle/LICENSE-THIRD-PARTY). Called by scripts/mac_e2e.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -s weights/shuttle/heatmap.pt ]; then
  echo "shuttle: weights/shuttle/heatmap.pt already present"
  exit 0
fi
python -m pip install -q gdown
python scripts/fetch_shuttle_weights.py
