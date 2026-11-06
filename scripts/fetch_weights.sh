#!/usr/bin/env bash
# Download every model the analysis worker needs into $WEIGHTS_DIR (default ./weights):
#   pose.onnx               RTMO (Apache-2.0), variant $POSE_VARIANT = rtmo-s | rtmo-m
#   shuttle/heatmap.pt      shuttle heatmap checkpoint (MIT, rallyiq/perception/shuttle/LICENSE-THIRD-PARTY)
# Used by docker/worker.Dockerfile; scripts/mac_e2e.sh does the same for a local Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS_DIR="${WEIGHTS_DIR:-$PWD/weights}"
POSE_VARIANT="${POSE_VARIANT:-rtmo-s}"
mkdir -p "$WEIGHTS_DIR/shuttle"

case "$POSE_VARIANT" in
  rtmo-s) POSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip" ;;
  rtmo-m) POSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip" ;;
  *) echo "unknown POSE_VARIANT=$POSE_VARIANT" >&2; exit 2 ;;
esac
if [ ! -s "$WEIGHTS_DIR/pose.onnx" ]; then
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/pose.zip" "$POSE_URL"
  unzip -q -o "$tmp/pose.zip" -d "$tmp/x"
  f="$(find "$tmp/x" -name end2end.onnx | head -1)"
  [ -n "$f" ] || { echo "end2end.onnx not found in $POSE_URL" >&2; exit 1; }
  mv "$f" "$WEIGHTS_DIR/pose.onnx"
  rm -rf "$tmp"
fi
echo "pose: $WEIGHTS_DIR/pose.onnx"

if [ ! -s "$WEIGHTS_DIR/shuttle/heatmap.pt" ]; then
  python -m pip install -q gdown
  python scripts/fetch_shuttle_weights.py --dest "$WEIGHTS_DIR/shuttle"
fi
echo "shuttle: $WEIGHTS_DIR/shuttle/heatmap.pt"
