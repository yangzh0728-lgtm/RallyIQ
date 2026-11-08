#!/usr/bin/env bash
# Download every model the analysis worker needs into $WEIGHTS_DIR (default ./weights):
#   pose.onnx               RTMO (Apache-2.0), variant $POSE_VARIANT = rtmo-s | rtmo-m
#   shuttle/heatmap.pt      shuttle heatmap checkpoint (MIT, rallyiq/perception/shuttle/LICENSE-THIRD-PARTY)
#   feet.tflite             heel / toe landmarks: BlazePose GHUM "full" from the MediaPipe wheel on PyPI (Apache-2.0)
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

# The MediaPipe Linux wheel carries the landmark model as a plain file; only that file is kept (the
# model is the same on every platform, so the wheel's platform is pinned and nothing is installed).
FEET_SHA256=e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f
if [ ! -s "$WEIGHTS_DIR/feet.tflite" ]; then
  tmp="$(mktemp -d)"
  python -m pip download -q --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 \
    --python-version 3.12 "mediapipe==0.10.21" -d "$tmp"
  python - "$tmp" "$WEIGHTS_DIR/feet.tflite" "$FEET_SHA256" <<'EOF'
import glob, hashlib, sys, zipfile
whl = glob.glob(sys.argv[1] + "/mediapipe-*.whl")[0]
data = zipfile.ZipFile(whl).read("mediapipe/modules/pose_landmark/pose_landmark_full.tflite")
if hashlib.sha256(data).hexdigest() != sys.argv[3]:
    sys.exit("feet.tflite: unexpected checksum")
open(sys.argv[2], "wb").write(data)
EOF
  rm -rf "$tmp"
fi
echo "feet: $WEIGHTS_DIR/feet.tflite"

if [ ! -s "$WEIGHTS_DIR/shuttle/heatmap.pt" ]; then
  python -m pip install -q gdown
  python scripts/fetch_shuttle_weights.py --dest "$WEIGHTS_DIR/shuttle"
fi
echo "shuttle: $WEIGHTS_DIR/shuttle/heatmap.pt"
