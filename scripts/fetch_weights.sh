#!/usr/bin/env bash
# Download every model the analysis worker needs into $WEIGHTS_DIR (default ./weights):
#   pose.onnx               RTMO (Apache-2.0), variant $POSE_VARIANT = rtmo-s | rtmo-m
#   shuttle/heatmap.pt      shuttle heatmap checkpoint (MIT, rallyiq/perception/shuttle/LICENSE-THIRD-PARTY)
#   feet.tflite             heel / toe landmarks: BlazePose GHUM "heavy" from MediaPipe's model bucket (Apache-2.0)
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

# Heel / toe: the landmark model inside MediaPipe's "pose landmarker heavy" bundle (BlazePose GHUM heavy,
# Apache-2.0, 27 MB). It replaces the "full" model older versions of this script put in feet.tflite: on
# amateur video its heel and toe points wander a third less (rallyiq/footwork/README.md).
FEET_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
FEET_SHA256=a0fe0e6cbaeb36b6075fe520cfca38b0e212b258b6c4c6caf407c7ef5e3ff55c
if ! python - "$WEIGHTS_DIR/feet.tflite" "$FEET_SHA256" <<'EOF'
import hashlib, sys
try:
    sys.exit(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])
except OSError:
    sys.exit(1)
EOF
then
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/pose.task" "$FEET_URL"
  python - "$tmp/pose.task" "$WEIGHTS_DIR/feet.tflite" "$FEET_SHA256" <<'EOF'
import hashlib, sys, zipfile
data = zipfile.ZipFile(sys.argv[1]).read("pose_landmarks_detector.tflite")
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
