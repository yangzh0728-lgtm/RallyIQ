#!/usr/bin/env bash
# RallyIQ local install (macOS or Linux). Run from the repo root:
#
#   bash scripts/setup_local.sh          # then: bash scripts/start_local.sh
#
# Steps (each is skipped when already done, so re-running is cheap):
#   1. check Python >= 3.11, ffmpeg/ffprobe, Node.js >= 20
#   2. .venv with RallyIQ and the analysis models' dependencies (PyTorch, ONNX Runtime, OpenCV)
#   3. model weights into weights/ (RTMO pose: Apache-2.0; shuttle heatmap net: MIT)
#   4. build the web UI into web/dist
#
# Env overrides: PYTHON (interpreter to use), POSE_VARIANT=rtmo-s|rtmo-m, SKIP_WEIGHTS=1, SKIP_WEB=1,
# EXTRAS (pip extras, default "calibration,players,shuttle").
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[失败] %s\033[0m\n' "$*" >&2; exit 1; }
os="$(uname -s)"
hint() {
  if [ "$os" = Darwin ]; then echo "  macOS:  brew install $1"; else echo "  Ubuntu/Debian:  sudo apt install $2"; fi
}

# ---------------------------------------------------------------- 1. system deps
say "1/4 检查系统依赖"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
[ -n "$PY" ] || { hint "python@3.12" "python3.12 python3.12-venv"; die "需要 Python 3.11 或更高版本"; }
echo "python: $("$PY" --version) ($PY)"
for b in ffmpeg ffprobe; do
  command -v "$b" >/dev/null || { hint ffmpeg ffmpeg; die "缺少 $b"; }
done
echo "ffmpeg: $(ffmpeg -version | head -1)"
if [ "${SKIP_WEB:-0}" != 1 ]; then
  command -v node >/dev/null && node -e 'process.exit(Number(process.versions.node.split(".")[0]) < 20 ? 1 : 0)' \
    || { [ "$os" = Darwin ] && echo "  macOS:  brew install node" \
           || echo "  Linux:  发行版自带的 nodejs 往往太旧，建议用 nvm：https://github.com/nvm-sh/nvm 然后 nvm install 22"; die "需要 Node.js 20 或更高版本（只用来构建网页界面；可设 SKIP_WEB=1 跳过）"; }
  echo "node: $(node --version)"
fi

# ---------------------------------------------------------------- 2. python env
say "2/4 Python 环境 .venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
EXTRAS="${EXTRAS:-calibration,players,shuttle}"
.venv/bin/python -m pip install -q -e ".[$EXTRAS]"
.venv/bin/python - <<'PY'
import torch, onnxruntime as ort
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__} -> {dev}; onnxruntime providers: {', '.join(ort.get_available_providers())}")
PY

# ---------------------------------------------------------------- 3. weights
if [ "${SKIP_WEIGHTS:-0}" = 1 ]; then
  say "3/4 跳过模型权重（SKIP_WEIGHTS=1）"
else
  say "3/4 模型权重 weights/"
  PATH="$PWD/.venv/bin:$PATH" WEIGHTS_DIR="$PWD/weights" bash scripts/fetch_weights.sh
fi

# ---------------------------------------------------------------- 4. web UI
if [ "${SKIP_WEB:-0}" = 1 ]; then
  say "4/4 跳过网页构建（SKIP_WEB=1）"
else
  say "4/4 构建网页界面 web/dist"
  (cd web && npm ci --no-audit --no-fund --loglevel=error && npm run build >/dev/null)
fi

say "完成。启动：bash scripts/start_local.sh   然后打开 http://localhost:8000"
