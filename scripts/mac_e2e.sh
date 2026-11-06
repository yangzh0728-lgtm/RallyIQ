#!/usr/bin/env bash
# RallyIQ: one-command setup + first real run on macOS (Apple Silicon or Intel).
#
#   curl -fsSL https://raw.githubusercontent.com/yangzh0728-lgtm/RallyIQ/main/scripts/mac_e2e.sh | bash
#   # or, inside a checkout:
#   bash scripts/mac_e2e.sh
#
# Steps (each is skipped when already done, so re-running is cheap):
#   1. clone / update the repo into $RALLYIQ_DIR (default ~/RallyIQ)
#   2. Homebrew deps: ffmpeg, yt-dlp; Python >= 3.11
#   3. .venv with the package + evaluation deps (+ module extras when they exist)
#   4. model weights into weights/ (RTMO-s pose, Apache-2.0)
#   5. pull a few BWF eval clips (only the 90 s sections; personal research use only,
#      see evaluation/README.md for the copyright note)
#   6. run the full pipeline on them -> pre-annotations + report
#
# Environment overrides:
#   RALLYIQ_DIR   checkout location            (default ~/RallyIQ)
#   RALLYIQ_REF   branch / ref to run          (default main)
#   CLIPS         space-separated clip ids     (default: 3 eval clips, see below)
#   POSE_VARIANT  rtmo-s | rtmo-m              (default rtmo-s)
#   FAR_BOOST=1   second pose pass for small far-side players
#   SKIP_DOWNLOAD=1  use only clips already in ~/rallyiq-data/bwf/clips
set -euo pipefail

RALLYIQ_DIR="${RALLYIQ_DIR:-$HOME/RallyIQ}"
RALLYIQ_REF="${RALLYIQ_REF:-main}"
CLIPS="${CLIPS:-ae26_ms_f wc25_ms_f ko26_court3}"
POSE_VARIANT="${POSE_VARIANT:-rtmo-s}"
DATA="$HOME/rallyiq-data"
REPO_URL="https://github.com/yangzh0728-lgtm/RallyIQ.git"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[失败] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. repo
say "1/6 代码：$RALLYIQ_DIR ($RALLYIQ_REF)"
if [ ! -d "$RALLYIQ_DIR/.git" ]; then
  git clone "$REPO_URL" "$RALLYIQ_DIR"
fi
cd "$RALLYIQ_DIR"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "工作区有未提交的修改，跳过更新，直接用当前代码。"
else
  git fetch -q origin "$RALLYIQ_REF"
  git checkout -q -B "$RALLYIQ_REF" "origin/$RALLYIQ_REF" 2>/dev/null || git checkout -q FETCH_HEAD
fi
git log -1 --oneline

# ---------------------------------------------------------------- 2. system deps
say "2/6 系统依赖：ffmpeg、yt-dlp、Python ≥ 3.11"
if ! command -v brew >/dev/null; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew "$HOME/.homebrew/bin/brew"; do
    [ -x "$b" ] && eval "$("$b" shellenv)" && break
  done
fi
need=()
command -v ffmpeg >/dev/null || need+=(ffmpeg)
command -v yt-dlp >/dev/null || need+=(yt-dlp)
if [ ${#need[@]} -gt 0 ]; then
  command -v brew >/dev/null || die "缺少 ${need[*]}，且没找到 Homebrew。请先安装 https://brew.sh"
  brew install "${need[@]}"
fi

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  command -v brew >/dev/null || die "需要 Python ≥ 3.11"
  brew install python@3.12
  PY="$(brew --prefix)/bin/python3.12"
fi
echo "Python: $PY ($("$PY" --version))"

# ---------------------------------------------------------------- 3. venv
say "3/6 Python 环境 .venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
extras="dev"
for e in players vision shuttle calibration; do
  python - "$e" <<'EOF' && extras="$extras,$e" || true
import sys, tomllib
sys.exit(0 if sys.argv[1] in tomllib.load(open("pyproject.toml", "rb"))["project"].get("optional-dependencies", {}) else 1)
EOF
done
echo "extras: $extras"
python -m pip install -q -e ".[$extras]"
[ -f evaluation/requirements.txt ] && python -m pip install -q -r evaluation/requirements.txt

# ---------------------------------------------------------------- 4. weights
say "4/6 模型权重 weights/"
mkdir -p weights
case "$POSE_VARIANT" in
  rtmo-s) POSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip" ;;
  rtmo-m) POSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip" ;;
  *) die "未知 POSE_VARIANT=$POSE_VARIANT" ;;
esac
POSE_ONNX="weights/${POSE_VARIANT}_body7-640x640.onnx"
if [ ! -s "$POSE_ONNX" ]; then
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/pose.zip" "$POSE_URL" || die "姿态模型下载失败：$POSE_URL"
  unzip -q -o "$tmp/pose.zip" -d "$tmp/x"
  f="$(find "$tmp/x" -name end2end.onnx | head -1)"
  [ -n "$f" ] || die "压缩包里没有 end2end.onnx"
  mv "$f" "$POSE_ONNX"
  rm -rf "$tmp"
fi
export RALLYIQ_POSE_MODEL="$PWD/$POSE_ONNX"
echo "pose: $RALLYIQ_POSE_MODEL"
# Other modules ship their own fetch script when they need weights.
for s in scripts/fetch_*_weights.sh; do
  [ -f "$s" ] && { echo "运行 $s"; bash "$s"; }
done
[ -f weights/env.sh ] && source weights/env.sh   # module scripts may export model paths here

# ---------------------------------------------------------------- 5. clips
say "5/6 BWF 评测片段：$CLIPS"
# shellcheck disable=SC2086
if [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then
  python -m evaluation.tools.extract_clips --only $CLIPS
else
  python -m evaluation.tools.extract_clips --download --only $CLIPS
fi

# ---------------------------------------------------------------- 6. run
say "6/6 运行完整流水线"
OUT="$DATA/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
args=(--out "$OUT" --only $CLIPS)
[ "${FAR_BOOST:-0}" = 1 ] && args+=(--far-boost)
set +e
python scripts/real_run.py "${args[@]}" 2>&1 | tee "$OUT.log"
rc=${PIPESTATUS[0]}
set -e
ln -sfn "$OUT" "$DATA/runs/latest"

say "完成（退出码 $rc）"
echo "报告：      $OUT/report.md"
echo "预标注：    $OUT/preds/<clip_id>.json"
echo "日志：      $OUT.log"
echo "标注下一步：source $RALLYIQ_DIR/.venv/bin/activate && cd $RALLYIQ_DIR && \\"
echo "            python -m evaluation.tools.label_tool ${CLIPS%% *} --prefill $OUT/preds/${CLIPS%% *}.json"
echo
cat "$OUT/report.md"
exit "$rc"
