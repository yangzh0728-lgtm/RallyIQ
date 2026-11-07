#!/usr/bin/env bash
# Start RallyIQ locally: API + analysis worker + web UI in one process on http://localhost:8000.
# Run scripts/setup_local.sh once first. Data (SQLite + videos + results) goes to $RALLYIQ_DATA
# (default ./data). Env: PORT (default 8000), HOST (default 127.0.0.1; 0.0.0.0 to open it to your LAN).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/uvicorn ] || { echo "先运行 bash scripts/setup_local.sh" >&2; exit 1; }
[ -f web/dist/index.html ] || echo "提示：web/dist 不存在，只启动 API（网页界面可用 cd web && npm run dev 单独启动）" >&2
export RALLYIQ_WEB_DIR="${RALLYIQ_WEB_DIR:-$PWD/web/dist}"
export RALLYIQ_DATA="${RALLYIQ_DATA:-$PWD/data}"
echo "RallyIQ: http://localhost:${PORT:-8000}   数据目录: $RALLYIQ_DATA   (Ctrl+C 停止)"
exec .venv/bin/uvicorn rallyiq.api.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
