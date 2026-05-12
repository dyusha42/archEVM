#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP="${PROJ_DIR}/build/bin/neon_benchmark"

if [ ! -x "$APP" ]; then
    echo "[run] Бинарь не найден ($APP). Сначала запустите build.sh."
    exit 1
fi

cd "${PROJ_DIR}/build/bin"
"$APP" "$@"

HTML_SRC="${PROJ_DIR}/build/bin/neon_benchmark_chart.html"
HTML_DST="${PROJ_DIR}/neon_benchmark_chart.html"
if [ -f "$HTML_SRC" ]; then
    cp -f "$HTML_SRC" "$HTML_DST"
    echo "[run] График скопирован: $HTML_DST"
fi
