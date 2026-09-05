#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v gcc >/dev/null 2>&1; then
  GOMP_PATH="$(dirname "$(gcc -print-file-name=libgomp.so)")"
  export LD_LIBRARY_PATH="$GOMP_PATH${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${PYTHON:-python}" main.py