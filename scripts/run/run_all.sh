#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
model="${1:?Usage: bash scripts/run/run_all.sh MODEL [run_tasks options]}"
shift
exec python scripts/run_tasks.py --model "$model" "$@"
