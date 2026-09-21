#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash run_G0C_forensic_comparison.sh G0_WORKDIR G0B_WORKDIR OUTPUT_DIR" >&2
  exit 2
fi

G0_WORKDIR=$1
G0B_WORKDIR=$2
OUTPUT_DIR=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python -u "$HERE/INVEST_G0C_forensic_comparison.py" \
  --g0-workdir "$G0_WORKDIR" \
  --g0b-workdir "$G0B_WORKDIR" \
  --output-dir "$OUTPUT_DIR"
