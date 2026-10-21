#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python_bin="${PYTHON_BIN:-python}"
o3_threads="${O3_THREADS:-32}"
o3_memory_mb="${O3_MEMORY_MB:-120000}"
o3_workdir="${O3_WORKDIR:-O3_gate4A_work}"

"$python_bin" -u O3_gate4A_run.py \
  --workdir "$o3_workdir" \
  --basis cc-pvtz \
  --ncas 12 \
  --nelecas 18 \
  --r-sum 2.7288 \
  --theta0 119.70 \
  --delta-r-step 0.010 \
  --theta-step 0.20 \
  --threads "$o3_threads" \
  --memory-mb "$o3_memory_mb" \
  2>&1 | tee O3_gate4A.stdout

"$python_bin" -u O3_gate4A_analyze.py \
  --workdir "$o3_workdir" \
  2>&1 | tee O3_gate4A_analysis.stdout

