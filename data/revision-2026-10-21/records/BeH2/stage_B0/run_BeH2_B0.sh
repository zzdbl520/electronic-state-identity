#!/usr/bin/env bash
set -euo pipefail

threads="${BEH2_B0_THREADS:-64}"
memory_mb="${BEH2_B0_MEMORY_MB:-100000}"
workdir="${BEH2_B0_WORKDIR:-BeH2_B0_work}"

python BeH2_B0_stationary_solutions.py \
  --workdir "$workdir" \
  --threads "$threads" \
  --memory-mb "$memory_mb" \
  2>&1 | tee BeH2_B0.stdout

python BeH2_B0_analyze.py \
  --workdir "$workdir" \
  2>&1 | tee BeH2_B0_analysis.stdout

