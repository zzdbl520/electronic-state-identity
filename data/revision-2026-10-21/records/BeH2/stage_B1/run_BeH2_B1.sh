#!/usr/bin/env bash
set -uo pipefail

B0_WORKDIR="${1:-BeH2_B0_work}"
B1_WORKDIR="${2:-BeH2_B1_work}"

THREADS="${BEH2_B1_THREADS:-64}"
MEMORY_MB="${BEH2_B1_MEMORY_MB:-100000}"

python BeH2_B1_branch_support.py \
  --b0-workdir "${B0_WORKDIR}" \
  --workdir "${B1_WORKDIR}" \
  --lambda-min 2.40 \
  --lambda-max 2.70 \
  --step 0.01 \
  --center 2.62425 \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}" \
  2>&1 | tee BeH2_B1.stdout
run_status=${PIPESTATUS[0]}

if [[ ${run_status} -ne 0 && ${run_status} -ne 2 ]]; then
  exit "${run_status}"
fi

python BeH2_B1_analyze.py \
  --workdir "${B1_WORKDIR}" \
  2>&1 | tee BeH2_B1_analysis.stdout
analysis_status=${PIPESTATUS[0]}

if [[ ${analysis_status} -ne 0 ]]; then
  exit "${analysis_status}"
fi

exit "${run_status}"
