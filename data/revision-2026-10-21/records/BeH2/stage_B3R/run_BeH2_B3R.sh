#!/usr/bin/env bash
set -uo pipefail

B2_WORKDIR="${1:-BeH2_B2_work}"
B3R_WORKDIR="${2:-BeH2_B3R_work}"

THREADS="${BEH2_B3R_THREADS:-64}"
MEMORY_MB="${BEH2_B3R_MEMORY_MB:-100000}"

python BeH2_B3R_scaling.py \
  --b2-workdir "${B2_WORKDIR}" \
  --workdir "${B3R_WORKDIR}" \
  --lambda-min 2.52120 \
  --lambda-max 2.52220 \
  --step 0.00005 \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}" \
  2>&1 | tee BeH2_B3R.stdout
run_status=${PIPESTATUS[0]}

if [[ ${run_status} -ne 0 && ${run_status} -ne 2 ]]; then
  exit "${run_status}"
fi

python BeH2_B3R_analyze.py \
  --workdir "${B3R_WORKDIR}" \
  --strides 1,2,4,8 \
  2>&1 | tee BeH2_B3R_analysis.stdout
analysis_status=${PIPESTATUS[0]}

if [[ ${analysis_status} -ne 0 ]]; then
  exit "${analysis_status}"
fi

exit "${run_status}"
