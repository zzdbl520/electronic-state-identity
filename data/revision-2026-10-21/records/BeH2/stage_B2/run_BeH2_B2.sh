#!/usr/bin/env bash
set -uo pipefail

B1_WORKDIR="${1:-BeH2_B1_work}"
B2_WORKDIR="${2:-BeH2_B2_work}"

THREADS="${BEH2_B2_THREADS:-64}"
MEMORY_MB="${BEH2_B2_MEMORY_MB:-100000}"

python BeH2_B2_refine.py \
  --b1-workdir "${B1_WORKDIR}" \
  --workdir "${B2_WORKDIR}" \
  --lambda-min 2.5200 \
  --lambda-max 2.5300 \
  --step 0.0005 \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}" \
  2>&1 | tee BeH2_B2.stdout
run_status=${PIPESTATUS[0]}

if [[ ${run_status} -ne 0 && ${run_status} -ne 2 ]]; then
  exit "${run_status}"
fi

python BeH2_B2_plaquettes.py \
  --workdir "${B2_WORKDIR}" \
  2>&1 | tee BeH2_B2_analysis.stdout
analysis_status=${PIPESTATUS[0]}

if [[ ${analysis_status} -ne 0 ]]; then
  exit "${analysis_status}"
fi

exit "${run_status}"
