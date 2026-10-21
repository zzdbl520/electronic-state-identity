#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GATE4B_WORKDIR="${GATE4B_WORKDIR:-/mnt/O3/O3_gate4B/O3_gate4B_work_v5}"
PILOT_WORKDIR="${PILOT_WORKDIR:-/mnt/O3/O3_weighted_SA2_pilot_work}"
THREADS="${THREADS:-32}"
MEMORY_MB="${MEMORY_MB:-120000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/O3_weighted_SA2_pilot_analyze.py" --self-test

"${PYTHON_BIN}" "${SCRIPT_DIR}/O3_weighted_SA2_pilot_run.py" \
  --gate4b-workdir "${GATE4B_WORKDIR}" \
  --workdir "${PILOT_WORKDIR}" \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}" \
  --dry-run

"${PYTHON_BIN}" "${SCRIPT_DIR}/O3_weighted_SA2_pilot_run.py" \
  --gate4b-workdir "${GATE4B_WORKDIR}" \
  --workdir "${PILOT_WORKDIR}" \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/O3_weighted_SA2_pilot_analyze.py" \
  --workdir "${PILOT_WORKDIR}"

echo "RESULT: ${PILOT_WORKDIR}/O3_weighted_SA2_summary.json"
