#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python_bin="${PYTHON_BIN:-python}"
o3_threads="${O3_THREADS:-32}"
o3_memory_mb="${O3_MEMORY_MB:-120000}"
o3_workdir="${O3_GATE4B_WORK:-O3_gate4B_work_v5}"

if [[ -n "${O3_GATE4A_WORK:-}" ]]; then
  seed_workdir="$O3_GATE4A_WORK"
elif [[ -d "../O3_gate4A_work/wavefunctions" ]]; then
  seed_workdir="../O3_gate4A_work"
elif [[ -d "../O3_gate4A/O3_gate4A_work/wavefunctions" ]]; then
  seed_workdir="../O3_gate4A/O3_gate4A_work"
else
  echo "Gate 4A seed directory not found." >&2
  echo "Set O3_GATE4A_WORK to the directory containing wavefunctions/." >&2
  exit 2
fi

"$python_bin" -u O3_gate4B_run.py \
  --seed-workdir "$seed_workdir" \
  --workdir "$o3_workdir" \
  --basis cc-pvtz \
  --ncas 12 \
  --nelecas 18 \
  --r-sum 2.7288 \
  --theta0 119.70 \
  --delta-r-steps 0.010,0.005,0.0025,0.00125 \
  --theta-steps 0.20,0.10,0.05,0.025 \
  --rhf-conv-tol 1.0e-12 \
  --fci-conv-tol 1.0e-12 \
  --mc-conv-tol 1.0e-10 \
  --mc-stage1-conv-tol-grad 1.0e-5 \
  --mc-conv-tol-grad 1.0e-5 \
  --mc-max-cycle-macro 60 \
  --mc-stationary-window 5 \
  --mc-stationary-energy-tol 1.0e-10 \
  --mc-stationary-grad-ceiling 1.0e-4 \
  --mc-stationary-ddm-tol 1.0e-7 \
  --mc-stationary-maxrot-tol 1.0e-7 \
  --mc-newton-max-cycle 12 \
  --no-mc-newton-polish \
  --threads "$o3_threads" \
  --memory-mb "$o3_memory_mb" \
  2>&1 | tee O3_gate4B.stdout

"$python_bin" -u O3_gate4B_analyze.py \
  --workdir "$o3_workdir" \
  2>&1 | tee O3_gate4B_analysis.stdout

if "$python_bin" -c "import matplotlib" >/dev/null 2>&1; then
  "$python_bin" -u O3_gate4B_plot.py \
    --workdir "$o3_workdir" \
    2>&1 | tee O3_gate4B_plot.stdout
else
  echo "matplotlib is unavailable; numerical analysis completed and plotting was skipped." \
    | tee O3_gate4B_plot.stdout
fi
