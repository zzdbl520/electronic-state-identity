#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_root="${1:-${script_dir}/BeH2_ccpVTZ_final_work}"
threads="${BEH2_FINAL_THREADS:-64}"
memory_mb="${BEH2_FINAL_MEMORY_MB:-100000}"
resume="${BEH2_FINAL_RESUME:-0}"

if [[ -e "${work_root}" && -n "$(find "${work_root}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "${resume}" != "1" ]]; then
  echo "ERROR: ${work_root} is not empty. Clean production mode will not reuse it." >&2
  echo "Use a new path, or set BEH2_FINAL_RESUME=1 only to resume this exact run." >&2
  exit 64
fi

mkdir -p "${work_root}/logs" "${work_root}/figures" "${work_root}/mplconfig"
export PYTHONPATH="${script_dir}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${work_root}/mplconfig"
export OMP_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS="${threads}"
export NUMEXPR_NUM_THREADS="${threads}"

run_logged() {
  local label="$1"
  shift
  echo
  echo "===== ${label} ====="
  set +e
  "$@" 2>&1 | tee "${work_root}/logs/${label}.stdout"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "ERROR: ${label} failed with exit code ${status}." >&2
    echo "The work directory is preserved. Fix the cause and resume with BEH2_FINAL_RESUME=1." >&2
    exit "${status}"
  fi
}

echo "BeH2 clean final production rerun"
echo "work root : ${work_root}"
echo "model     : SA3/SA4-CASSCF(6,6)/cc-pVTZ; frozen Be 1s; m=3"
echo "resources : ${threads} threads; ${memory_mb} MB"

run_logged preflight \
  python "${script_dir}/preflight_BeH2_final.py" --output-root "${work_root}"

run_logged B0_run \
  python "${script_dir}/scripts/BeH2_B0_stationary_solutions.py" \
    --workdir "${work_root}/B0" \
    --basis cc-pVTZ --ncas 6 --nelecas 6 --freeze-core \
    --threads "${threads}" --memory-mb "${memory_mb}"
run_logged B0_analyze \
  python "${script_dir}/scripts/BeH2_B0_analyze.py" --workdir "${work_root}/B0"

run_logged B1_run \
  python "${script_dir}/scripts/BeH2_B1_branch_support.py" \
    --b0-workdir "${work_root}/B0" --workdir "${work_root}/B1" \
    --lambda-min 2.40 --lambda-max 2.70 --step 0.01 --center 2.62425 \
    --threads "${threads}" --memory-mb "${memory_mb}"
run_logged B1_analyze \
  python "${script_dir}/scripts/BeH2_B1_analyze.py" --workdir "${work_root}/B1"

run_logged B2_run \
  python "${script_dir}/scripts/BeH2_B2_refine.py" \
    --b1-workdir "${work_root}/B1" --workdir "${work_root}/B2" \
    --lambda-min 2.5200 --lambda-max 2.5300 --step 0.0005 \
    --threads "${threads}" --memory-mb "${memory_mb}"
run_logged B2_analyze \
  python "${script_dir}/scripts/BeH2_B2_plaquettes.py" --workdir "${work_root}/B2"

run_logged B3R_run \
  python "${script_dir}/scripts/BeH2_B3R_scaling.py" \
    --b2-workdir "${work_root}/B2" --workdir "${work_root}/B3R" \
    --lambda-min 2.52120 --lambda-max 2.52220 --step 0.00005 \
    --threads "${threads}" --memory-mb "${memory_mb}"
run_logged B3R_analyze \
  python "${script_dir}/scripts/BeH2_B3R_analyze.py" \
    --workdir "${work_root}/B3R" --strides 1,2,4,8

run_logged Figure5 \
  python "${script_dir}/plotting/plot_BeH2_B3R_validation.py" \
    --data-dir "${work_root}/B3R" \
    --output-prefix "${work_root}/figures/BeH2_B3R_validation"
run_logged FigureS2 \
  python "${script_dir}/plotting/plot_BeH2_B3R_components.py" \
    --data-dir "${work_root}/B3R" \
    --output-prefix "${work_root}/figures/BeH2_B3R_components"

run_logged acceptance \
  python "${script_dir}/validate_BeH2_final.py" --work-root "${work_root}"
run_logged collect \
  python "${script_dir}/collect_BeH2_final_results.py" --work-root "${work_root}"

echo
echo "ALL CHECKS PASSED"
echo "Return bundle: ${work_root}/BeH2_ccpVTZ_final_return.zip"
