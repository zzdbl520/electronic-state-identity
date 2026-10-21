#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  qualify|history|sa3) ;;
  *)
    echo "Usage: $0 {qualify|history|sa3}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

cpdc_python="${CPDC_OPH_PYTHON:-python}"
cpdc_work_base="${CPDC_WORK_BASE:-$script_dir/../work}"
cpdc_root="${CPDC_OPH_ROOT:-$cpdc_work_base/CPDC_ccpVTZ_rank2_opposed_history_v1_work}"
cpdc_source_root="${CPDC_OPH_SOURCE_ROOT:-$cpdc_work_base/CPDC_secondary_CI_ccpVTZ_lift_v1_work}"
cpdc_loop_root="${CPDC_OPH_LOOP_ROOT:-$cpdc_work_base/CPDC_ccpVTZ_relaxed_multiring_v1_work}"
cpdc_screen_root="${CPDC_OPH_SCREEN_ROOT:-$cpdc_work_base/CPDC_ccpVTZ_central_gap_landscape_v1_work}"
cpdc_rank2_root="${CPDC_OPH_RANK2_ROOT:-$cpdc_work_base/CPDC_ccpVTZ_rank2_projector_v1_work}"
history_threads="${CPDC_OPH_HISTORY_THREADS:-12}"
history_memory_mb="${CPDC_OPH_HISTORY_MEMORY_MB:-200000}"
sa3_threads="${CPDC_OPH_SA3_THREADS:-6}"
sa3_memory_mb="${CPDC_OPH_SA3_MEMORY_MB:-100000}"
sa3_max_jobs="${CPDC_OPH_SA3_MAX_JOBS:-4}"
log_file="CPDC_ccpVTZ_rank2_opposed_history_v1_${mode}.stdout"

sha256sum -c SHA256SUMS_CPDC_RANK2_OPPOSED_HISTORY_v1
"$cpdc_python" CPDC_ccpVTZ_rank2_opposed_history_v1.py selftest

required_files=(
  "$cpdc_source_root/CPDC_secondary_CI_ccpVTZ_lift_v1_decision.json"
  "$cpdc_loop_root/CPDC_ccpVTZ_loop_geometry_v1_decision.json"
  "$cpdc_screen_root/CPDC_ccpVTZ_central_gap_landscape_v1_decision.json"
  "$cpdc_rank2_root/CPDC_ccpVTZ_rank2_projector_v1_decision.json"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required antecedent: $required_file" >&2
    exit 3
  fi
done

if [[ "$mode" != "qualify" ]]; then
  "$cpdc_python" - <<'PY'
import pyscf
if pyscf.__version__ != "2.13.1":
    raise SystemExit(
        "CPDC opposed-history protocol requires exact PySCF 2.13.1; "
        f"found {pyscf.__version__}"
    )
print(f"CPDC_PYSCF_VERSION={pyscf.__version__}")
PY
fi

mkdir -p "$cpdc_root"
command=(
  "$cpdc_python" -u CPDC_ccpVTZ_rank2_opposed_history_v1.py "$mode"
  --root "$cpdc_root"
  --source-root "$cpdc_source_root"
  --loop-root "$cpdc_loop_root"
  --screen-root "$cpdc_screen_root"
  --rank2-root "$cpdc_rank2_root"
)
if [[ "$mode" == "history" ]]; then
  command+=(
    --basis cc-pvtz
    --threads "$history_threads"
    --memory-mb "$history_memory_mb"
  )
elif [[ "$mode" == "sa3" ]]; then
  command+=(
    --basis cc-pvtz
    --threads "$sa3_threads"
    --memory-mb "$sa3_memory_mb"
    --max-jobs "$sa3_max_jobs"
  )
fi

{
  echo
  echo "===== launch $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$mode ====="
  echo "root=$cpdc_root"
  echo "source_root=$cpdc_source_root"
  echo "loop_root=$cpdc_loop_root"
  echo "screen_root=$cpdc_screen_root"
  echo "rank2_root=$cpdc_rank2_root"
  if [[ "$mode" == "history" ]]; then
    echo "parallel_histories=2 threads_per_history=$history_threads memory_mb_per_history=$history_memory_mb"
  elif [[ "$mode" == "sa3" ]]; then
    echo "threads_per_job=$sa3_threads memory_mb_per_job=$sa3_memory_mb max_jobs=$sa3_max_jobs"
  fi
} >> "$log_file"

nohup "${command[@]}" >> "$log_file" 2>&1 &
pid=$!

echo "Started CPDC opposed-history $mode: PID=$pid"
echo "Root:        $cpdc_root"
echo "Source root: $cpdc_source_root"
echo "Loop root:   $cpdc_loop_root"
echo "Screen root: $cpdc_screen_root"
echo "Rank-2 root: $cpdc_rank2_root"
echo "Log:         $script_dir/$log_file"
echo "Follow with: tail -f $log_file"
