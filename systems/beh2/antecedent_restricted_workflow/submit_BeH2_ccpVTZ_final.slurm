#!/usr/bin/env bash
#SBATCH --job-name=BeH2_ccpVTZ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=BeH2_ccpVTZ_final.%j.out
#SBATCH --error=BeH2_ccpVTZ_final.%j.err

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BEH2_FINAL_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export BEH2_FINAL_MEMORY_MB="${BEH2_FINAL_MEMORY_MB:-100000}"
bash "${script_dir}/run_BeH2_ccpVTZ_final.sh" "${1:-${script_dir}/BeH2_ccpVTZ_final_work}"
