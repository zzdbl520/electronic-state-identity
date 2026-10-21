#!/usr/bin/env bash
#SBATCH --job-name=O3_gate4B
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=O3_gate4B.slurm.%j.out
#SBATCH --error=O3_gate4B.slurm.%j.err

set -euo pipefail

# Edit these environment lines for the local cluster when necessary.
source "${HOME}/.bashrc"
conda activate pyscf

export O3_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export O3_MEMORY_MB=120000
# Example when Gate 4A was run in /mnt/O3/O3_gate4A:
# export O3_GATE4A_WORK=/mnt/O3/O3_gate4A/O3_gate4A_work

bash run_O3_gate4B.sh
