#!/usr/bin/env bash
#SBATCH --job-name=O3_gate4A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=O3_gate4A.slurm.%j.out
#SBATCH --error=O3_gate4A.slurm.%j.err

set -euo pipefail

# Edit the following two lines for your cluster if needed.
source "${HOME}/.bashrc"
conda activate pyscf

export O3_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export O3_MEMORY_MB=120000

bash run_O3_gate4A.sh

