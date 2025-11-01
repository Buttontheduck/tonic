#!/bin/bash

# Parameters
#SBATCH --array=0-23%24
#SBATCH --cpus-per-task=12
#SBATCH --error=/pfs/data6/home/ka/ka_anthropomatik/ka_ln2554/mpo_diff/tonic/outputs/training/2025-07-22/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_tonic
#SBATCH --mem-per-gpu=8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/pfs/data6/home/ka/ka_anthropomatik/ka_ln2554/mpo_diff/tonic/outputs/training/2025-07-22/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu_a100_il,gpu_h100_il,gpu_h100
#SBATCH --signal=USR2@120
#SBATCH --time=2880
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /pfs/data6/home/ka/ka_anthropomatik/ka_ln2554/mpo_diff/tonic/outputs/training/2025-07-22/.submitit/%A_%a/%A_%a_%t_log.out --error /pfs/data6/home/ka/ka_anthropomatik/ka_ln2554/mpo_diff/tonic/outputs/training/2025-07-22/.submitit/%A_%a/%A_%a_%t_log.err /home/ka/ka_anthropomatik/ka_ln2554/anaconda3/envs/gin_tonic/bin/python -u -m submitit.core._submit /pfs/data6/home/ka/ka_anthropomatik/ka_ln2554/mpo_diff/tonic/outputs/training/2025-07-22/.submitit/%j
