#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --time 24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=bash
#SBATCH --output=log.out
#SBATCH --verbose
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpunode
module load gnu12
module load R

source activate mvtlasso

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

python3.11 runICATLASSO_jgl_mu.py
