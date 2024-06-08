#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --time 24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=bash
#SBATCH --output=log_0.5.out
#SBATCH --verbose
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000
module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

python3.11 runICATLASSO_jgl_mu.py  l=0.5
