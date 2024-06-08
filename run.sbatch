#!/bin/bash
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --verbose                     # Verbose output
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=4           # Number of CPUs per task
#SBATCH --mem-per-cpu=MaxMemPerCPU   # Replace MAX_MEM with the maximum memory per CPU

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$2 end=$(($2+1))
