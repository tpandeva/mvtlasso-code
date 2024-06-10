#!/bin/bash
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --error=err_%j.err
#SBATCH --verbose                     # Verbose output
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=4           # Number of CPUs per task
#SBATCH --mem=16g

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu12/R/4.2.1/lib64/R/lib:$LD_LIBRARY_PATH

cd /zfs/ivi/causality/tpandeva/mvtlasso-code


s=$((8+$2))
e=$((9+$i))
python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$s end=$e
wait

s=$((9+$2))
e=$((10+$i))
python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$s end=$e
