#!/bin/bash
#SBATCH --cpus-per-task=4             # Allocate 4 cores per task
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --ntasks=10                   # Number of tasks (one for each Python program)
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --verbose                     # Verbose output
#SBATCH --mem=0                       # Use maximum available memory
#SBATCH --ntasks-per-node=1           # Number of tasks per node (can be adjusted as per node capacity)

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate

cd /zfs/ivi/causality/tpandeva/mvtlasso-code
for i in {0..9}
do
    srun --exclusive -N1 -n1 python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$(($2+$i)) end=$(($2+$i+1))
done