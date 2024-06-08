#!/bin/bash
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --verbose                     # Verbose output
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=30           # Number of CPUs per task
#SBATCH --mem=0

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu12/R/4.2.1/lib64/R/lib:$LD_LIBRARY_PATH

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

for i in {0..4}
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$s end=$e&
done
wait

for i in {5..9}
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runICATLASSO_jgl_mu.py l=$1 alias="X12_joint_all_mus_reduction_scaling" start=$s end=$e&
done