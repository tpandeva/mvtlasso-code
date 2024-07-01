#!/bin/bash
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --error=err_%j.err
#SBATCH --verbose                     # Verbose output
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=10           # Number of CPUs per task
#SBATCH --mem=10g

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu12/R/4.2.1/lib64/R/lib:$LD_LIBRARY_PATH

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

s=0
e=99
m=$(($1+1))
#python runICATLASSO_simulated.py l=$1 alias="X12_simulated_icatlasso_n_100_no_scaling_no_corr" start=$s end=$e scaling=true corr=false scaling=false n=100 d1=100 d2=0 k="[50]"&
python runTLASSO_simulated.py l=$1 alias="X12_simulated_tlasso_n_100_no_scaling_no_corr_with_mu" start=$s end=$e scaling=true corr=false scaling=false n=100 d1=100 d2=0 k="[50]"&
python runTLASSO_simulated.py l=$m alias="X12_simulated_tlasso_n_100_no_scaling_no_corr" start=$s end=$e scaling=true corr=false scaling=false n=100 d1=100 d2=0 k="[50]"&

wait