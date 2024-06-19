#!/bin/bash
#SBATCH --time=24:00:00               # Set the maximum run time
#SBATCH --job-name=bash               # Name of the job
#SBATCH --output=R-%x.%j.out          # Output file
#SBATCH --error=err_%j.err
#SBATCH --verbose                     # Verbose output
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=30           # Number of CPUs per task
#SBATCH --mem=30g

module load gnu12
module load R

cd /home/tpandev
source mvtlasso/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu12/R/4.2.1/lib64/R/lib:$LD_LIBRARY_PATH

cd /zfs/ivi/causality/tpandeva/mvtlasso-code

for i in {0..2}
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runTLASSO_simulated.py l=$1 alias="X_sim_tlasso_0.75" start=$s end=$e seed=$s k="[38]"
done
wait
for i in {3..5}
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runTLASSO_simulated.py l=$1 alias="X_sim_tlasso_0.75" start=$s end=$e seed=$s k="[38]"
done
wait
for i in 6 7 8
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runTLASSO_simulated.py l=$1 alias="X_sim_tlasso_0.75" start=$s end=$e seed=$s k="[38]"
done
wait

for i in 9
do
s=$(($i+$2))
e=$(($2+$i+1))
python3.11 runTLASSO_simulated.py l=$1 alias="X_sim_tlasso_0.75" start=$s end=$e seed=$s k="[38]"
done
wait
