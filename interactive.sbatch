#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition main
#SBATCH --qos normal
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -o /home/gavrilev/DLResearchPPI/PPI Research/src_/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

#
echo Tue 10 Mar 2026 17:37:50 IST
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
#module load cuda/11.1
while true; do
        eval $launch
done
