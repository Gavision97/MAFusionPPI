#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition main
#SBATCH --qos normal

#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -o /home/gavrilev/DLResearchPPI/PPI Research/src_/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

echo Wed 25 Mar 2026 16:50:34 IST
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
while true; do
        eval $launch
done
