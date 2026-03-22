#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=7-00:00:00
#SBATCH --job-name=research_project
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

EXP_ID=2
JOB_ID="2003@j2"
DATE="2003"

EVAL_METHOD="cv_cold"
DEVICE="cuda"

USE_STRUCT="False"
SAVE_PROBS="False"
N_EXP=10



RES_FILE_NAME="cold_both_eval_${DATE}"
LOG_FILE_NAME="${DATE}_cold_both_eval_exp${EXP_ID}"
EXP_LOG_DIR="2003"

LOG_MSG="${DATE} - cold start evaluation ${JOB_ID}"

################  Following lines will be executed by a compute node  ################

echo "$(date)"
echo -e "\nSLURM_JOBID:\t\t$SLURM_JOBID"
echo -e "SLURM_JOB_NODELIST:\t$SLURM_JOB_NODELIST\n"

module load anaconda
module load cuda/12.1

source activate chemprop
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

nvcc --version
nvidia-smi

python main.py \
  --eval_method "$EVAL_METHOD" \
  --job_id "$JOB_ID" \
  --use_struct "$USE_STRUCT" \
  --save_probs "$SAVE_PROBS" \
  --n_exp "$N_EXP" \
  --log_msg "$LOG_MSG" \
  --res_file_name "$RES_FILE_NAME" \
  --log_file_name "$LOG_FILE_NAME" \
  --exp_log_dir "$EXP_LOG_DIR" \
  --device "$DEVICE"