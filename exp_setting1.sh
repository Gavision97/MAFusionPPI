#!/bin/bash

################################################################################################
#SBATCH --partition=main
#SBATCH --time=7-00:00:00
#SBATCH --job-name=hyperparam_search
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G
################################################################################################

############################
### Experiment variables ###
############################

EXP_ID=1
JOB_ID="2503@j1"
DATE="2503"

EVAL_METHOD="hyperparam_search"
DEVICE="cuda"

USE_STRUCT="True"
STRCT_DATASET="dataset1"
STRCT_STRATEGY="conditional"
STRCT_AUG_TRAIN="False"
STRCT_AUG_EVAL="False"

HEAD_DROPOUT=0.3
JOIN_ATTN_FEAT="ppiformer"
COMPOUND_DIM=128
HEAD_FUSE="cat"
BATCH_SIZE=64

SAVE_PROBS="False"
N_EXP=1

RES_FILE_NAME="hyperparam_search_${DATE}"
LOG_FILE_NAME="${DATE}_hyperparam_search_${JOB_ID}"
EXP_LOG_DIR="2503"

LOG_MSG="${DATE} - hyperparameter search ${JOB_ID}"




############################
### Hyperparameters block ###
############################

LR="1e-5 5e-5"
WEIGHT_DECAY="1e-3 1e-4 1e-5"
MLP_DROPOUT="0.3"
SELF_ATTN_DROPOUT="0.0 "
COMPOUND_PROJ_DIM="128"
PPI_FUSE_SETTING="gate"
PROJ_FEAT="True False"

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
  --eval_method "${EVAL_METHOD}" \
  --job_id "${JOB_ID}" \
  --use_struct "${USE_STRUCT}" \
  --strct_dataset "${STRCT_DATASET}" \
  --strct_strategy "${STRCT_STRATEGY}" \
  --strct_aug_train "${STRCT_AUG_TRAIN}" \
  --strct_aug_eval "${STRCT_AUG_EVAL}" \
  --save_probs "${SAVE_PROBS}" \
  --n_exp "${N_EXP}" \
  --log_msg "${LOG_MSG}" \
  --res_file_name "${RES_FILE_NAME}" \
  --log_file_name "${LOG_FILE_NAME}" \
  --exp_log_dir "${EXP_LOG_DIR}" \
  --device "${DEVICE}" \
  --head_dropout "${HEAD_DROPOUT}" \
  --join_attn_feat "${JOIN_ATTN_FEAT}" \
  --compound_dim "${COMPOUND_DIM}" \
  --head_fuse "${HEAD_FUSE}" \
  --batch_size "${BATCH_SIZE}" \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --mlp_dropout ${MLP_DROPOUT} \
  --self_attn_dropout ${SELF_ATTN_DROPOUT} \
  --compound_proj_dim ${COMPOUND_PROJ_DIM} \
  --ppi_fuse_setting ${PPI_FUSE_SETTING} \
  --proj_feat ${PROJ_FEAT}