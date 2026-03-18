#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name research_project			### name of the job
#SBATCH --output jobs/job-%J.out
#SBATCH --error  jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=1

# Note: the following 4 lines are commented out

#SBATCH --mail-user=gavrilev@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=60G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
module load cuda/12.1
source activate chemprop				### activate a conda environment, replace my_env with your conda environme

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Check CUDA version
nvcc --version

# Check nvidia-smi
nvidia-smi
					### this command executes jupyter lab – replace with your own command
#python mcd_job3.py
jupyter lab
pyhthon main.py --eval_method 'cv_neg_smoo' --job_id '1803@j1@' --use_struct "False" --save_probs "False" --n_exp 10 --log_msg "18/03 - cold start evaluation @ negative factor=5, smoothing factor=0.9" --neg_factor "5" --smoo_factor "0.9"  --res_file_name "cold_eval_neg_smoo_1503" --log_file_name "1803_cold_eval_neg_5_smoo_0.9_exp1" --exp_log_dir "1503" --device 'cuda'
#python main.py --eval_method 'cv_neg_smoo' --job_id '1503@j1' --strct_dataset "dataset1" --strct_strategy "conditional" --strct_aug_train "False" --strct_aug_eval "True" --n_exp 10 --log_msg "15/03 - cold start evaluation @ negative factor=5, smoothing factor=0.9" --neg_factor "5" --smoo_factor "0.9"  --res_file_name "cold_eval_neg_smoo_1503" --log_file_name "1503_cold_eval_neg_5_smoo_0.9_exp1" --exp_log_dir "1503" --device 'cuda'
#python main.py --eval_method 'cold' --n_exp 10 --epo_f1 31 --epo_f2 28 --epo_f3 34 --epo_f4 33 --epo_f5 26 --log_msg "02/03 - cold start experiment @ old hypeparameters - senaty check" --log_file_name "0203_cold_eval_old" --device 'cuda'
