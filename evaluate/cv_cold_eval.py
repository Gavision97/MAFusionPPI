import os
import glob
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from MAFusionPPI.MAFusionPPI import MAFusionPPI, MFusionPPI
from MAFusionPPI.MAStructFusionPPI import choose_model_setting
from utils.tools import plot_train_val_auc, set_seed, preprocess_dataset

#DATA_PATH = 'datasets/splits_with_ppimi_test_folds_s3_corrected_'
#DATA_PATH = 'datasets/train_test_5_0.75'
#DATA_PATH = 'datasets/cold_both_folds' # j1
#DATA_PATH = 'datasets/multi_ppimi_cold_both' # j2

DATA_PATH = 'datasets/multi_ppimi_s4_tests_splits_with_my_train' # j3
UNIPROT_MAPPING_PATH = 'datasets/idmapping_unip.tsv'

# best hyperparameters; extracted from ablation study & vast hyperparameter search
LR = 1e-5 # 0.00005
WEIGHT_DECAY = 1e-3 # 0.001
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_WORKERS = 6 
MAX_N_EPOCHS = 1 # max number of epochs for heldout evaluation with early stopping (default=500)

# same scaffold splitter seed across all folds & experiments
# (folds have different split, thus no need to use different seed across folds)
SCAFFOLD_SPLIT_SEED = 42 # same scaffold splitter seed across all folds & experiments (folds are have different split)

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


def hv_scaffold(model_kwargs=None, train_kwargs=None, use_struct=True,save_probs=False, strct_dataset='dataset1', strct_strategy='conditional', strct_aug_train=False,
                        strct_aug_eval=False, fold=1, exp=1, job_id='date@j1', device='cuda', seed=42):
    logger.info(f'--- Executing CV with scaffold split with hyperparameters: use_struct={use_struct}, save_probs={save_probs}, seed={seed} ...')
    logger.info(f'Dataset path -> {DATA_PATH}')
    train_fp = glob.glob(os.path.join(DATA_PATH, f"train_fold{fold}_*.csv"))[0] # catch files like 'trian_fold2_5_0.9.csv'
    train_df = pd.read_csv(train_fp)

    lr, weight_decay, batch_size = train_kwargs['lr'], train_kwargs['weight_decay'], train_kwargs['batch_size']
    # initialize model w or w/o strucure features
    if use_struct:
        model = choose_model_setting(**model_kwargs).to(device=device)
    else:
        train_df = preprocess_dataset(pd.read_csv(train_fp)) # drop 'ppi_id' column & duplicates
        model = MFusionPPI().to(device=device)

    best_model, best_val_metrics_dict, train_aucs, val_aucs = model.heldout_val_model(f"{job_id}_{fold}_{exp}", use_struct=use_struct, num_epochs=MAX_N_EPOCHS, dataset=train_df,
                                                                                      strct_dataset=strct_dataset, strct_strategy=strct_strategy, strct_aug_train=strct_aug_train, 
                                                                                      strct_aug_eval=strct_aug_eval, optimizer=optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
                                                                                      criterion=nn.BCEWithLogitsLoss(), is_neg_smoo=False, save_probs=save_probs, batch_size=batch_size, device=device, num_workers=NUM_WORKERS, seed=seed)
    
    # plot train vs. validation AUC over epochs
    date, job_id = job_id.split('@') # ['date', 'job_id'] e.g. [''1403', 'j4']
    plots_dir = os.path.join("results", "plots", date, job_id)
    os.makedirs(plots_dir, exist_ok=True)
    plot_train_val_auc(
            train_values=train_aucs,
            val_values=val_aucs,
            save_path=f"{plots_dir}/fold_{fold}.png",
            title="Train vs. Val AUC Over Time",
            xlabel="Training Steps (epochs)",
            ylabel="AUC"
    )
    logger.info(f'--- Saved train vs. val AUC curves to {plots_dir}/fold_{fold}.png successfullys')
    return best_model, best_val_metrics_dict


def cv_cold_eval(use_struct=True, save_probs=False, strct_dataset='dataset1',strct_strategy='conditional', strct_aug_train=False,
                        strct_aug_eval=False, n=10, job_id='date@j1', device='cuda', model_kwargs=None, train_kwargs=None):
    '''
    Cold start setting evaluation for dataset with some negative sampling factor 
    and smoothing factor in order to select the best dataset

    Steps:
    (1) For every fold of the 5 folds
    (2) For every experiment of the n number of experiments (default n=10)
    (3) Execute heldout-validation using scaffold-splitter with early-stopping
    (4) Evaluate the model on the testing set
    (5) Save results in result dictionary that maps each fold i to list with its
        corresponding n experiments results metrcis (AUC, AUPR etc..)
    
    Notes:
    - We save the probability of each prediction in the (1) heldout-validation
    steps, and (2) test phase, in order to better analyse the results with statistical
    techniques (we save the results for every experiment j of fold i in pd.DataFrame())
    - We set seed j+1 for every experiment j in for i for reproducibility.
    '''
    # val & test results dictionary; maps each fold i to its list of 
    # j experiments results,where each results is tuple of metrics (e.g., AUC, AUPR ..)
    test_res_dict = {f'fold{i}': [] for i in range(1, 6)}
    val_res_dict = {f'fold{i}': [] for i in range(1, 6)}

    for i in range(1, 6):
        logger.info (f"---- Start Training & Testing Fold {i} ----")
        for exp_num in range(1, n + 1):
            logger.info(f"experiment {exp_num} ...")
            set_seed(seed=exp_num) 
            # heldout validation using scaffold splitter & early stopping on the validation set
            best_model, val_metric_dict = hv_scaffold(model_kwargs=model_kwargs, train_kwargs=train_kwargs, use_struct=use_struct, save_probs=save_probs,
                                                      strct_dataset=strct_dataset, strct_strategy=strct_strategy, strct_aug_train=strct_aug_train,
                                                      strct_aug_eval=strct_aug_eval, fold=i, exp=exp_num, job_id=job_id, 
                                                      device=device, seed=SCAFFOLD_SPLIT_SEED)
            curr_exp_val_metric_tuple = (val_metric_dict['AUC'], val_metric_dict['AUPR'],
                                         val_metric_dict['Precision'], val_metric_dict['Sensitivity'])
            val_res_dict[f'fold{i}'].append(curr_exp_val_metric_tuple)


            test_fp = glob.glob(os.path.join(DATA_PATH, f"test_fold{i}_*.csv"))[0] # catch files like 'trian_fold2_5_0.9.csv'
            #test_df  = preprocess_dataset(test_fp)
            test_df = pd.read_csv(test_fp)

            # evaluate best model from heldout evaluation step on the cold test set & return
            # metrics (auc, aupr, etc ..); set save=True in order to save predicted probabilities in csv
            batch_size = train_kwargs['batch_size']
            test_metrics_dict, _ = best_model.test_model(fold=f"{job_id}_{i}_{exp_num}", use_struct=use_struct, dataset=test_df,
                                                         strct_dataset=strct_dataset, strct_strategy=strct_strategy, eval_all_confs=strct_aug_eval, criterion=nn.BCEWithLogitsLoss(),
                                                         is_neg_smoo=False, save_probs=save_probs, batch_size=batch_size, device=device, num_workers=NUM_WORKERS)
            curr_exp_metric_tuple = (test_metrics_dict['AUC'], test_metrics_dict['AUPR'], test_metrics_dict['Precision'],
                                     test_metrics_dict['Sensitivity'], test_metrics_dict['Specificity'])
            test_res_dict[f'fold{i}'].append(curr_exp_metric_tuple)

    return test_res_dict, val_res_dict

