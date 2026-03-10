import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from MAFusionPPI.MAFusionPPI import MAFusionPPI
from utils.tools import plot_train_val_auc, set_seed

DATA_PATH = 'datasets/splits_with_ppimi_test_folds_s3'
UNIPROT_MAPPING_PATH = 'datasets/idmapping_unip.tsv'

# best hyperparameters; extracted from ablation study & vast hyperparameter search
LR = 1e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_WORKERS = 6
MAX_N_EPOCHS = 500 # max number of epochs for heldout evaluation with early stopping (default=500)

# same scaffold splitter seed across all folds & experiments
# (folds have different split, thus no need to use different seed across folds)
SCAFFOLD_SPLIT_SEED = 42 # same scaffold splitter seed across all folds & experiments (folds are have different split)

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


def preprocess_dataset(curr_df):
    ''' preprocess curr_df by removing ppi_id column & dropping duplicated rows'''
    return curr_df.drop(columns=['ppi_id']).drop_duplicates()

def hv_scaffold(neg_factor='1', smoo_factor='1', fold=1, exp=1, device='cuda', seed=42):
    logger.info(f'--- Executing CV with scaffold split with hyperparameters of neg_factor={neg_factor} & smoo_factor={smoo_factor} (seed={seed}) ...')

    fold_dir = os.path.join(DATA_PATH, f"fold{fold}", str(neg_factor), str(smoo_factor))
    train_fp = os.path.join(fold_dir, f"train_fold{fold}_{neg_factor}_{smoo_factor}.csv")
    #train_df = preprocess_dataset(pd.read_csv(train_fp))
    train_df = pd.read_csv(train_fp)

    model = MAFusionPPI().to(device=device)
    best_model, best_val_metrics_dict, train_aucs, val_aucs = model.heldout_val_model(f"{fold}_{neg_factor}_{smoo_factor}_{exp}", num_epochs=MAX_N_EPOCHS, dataset=train_df,
                                        optimizer=optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
                                        criterion=nn.BCEWithLogitsLoss(),
                                        batch_size=BATCH_SIZE, device=device, num_workers=NUM_WORKERS, seed=seed)
    

    # plot train vs. validation AUC over epochs
    plot_name = f'neg_fct_{neg_factor}_smoo_fct_{smoo_factor}'
    plots_dir = os.path.join("results", "plots", f"{plot_name}")
    os.makedirs(plots_dir, exist_ok=True)
    plot_train_val_auc(
            train_values=train_aucs,
            val_values=val_aucs,
            save_path=f"{plots_dir}/{plot_name}_fold_{fold}.png",
            title="Train vs. Val AUC Over Time",
            xlabel="Training Steps (epochs)",
            ylabel="AUC"
    )
    logger.info(f'--- Saved train vs. val AUC curves to {plots_dir}/{plot_name}_fold_{fold}.png successfullys')
    return best_model, best_val_metrics_dict


def cold_neg_smoo_eval(neg_factor='1', smoo_factor='1', n=10, device='cuda'):
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
            best_model, val_metric_dict = hv_scaffold(neg_factor=neg_factor, smoo_factor=smoo_factor, fold=i,
                                                      exp=exp_num, device=device, seed=SCAFFOLD_SPLIT_SEED)
            curr_exp_val_metric_tuple = (val_metric_dict['AUC'], val_metric_dict['AUPR'],
                                         val_metric_dict['Precision'], val_metric_dict['Sensitivity'])
            val_res_dict[f'fold{i}'].append(curr_exp_val_metric_tuple)

            fold_name = f"fold{i}_{neg_factor}_{smoo_factor}"
            fold_dir = os.path.join(DATA_PATH, f'fold{i}', str(neg_factor), str(smoo_factor))
            #test_df  = preprocess_dataset(pd.read_csv(os.path.join(fold_dir, f"test_{fold_name}.csv")))
            test_df = pd.read_csv(os.path.join(fold_dir, f"test_{fold_name}.csv"))

            # evaluate best model from heldout evaluation step on the cold test set & return
            # metrics (auc, aupr, etc ..); set save=True in order to save predicted probabilities in csv
            test_metrics_dict, _ = best_model.test_model(fold=f"{i}_{neg_factor}_{smoo_factor}_{exp_num}", dataset=test_df, criterion=nn.BCEWithLogitsLoss(), batch_size=BATCH_SIZE,
                                         device=device, num_workers=NUM_WORKERS, save=True)
            curr_exp_metric_tuple = (test_metrics_dict['AUC'], test_metrics_dict['AUPR'], test_metrics_dict['Precision'],
                                     test_metrics_dict['Sensitivity'], test_metrics_dict['Specificity'])
            test_res_dict[f'fold{i}'].append(curr_exp_metric_tuple)

    return test_res_dict, val_res_dict

