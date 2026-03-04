import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from MAFusionPPI.MAFusionPPI import MAFusionPPI
from utils.tools import plot_train_val_auc

DATA_PATH = 'datasets/splits_with_ppimi_test_folds_s3'
UNIPROT_MAPPING_PATH = 'datasets/idmapping_unip.tsv'

# best hyperparameters; extracted from ablation study & vast hyperparameter search
LR = 1e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_WORKERS = 16
MAX_N_EPOCHS = 500 # max number of epochs for heldout evaluation with early stopping (default=500)

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


def preprocess_dataset(curr_df):
    return curr_df.drop(columns=['ppi_id']).drop_duplicates()

def hv_scaffold(neg_factor='1', smoo_factor='1', device='cuda', seed=42):
    logger.info(f'--- Executing CV with scaffold split with hyperparameters of neg_factor={neg_factor} & smoo_factor={smoo_factor} ...')
    n_epochs = [] # list of number of epochs for each fold
    val_metrics_dict = {}
    for i in range(1, 6):
        logger.info(f'fold number={i}')

        fold_dir = os.path.join(DATA_PATH, f"fold{i}", str(neg_factor), str(smoo_factor))
        train_fp = os.path.join(fold_dir, f"train_fold{i}_{neg_factor}_{smoo_factor}.csv")
        train_df = preprocess_dataset(pd.read_csv(train_fp))

        model = MAFusionPPI(dropout=DROPOUT).to(device=device)
        epo, val_matrics, train_aucs, val_aucs = model.train_val_model(f"train_fold{i}_{neg_factor}_{smoo_factor}", num_epochs=MAX_N_EPOCHS, dataset=train_df,
                                            optimizer=optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
                                            criterion=nn.BCEWithLogitsLoss(),
                                            batch_size=BATCH_SIZE, device=device, num_workers=NUM_WORKERS, seed=seed)
        n_epochs.append(epo) # add number of epochs for fold i
        val_metrics_dict[f'fold{i}'] = val_matrics

        # plot train vs. validation AUC over epochs
        plot_name = f'neg_fct_{neg_factor}_smoo_fct_{smoo_factor}'
        plots_dir = os.path.join("results", "plots", f"{plot_name}")
        os.makedirs(plots_dir, exist_ok=True)
        plot_train_val_auc(
                train_values=train_aucs,
                val_values=val_aucs,
                save_path=f"{plots_dir}/{plot_name}_fold_{i}.png",
                title="Train vs. Val AUC Over Time",
                xlabel="Training Steps (epochs)",
                ylabel="AUC"
        )
        logger.info(f'--- Saved train vs. val AUC curves to {plots_dir}/{plot_name}_fold_{i}.png successfullys')
    return model, val_metrics_dict

# for i in range(5) -> for j in range(exp_num) -> seed(j+1)

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
    # results dictionary; maps each fold i to its list of j experiments results,
    # where each results is tuple of metrics (e.g., AUC, AUPR ..)
    res_dict = {f'fold{i}': [] for i in range(1, 6)}

    logger.info(f'###### Number of epochs to train per fold after heldout validation -> {n_epochs} ######')
    for i in range(5):
        logger.info (f"---- Start Training & Testing Fold {i+1} ----")
        for exp_num in range(1, n + 1):
            logger.info(f"experiment {exp_num} ...")
            # heldout validation using scaffold splitter & early stopping on the validation set
            n_epochs, val_metrics_dict = hv_scaffold(neg_factor=neg_factor, smoo_factor=smoo_factor, device=device)



            fold_name = f"fold{i+1}_{neg_factor}_{smoo_factor}"
            fold_dir = os.path.join(DATA_PATH, f'fold{i+1}', str(neg_factor), str(smoo_factor))

            # load train, val, & test dataset and then we drop ppi_id columns & drop duplicated rows
            train_df = preprocess_dataset(pd.read_csv(os.path.join(fold_dir, f"train_{fold_name}.csv")))
            test_df  = preprocess_dataset(pd.read_csv(os.path.join(fold_dir, f"test_{fold_name}.csv")))

            model = MAFusionPPI(dropout=DROPOUT).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()

            # train the model with N epochs from the heldout validation step
            model.train_model(fold=fold_name, num_epochs=n_epochs[i], dataset=train_df,
                              optimizer=optimizer, criterion=criterion, batch_size=BATCH_SIZE,
                              device=device, num_workers=NUM_WORKERS)

            # test the model on the cold test set & return metrics (auc, aupr, etc ..)
            test_matric_dict, _ = model.test_model(test_dataset=test_df, criterion=criterion, batch_size=BATCH_SIZE,
                                         device=device, num_workers=NUM_WORKERS)
            curr_exp_metric_tuple = (test_matric_dict['AUC'], test_matric_dict['AUPR'], test_matric_dict['Precision'],
                                     test_matric_dict['Sensitivity'], test_matric_dict['Specificity'])
            res_dict[f'fold{i+1}'].append(curr_exp_metric_tuple)

    return res_dict, val_metrics_dict

