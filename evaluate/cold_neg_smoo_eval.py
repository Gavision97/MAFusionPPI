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

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


def preprocess_dataset(curr_df):
    return curr_df.drop(columns=['ppi_id']).drop_duplicates()

def hv_scaffold_to_get_n_epochs(neg_factor='1', smoo_factor='1', device='cuda'):
    logger.info(f'--- Executing CV with scaffold split with hyperparameters of neg_factor={neg_factor} & smoo_factor={smoo_factor} ...')
    n_epochs = [] # list of number of epochs for each fold
    val_metrics_dict = {}
    for i in range(1, 6):
        logger.info(f'fold number={i}')

        fold_dir = os.path.join(DATA_PATH, f"fold{i}", str(neg_factor), str(smoo_factor))
        train_fp = os.path.join(fold_dir, f"train_fold{i}_{neg_factor}_{smoo_factor}.csv")
        train_df = preprocess_dataset(pd.read_csv(train_fp))

        model = MAFusionPPI(dropout=0.3).to(device=device)
        epo, val_matrics, train_aucs, val_aucs = model.train_val_model(f"train_fold{i}_{neg_factor}_{smoo_factor}", num_epochs=100, dataset=train_df,
                                            optimizer=optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3),
                                            criterion=nn.BCEWithLogitsLoss(),
                                            batch_size=64, device=device, num_workers=16)
        n_epochs.append(epo) # add number of epochs for fold i
        val_metrics_dict[f'fold{i}'] = val_matrics

        # plot train vs. validation AUC over epochs
        plot_name = f'neg_fct_{neg_factor}_smoo_fct_{smoo_factor}'
        plots_dir = os.path.join("results", "plots", f"{plot_name}")
        os.makedirs(plots_dir, exist_ok=True)
        plot_train_val_auc(
                train_values=train_aucs,
                val_values=val_aucs,
                save_path=f"{plots_dir}/{plot_name}_fold_{i+1}.png",
                title="Loss Curve (train vs. val MSE)"
        )

    return n_epochs, val_metrics_dict


def cold_neg_smoo_eval(neg_factor='1', smoo_factor='1', n=15, device='cuda'):

    res_dict = {f'fold{i}': [] for i in range(1, 6)}

    # CV using scaffold splitter & early stopping on the validation set, 
    # in order to get number of epochs to train the model & validation metrics
    n_epochs, val_metrics_dict = hv_scaffold_to_get_n_epochs(neg_factor=neg_factor, smoo_factor=smoo_factor, device=device)

    for i in range(5):
        logger.info(f"---- Start Training & Testing Fold {i+1} ----")
        for exp_num in range(1, n + 1):
            logger.info(f"---- Starting Experiment {exp_num} ----")
            fold_name = f"fold{i+1}_{neg_factor}_{smoo_factor}"
            fold_dir = os.path.join(DATA_PATH, fold_name, str(neg_factor), str(smoo_factor))

            # load train, val, & test dataset and then we drop ppi_id columns & drop duplicated rows
            train_df = preprocess_dataset(pd.read_csv(os.path.join(fold_dir, f"train_{fold_name}.csv")))
            test_df  = preprocess_dataset(pd.read_csv(os.path.join(fold_dir, f"test_{fold_name}.csv")))

            model = MAFusionPPI(dropout=0.3).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            # train the model with N epochs from the heldout validation step
            model.train_model(fold=fold_name, num_epochs=n_epochs[i], dataset=train_df,
                              optimizer=optimizer, criterion=criterion, batch_size=64,
                              device=device, num_workers=16)

            # test the model on the cold test set & return metrics (auc, aupr, etc ..)
            test_matric_dict, _ = model.test_model(criterion=criterion, batch_size=64,
                                         device=device, num_workers=16)
            curr_exp_metric_tuple = (test_matric_dict['AUC'], test_matric_dict['AUPR'], test_matric_dict['Precision'],
                                     test_matric_dict['Sensitivity'], test_matric_dict['Specificity'])
            res_dict[f'fold{i+1}'].append(curr_exp_metric_tuple)

    return res_dict, val_metrics_dict