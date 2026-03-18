import argparse
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

from utils.parser import get_args
from evaluate.train_test_cold_start_ppi import train_test_cold_start_ppi
from evaluate.cold_neg_smoo_eval import cold_neg_smoo_eval
from evaluate.cv_cold_eval import cv_cold_eval
from evaluate.mcd_eval import mcd_eval

PPI_DICT_PATH = 'saved_obj/ppi_dict.pkl'
PPI_DFS_DICT_PATH = "saved_obj/enamine_ppi_dfs_dict.pkl"

def cv_cold_eval_(
    res_file_name = "cv_cold_results",
    save_probs=False,
    use_struct= True,
    strct_dataset='dataset1',
    strct_strategy="conditional",
    strct_aug_train=False,
    strct_aug_eval=False,
    job_id= "date@j1",
    n = 10,
    device = 'cuda'
) -> None:
    """
    Runs cv_cold_neg_smoo and appends one row to results/<res_file_name>.csv

    Output row schema (7 columns):
      job_id, val_res, fold1, fold2, fold3, fold4, fold5

    - val_res: JSON string representing validation results for all folds
               e.g. {"fold1":[...], "fold2":[...], ...} or {"fold1":(auc,aupr,...), ...}
               depending on what hv_scaffold_to_get_n_epochs returns.
    - foldK: JSON string representing list of n tuples (auc, aupr, precision, sensitivity, specificity)
    """

    exp_name = job_id
    logger.info(f'--- Start Cold Start Experiments for Dataset -> {exp_name} ---')
    logger.info(f'Job hyperparameters -> use_struct={use_struct} strct dataset={strct_dataset}, strct strategy={strct_strategy}, strct train aug={strct_aug_train}, strct eval aug={strct_aug_eval}')

    res_dict, val_metrics_dict = cv_cold_eval(use_struct=use_struct, save_probs=save_probs, strct_dataset=strct_dataset,
                                                    strct_strategy=strct_strategy, strct_aug_train=strct_aug_train,
                                                    strct_aug_eval=strct_aug_eval,  n=n, job_id=job_id, device=device)


    fold_keys = [f"fold{i}" for i in range(1, 6)]    
    row = {"exp": exp_name, "val_res": json.dumps(val_metrics_dict or {}, default=list)}
    for fk in fold_keys:
        row[fk] = json.dumps(res_dict.get(fk, []))

    summary_df = pd.DataFrame([row], columns=["exp", "val_res"] + fold_keys)

    # save/append
    res_path = os.path.join("results", "result_tables", "final_results")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)
    logger.info(f"Saved/updated summary to: {summary_path}")


def cold_neg_smoo_eval_(
    res_file_name = "cv_cold_neg_smoo_results",
    save_probs=False,
    use_struct=True,
    neg_factor = "1",
    smoo_factor= "1",
    strct_dataset='dataset1',
    strct_strategy="conditional",
    strct_aug_train=False,
    strct_aug_eval=False,
    job_id= "date@j1",
    n = 10,
    device = 'cuda'
) -> None:
    """
    Runs cv_cold_neg_smoo and appends one row to results/<res_file_name>.csv

    Output row schema (7 columns):
      job_id, val_res, fold1, fold2, fold3, fold4, fold5

    - val_res: JSON string representing validation results for all folds
               e.g. {"fold1":[...], "fold2":[...], ...} or {"fold1":(auc,aupr,...), ...}
               depending on what hv_scaffold_to_get_n_epochs returns.
    - foldK: JSON string representing list of n tuples (auc, aupr, precision, sensitivity, specificity)
    """

    exp_name = f'dataset_neg_fct_{neg_factor}_smoo_fct_{smoo_factor}_{job_id}'
    logger.info(f'--- Start Cold Start Experiments for Dataset -> {exp_name} ---')
    logger.info(f'Job hyperparameters -> strct dataset={strct_dataset}, strct strategy={strct_strategy}, strct train aug={strct_aug_train}, strct eval aug={strct_aug_eval}')

    res_dict, val_metrics_dict = cold_neg_smoo_eval(use_struct=use_struct, save_probs=save_probs, neg_factor=neg_factor, smoo_factor=smoo_factor, strct_dataset=strct_dataset,
                                                    strct_strategy=strct_strategy, strct_aug_train=strct_aug_train,
                                                    strct_aug_eval=strct_aug_eval,  n=n, job_id=job_id, device=device)


    fold_keys = [f"fold{i}" for i in range(1, 6)]    
    # we store *all* val results in one cell (as JSON)
    row = {"exp": exp_name, "val_res": json.dumps(val_metrics_dict or {}, default=list)}

    # we store each fold’s list-of-tuples in its own cell (as JSON)
    for fk in fold_keys:
        # fk in val_metrics_dict is probably 'fold1'.. 'fold5'
        # fk in res_dict is also 'fold1'.. 'fold5'
        row[fk] = json.dumps(res_dict.get(fk, []))

    summary_df = pd.DataFrame([row], columns=["exp", "val_res"] + fold_keys)

    # save/append
    res_path = os.path.join("results", "result_tables", "final_results")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)
    logger.info(f"Saved/updated summary to: {summary_path}")


def mcd_eval_(exp_name='Enamine', smiles_column='smiles'):

    # load PPI partition dictionary & PPI dataframe dictionary 
    with open(PPI_DICT_PATH, "rb") as f:
        ppi_partiton_dict = pickle.load(f)
    with open(PPI_DFS_DICT_PATH, "rb") as f:
        ppi_dict = pickle.load(f)

    logger.info(f'PPI pair set length -> {len(list(ppi_dict.keys()))}')

    df = pd.read_csv(f'datasets/mcd/{exp_name}.csv')
    smiles_df = df[[smiles_column]]
    logging.info(f'number of smiles -> {smiles_df.shape[0]}')
    output_directory = "./mc_dropout_results_enamine"

    # execute monte carlo dropout evaluation 
    mcd_eval(
        ppi_partition_number=1,
        ppi_dict=ppi_dict,
        ppi_partiton_dict = ppi_partiton_dict,
        smiles_df=smiles_df,
        output_dir=output_directory,
    )



def main():

    # get all arguments
    args = get_args()

    log_msg = args.log_msg
    res_file_name = args.res_file_name
    log_file_name = args.log_file_name
    eval_method = args.eval_method
    exp_log_dir = args.exp_log_dir
    job_id = args.job_id
    n_exp = args.n_exp
    
    use_struct = True if args.use_struct == "True" else False
    save_probs = True if args.save_probs == "True" else False # whether to save model probs for val & test

    # cold evaluation @ negative sampling & smoothing factor unique hyperparameters
    neg_factor = args.neg_factor # ["1", "2", "3", "4", "5"]
    smoo_factor = args.smoo_factor # ["0.75", "0.8", "0.9", "0.95", "1.0"]
    strct_dataset = args.strct_dataset # ["dataset1", "dataset2", "dataset3"]
    strct_strategy = args.strct_strategy # ["conditional", "full_mean", "subset_mean"]
    strct_aug_train = True if args.strct_aug_train == "True" else False
    strct_aug_eval = False if args.strct_aug_eval == "False" else True

    epo_f1 = args.epo_f1
    epo_f2 = args.epo_f2
    epo_f3 = args.epo_f3
    epo_f4 = args.epo_f4
    epo_f5 = args.epo_f5
    n_epochs = args.n_epochs
    folds = args.folds

    device = args.device


    logger.info(log_msg)
    # constract log file in order to log the results
    log_dir = os.path.join("results", "logs", exp_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_file_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', force=True)
    
    # choose experiment & execute
    if eval_method == 'cold':
        nel = [epo_f1, epo_f2, epo_f3, epo_f4, epo_f5]
        train_test_cold_start_ppi(nel=nel, n=n_exp)
    elif eval_method =='cv_neg_smoo':
        cold_neg_smoo_eval_(res_file_name=res_file_name, save_probs=save_probs, use_struct=use_struct, neg_factor=neg_factor, smoo_factor=smoo_factor,
                            strct_dataset=strct_dataset, strct_strategy=strct_strategy, strct_aug_train=strct_aug_train,
                             strct_aug_eval=strct_aug_eval, job_id=job_id , n=n_exp, device=device)
    elif eval_method == 'cv_cold':
        cv_cold_eval_(res_file_name=res_file_name, save_probs=save_probs, use_struct=use_struct, strct_dataset=strct_dataset, strct_strategy=strct_strategy,
                      strct_aug_train=strct_aug_train, strct_aug_eval=strct_aug_eval, job_id=job_id, n=n_exp, device=device)
    elif eval_method == 'mcd':
        pass # TODO: finish building mcd pipeline & test it


if __name__ == "__main__":
    main()


