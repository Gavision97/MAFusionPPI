import argparse
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

from evaluate.train_test_cold_start_ppi import train_test_cold_start_ppi
from evaluate.cold_neg_smoo_eval import cold_neg_smoo_eval


def cold_neg_smoo_eval_(
    res_file_name: str = "cv_cold_neg_smoo_results",
    neg_factor: str = "1",
    smoo_factor: str = "1",
    n: int = 10,
    device: str = 'cuda'
):
    """
    Runs cv_cold_neg_smoo and appends one row to results/<res_file_name>.csv

    Output row schema (7 columns):
      exp, val_res, fold1, fold2, fold3, fold4, fold5

    - val_res: JSON string representing validation results for all folds
               e.g. {"fold1":[...], "fold2":[...], ...} or {"fold1":(auc,aupr,...), ...}
               depending on what hv_scaffold_to_get_n_epochs returns.
    - foldK: JSON string representing list of n tuples (auc, aupr, precision, sensitivity, specificity)
    """

    exp_name = f'dataset_neg_fct_{neg_factor}_smoo_fct_{smoo_factor}'
    logger.info(f'--- Start Cold Start Experiments for Dataset -> {exp_name} ---')
    res_dict, val_metrics_dict = cold_neg_smoo_eval(neg_factor=neg_factor, smoo_factor=smoo_factor, n=n, device=device)


    fold_keys = [f"fold{i}" for i in range(1, 6)]    
    # we store *all* val results in one cell (as JSON)
    row = {"exp": exp_name, "val_res": json.dumps(val_metrics_dict, default=list)}

    # we store each fold’s list-of-tuples in its own cell (as JSON)
    for fk in fold_keys:
        # fk in val_metrics_dict is probably 'fold1'.. 'fold5'
        # fk in res_dict is also 'fold1'.. 'fold5'
        row[fk] = json.dumps(res_dict[fk])

    summary_df = pd.DataFrame([row], columns=["exp", "val_res"] + fold_keys)

    # save/append
    res_path = os.path.join("results", "result_tables")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)
    logger.info(f"Saved/updated summary to: {summary_path}")

    return res_dict, val_metrics_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_msg", type=str, default='log message', help="First message to display in the log file; for experiment description")
    parser.add_argument("--res_file_name", type=str, default='res', help="name of the result; pandas.DataFrame() object")
    parser.add_argument("--eval_method", type=str, default='cv_neg_smoo', choices=['cv','cv_neg_smoo' 'cold'], help="path to the smiles file")
    parser.add_argument("--cv_method", type=str, default='all', choices=['all', 'per_fam'], help="The split for 10-fold cross-validation - all data or per RNA subtype ")
    parser.add_argument('--cv_neg_factor', type=str, default='1', choices=['1.0', '2', '3', '4', '5'], help="negative sampling hyperparameter (e.g., 1, 2 ... 5)")
    parser.add_argument('--cv_smoo_factor', type=str, default='1.0', choices=['0.75', '0.8', '0.9', '0.95', '1.0'], help="")
    parser.add_argument("--plot_name", type=str, default='loss_curve', help="loss curve plot name; will be saved to results/plots/plot_name.png")
    parser.add_argument("--log_file_name", type=str, default="res", help="name of the logger file (e.g., result.log)")
    parser.add_argument("--k_fold", type=int,  default=5, help="Number of folds for stratified K-fold cross-validation (default: 10)")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="device to use - cuda/cpu")
    args = parser.parse_args()


    log_msg = args.log_msg
    res_file_name = args.res_file_name
    log_file_name = args.log_file_name
    eval_method= args.eval_method
    cv_neg_factor = args.cv_neg_factor
    cv_smoo_factor = args.cv_smoo_factor
    plot_name = args.plot_name
    k_fold = args.k_fold
    device = args.device


    # constract log file in order to log the results
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_file_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', force=True)
    
    
    logger.info(log_msg)
    if eval_method == 'cold':
        train_test_cold_start_ppi()
    elif eval_method =='cv_neg_smoo':
        cold_neg_smoo_eval_(neg_factor=cv_neg_factor, smoo_factor=cv_smoo_factor, res_file_name=res_file_name, device=device)


if __name__ == "__main__":
    main()


