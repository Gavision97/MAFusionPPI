import argparse
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import warnings
warnings.filterwarnings("ignore")


import logging
logger = logging.getLogger(__name__)

from evaluate.train_test_cold_start_ppi import train_test_cold_start_ppi
from evaluate.cv_cold_neg_smoo import cv_cold_neg_smoo


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_msg", type=str, default='log message', help="First message to display in the log file; for experiment description")
    parser.add_argument("--res_file_name", type=str, default='res', help="name of the result; pandas.DataFrame() object")
    parser.add_argument("--eval_method", type=str, default='cold', choices=['cv','cv_neg_smoo' 'cold'], help="path to the smiles file")
    parser.add_argument("--cv_method", type=str, default='all', choices=['all', 'per_fam'], help="The split for 10-fold cross-validation - all data or per RNA subtype ")
    parser.add_argument('--cv_neg_factor', type=str, default='1', choices=['1', '2', '3', '4', '5'], help="negative sampling hyperparameter (e.g., 1, 2 ... 5)")
    parser.add_argument('--cv_smoo_factor', type=str, default='1', choices=['0.75', '0.8', '0.9', '0.95', '1'], help="")
    parser.add_argument("--plot_name", type=str, default='loss_curve', help="loss curve plot name; will be saved to results/plots/plot_name.png")
    parser.add_argument("--log_file_name", type=str, default="res", help="name of the logger file (e.g., result.log)")
    parser.add_argument("--k_fold", type=int,  default=5, help="Number of folds for stratified K-fold cross-validation (default: 10)")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="device to use - cuda/cpu")
    args = parser.parse_args()


    log_msg = args.log_msg
    res_file_name = args.res_file_name
    log_file_name = args.log_file_name
    eval_method= args.eval_method
    cv_neg_factor = parser.cv_neg_factor
    cv_smoo_factor = parser.cv_smoo_factor
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
        cv_cold_neg_smoo(neg_factor=cv_neg_factor, smoo_factor=cv_smoo_factor)


if __name__ == "__main__":
    main()


