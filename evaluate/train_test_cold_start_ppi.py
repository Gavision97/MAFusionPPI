import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from utils.tools import *
from MAFusionPPI.MAFusionPPI import MAFusionPPI

COLD_PATH = 'datasets/train_test_5_0.75'
UNIPROT_MAPPING_PATH = 'datasets/idmapping_unip.tsv'

def avg_expirements_auc(dataframes, num_epochs_list, n):
    logger.info('--- Start training! ---')

    res_dict = {f'exp{i+1}': [] for i in range(n)}

    for exp_num in range(1, n + 1):
        logger.info(f"Starting Experiment {exp_num}")

        for fold_num in range(1, 6):
            fold_name = f'fold{fold_num}'
            train_fold = dataframes[f'train_{fold_name}_df']
            test_fold  = dataframes[f'test_{fold_name}_df']
            num_epochs = num_epochs_list[fold_num - 1]

            model = MAFusionPPI(dropout=0.3).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            model.train_model(
                fold_name,
                num_epochs=num_epochs,
                dataset=train_fold,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=64,
                device=device,
                num_workers=16
            )

            res_tuple = model.test_model(
                test_fold,
                criterion=criterion,
                batch_size=64,
                device=device,
                num_workers=16
            )

            res_dict[f'exp{exp_num}'].append(res_tuple)

    return res_dict


def train_test_cold_start_ppi():
    # Train & test on cold start data
    uniprot_mapping = pd.read_csv(UNIPROT_MAPPING_PATH, delimiter="\t")

    dataframes = {}
    for file in os.listdir(COLD_PATH):
        file_path = os.path.join(COLD_PATH, file)
        df_name = file.replace('_5_0.75.csv', '_df')
        dataframes[df_name] = pd.read_csv(file_path)

    for df_name in dataframes:
        dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)

    nel = [23, 32, 23, 54, 30]
    n = 10
    ten_exp_res_dict = avg_expirements_auc(dataframes=dataframes, num_epochs_list=nel, n=n)
    
    logger.info('Done evaluating using cold start stetting, results:')
    for exp, res_list in ten_exp_res_dict.items():
        logger.info(f"{exp}: {res_list}")


def main():
    pass


if __name__ == "__main__":
    main()