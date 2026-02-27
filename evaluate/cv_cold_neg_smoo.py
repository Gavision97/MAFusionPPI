
import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

DATA_PATH = 'datasets/splits_with_ppimi_test_folds_s3'
UNIPROT_MAPPING_PATH = 'datasets/idmapping_unip.tsv'

def preprocess_dataset(curr_df):
    return curr_df.drop(columns=['ppi_id'], inplace=True).drop_duplicates(inplace=True)

def cv_cold_neg_smoo(neg_factor='1', smoo_factor='1'):

    def _preprocess_dataset(curr_df):
        return curr_df.drop(columns=['ppi_id'], inplace=True).drop_duplicates(inplace=True)
    datasets = {}
    '''
    We prepare dictionary that maps each fold_i {i=1,2 ... 5} to its
    corresponding train, val, & test dataset (i.e., pd.DataFrame() objects)
    with the columns -> ['smiles', 'uniprot1', 'uniprots2', 'label']
    '''
    for i in range(1, 6):
        fold_name = f"fold{i}"
        fold_dir = os.path.join(DATA_PATH, fold_name, str(neg_factor), str(smoo_factor))

        train_fp = os.path.join(fold_dir, f"train_{fold_name}_{neg_factor}_{smoo_factor}.csv")
        val_fp   = os.path.join(fold_dir, f"valid_{fold_name}_{neg_factor}_{smoo_factor}.csv")
        test_fp  = os.path.join(fold_dir, f"test_{fold_name}_{neg_factor}_{smoo_factor}.csv")

        '''
        load train, val, & test dataset and then we drop
        ppi_id columns & drop duplicated rows
        '''
        train_df = _preprocess_dataset(pd.read_csv(train_fp))
        val_df   = _preprocess_dataset(pd.read_csv(val_fp))
        test_df  = _preprocess_dataset(pd.read_csv(test_fp))

        # key1=train, key2=val, & key3=test
        datasets[fold_name] = [train_df, val_df, test_df]

        '''
        TODO: we first call train_val() in order to decide number of epochs, the train() then test()
        TIME 15 for statistical significants ...
        '''