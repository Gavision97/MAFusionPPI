import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from utils.tools import *
from MultiPPIMI.MAFusionPPI import MAFusionPPI

if torch.cuda.is_available():
    logger.info(f"GPU is available.")
    device = "cuda"
else:
    logger.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"





# Train & test on cold start data
uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
ds_folder_path = os.path.join('datasets', 'test_dataset', 'train_test_5_0.75')
all_files = os.listdir(ds_folder_path)
dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('_5_0.75.csv', '_df')
    dataframes[df_name] = df

for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)