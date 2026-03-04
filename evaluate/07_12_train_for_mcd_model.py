import os
import copy
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from MAFusionPPI.MAFusionPPI import MAFusionPPI
from utils.MoleculeDataset import MoleculeDataset
from utils.tools import *

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"

# best hyperparameters; extracted from ablation study & vast hyperparameter search
LR = 1e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_WORKERS = 16

# Train & test on cold start data
uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
f_df = pd.read_csv("datasets/final_dataset_5_0.75_25_09_2024_without_long_uncategorized_PPIs.csv")

logging.info('--- Load final dataset successfully ! ---')

# Train val model in order to get number of epochs 
logging.info('--- Start train val phase in order to get the right number of epochs for training ! ---')

model = MAFusionPPI(dropout=DROPOUT)
n_epochs, _ = model.train_val_model('final_ds_without_long_uncategorized_PPIs', num_epochs=100, dataset=f_df,
                  optimizer=optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
                  criterion=nn.BCEWithLogitsLoss(),
                  batch_size=BATCH_SIZE, device=device, num_workers=NUM_WORKERS)
logging.info('--- Done train val phase ! ---')
logging.info('--------------------------------------------------')

# train new model on 100% of the data n_epochs epochs and save the model.
logging.info('--- Start training new model ! ---')
model = MAFusionPPI(dropout=DROPOUT)

model.train_model('final_ds_without_long_uncategorized_PPIs', num_epochs=n_epochs, dataset=f_df,
                  optimizer=optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
                  criterion=nn.BCEWithLogitsLoss(),
                  batch_size=BATCH_SIZE, device=device, num_workers=NUM_WORKERS)

model_state_dict = copy.deepcopy(model.state_dict())
torch.save(model_state_dict, 'saved_model.pth')  
logging.info('--- Model saved successfully to "saved_model.pth" !!! ---')
