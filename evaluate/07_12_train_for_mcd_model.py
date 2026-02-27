import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from MultiPPIMI.MAFusionPPI import MAFusionPPI
from utils.MoleculeDataset import MoleculeDataset
from utils.tools import *

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


# Train & test on cold start data
uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
f_df = pd.read_csv("datasets/final_dataset_5_0.75_25_09_2024_without_long_uncategorized_PPIs.csv")

logging.info('--- Load final dataset successfully ! ---')

# Train val model in order to get number of epochs 
logging.info('--- Start train val phase in order to get the right number of epochs for training ! ---')

model = MultiPPIMI(batch_size=64, dropout=0.3)
n_epochs, _ = model.train_val_model('final_ds_without_long_uncategorized_PPIs', num_epochs=100, dataset=f_df,
                  optimizer=optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3),
                  criterion=nn.BCEWithLogitsLoss(),
                  batch_size=64, device=device, num_workers=16)
logging.info('--- Done train val phase ! ---')
logging.info('--------------------------------------------------')

# train new model on 100% of the data n_epochs epochs and save the model.
logging.info('--- Start training new model ! ---')
model = MultiPPIMI(batch_size=64, dropout=0.3)

model.train_model('final_ds_without_long_uncategorized_PPIs', num_epochs=n_epochs, dataset=f_df,
                  optimizer=optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3),
                  criterion=nn.BCEWithLogitsLoss(),
                  batch_size=64, device=device, num_workers=16)

model_state_dict = copy.deepcopy(model.state_dict())
torch.save(model_state_dict, 'saved_model.pth')  
logging.info('--- Model saved successfully to "saved_model.pth" !!! ---')
