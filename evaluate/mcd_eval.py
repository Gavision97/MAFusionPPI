import os
import copy
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from utils.tools import *
from utils.MoleculeDataset import MCDMoleculeDataset
from MAFusionPPI.MAFusionPPI import MAFusionPPI


logger.info('----------- Monte Carlo Dropout Evaluation - Job n1 -----------')
if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"



SAVED_MODEL_WEIGHTS = 'saved_obj/saved_model.pth'

DROPOUT = 0.3
   #auvg_model = AUVG_PPI(dropout=0.3).to(device)

    # load saved model and move to CPU (we are going to validate the model, no need for GPU)
    #auvg_model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))

    
def mcd_eval(ppi_partition_number, ppi_dict, ppi_partiton_dict, smiles_df, output_dir, mc_iterations=100, batch_size=64, num_workers=0, device='cpu'):
    # Ensure the model is in evaluation mode but with dropout enabled
    model = MAFusionPPI(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(SAVED_MODEL_WEIGHTS, map_location=torch.device('cpu')))
  
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

 
    ppi_list = ppi_partiton_dict[f'ppi_list_{ppi_partition_number}']
   
    for ppi_pair_name in ppi_list:
        probabilities = []
        ppi_pair_name = f'{ppi_pair_name[0]}_{ppi_pair_name[1]}'
                
        logging.info(f"Current PPI pair -> {ppi_pair_name}")
        curr_df = ppi_dict[ppi_pair_name]
        curr_prob_df = copy.deepcopy(smiles_df)
        test_dataset = MCDMoleculeDataset(curr_df)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        for i in range(mc_iterations):
            mc_probs = []
            # Collect MC Dropout predictions
            for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                     batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels) in test_loader:
                with torch.no_grad():
                    batch_chemprop_features, batch_chemberta_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_morgan, batch_chem_desc, batch_labels = [batch.to(device) for batch in [batch_chemprop_features, batch_chemberta_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_morgan, batch_chem_desc, batch_labels]]

                    outputs = model(batch_chemprop_features, batch_esm_features, batch_fegs_features,
                                   batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)
                    prob = torch.sigmoid(outputs).cpu().numpy()
                    mc_probs.extend(prob.flatten())  
                    
            curr_prob_df[f"mcd_prob_{i+1}"] = mc_probs

        output_path = os.path.join(output_dir, f"{ppi_pair_name}.csv")
        curr_prob_df.to_csv(output_path, index=False)
        logging.info(f"Saved results for {ppi_pair_name} to {output_path}")
