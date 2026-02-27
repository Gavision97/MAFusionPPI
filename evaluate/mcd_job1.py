import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from utils.tools import *
from MultiPPIMI.MAFusionPPI import MultiPPIMI


logger.info('----------- Monte Carlo Dropout Evaluation - Job n1 -----------')
if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"
def create_dataframes_dict(smiles_list, ppi_list):
    ppi_dict = {}

    for uniprot_id1, uniprot_id2 in ppi_list:
        key = f"{uniprot_id1}_{uniprot_id2}"
        
        # generate DataFrame for this ppi
        df = pd.DataFrame({
            "smiles": smiles_list, 
            "uniprot_id1": [uniprot_id1] * len(smiles_list),
            "uniprot_id2": [uniprot_id2] * len(smiles_list),
            "label": [-1] * len(smiles_list)
        })
        ppi_dict[key] = df
        
    return ppi_dict
    
def evaluate_with_mc_dropout(model, ppi_partition_number, ppi_dict, ppi_partiton_dict, smiles_df, output_dir, mc_iterations=100, batch_size=64, num_workers=0, device='cpu'):
    # Ensure the model is in evaluation mode but with dropout enabled
    model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    flag = False
    ppi_list = ppi_partiton_dict[f'ppi_list_{ppi_partition_number}']
    #ppi_list = ['P01137_P84022', 'P04637_Q00987', 'P06756_P26012']
    for ppi_pair_name in ppi_list:
        probabilities = []
        ppi_pair_name = f'{ppi_pair_name[0]}_{ppi_pair_name[1]}'
        
        if ppi_pair_name != 'P56199_P05556' and flag == False:
          continue
        elif ppi_pair_name == 'P56199_P05556' and flag == False:
          flag = True
          continue
          
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





with open("ppi_dict_.pkl", "rb") as f:
    ppi_partiton_dict = pickle.load(f)
with open("enamine_ppi_dfs_dict.pkl", "rb") as f:
    ppi_dict = pickle.load(f)
logging.info(f'PPI pair set length -> {len(list(ppi_dict.keys()))}')

df = pd.read_csv('datasets/Enamine.csv')
smiles_df = df[['smiles']]
logging.info(f'number of smiles -> {smiles_df.shape[0]}')
output_directory = "./mc_dropout_results_enamine"


auvg_model = AUVG_PPI(dropout=0.3).to(device)

# load saved model and move to CPU (we are going to validate the model, no need for GPU)
auvg_model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))

# execute monte carlo dropout evaluation 
evaluate_with_mc_dropout(
    model=auvg_model, 
    ppi_partition_number=1,
    ppi_dict=ppi_dict,
    ppi_partiton_dict = ppi_partiton_dict,
    smiles_df=smiles_df,
    output_dir=output_directory,
)