
import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import random
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"

RES_TABLES_PATH = 'results/result_tables/'

def seed_worker(worker_id):
    '''for dataloader workers reproducibility'''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def preprocess_dataset(curr_df):
    ''' preprocess curr_df by removing ppi_id column & dropping duplicated rows'''
    if 'ppi_id' in list(curr_df.columns):
        return curr_df.drop(columns=['ppi_id']).drop_duplicates()
    else:
        return curr_df.drop_duplicates()


def get_out_dir(fold_name, is_neg_smoo = False):
        fold_splitted_name = fold_name.split("_")
        if is_neg_smoo:
            # fold format -> f'{job_id}_{fold_id}_{neg_factor}_{smoo_factor}_{exp_num}'
            job_id, fold_id, neg_factor, smoo_factor, exp_num = (fold_splitted_name[0], fold_splitted_name[1], 
                                                            fold_splitted_name[2], fold_splitted_name[3], fold_splitted_name[4])
            
            # save evaluation results in results/date/job_id/results_tables/cold_neg_{i}_smoo_{j}/{fold_k}/exp_{x}/test_exp_{x}.csv
            date, job_id = job_id.split("@") # ['date', 'job_id']
            out_dir = os.path.join(RES_TABLES_PATH, date, job_id, f"cold_neg_{neg_factor}_smoo_{smoo_factor}", fold_id, f"exp_{exp_num}")
        else:
            # save evaluation results in results/date/job_id/results_tables/{fold_k}/exp_{x}/test_exp_{x}.csv
            job_id, fold_id, exp_num = (fold_splitted_name[0], fold_splitted_name[1], fold_splitted_name[2])
            date, job_id = job_id.split("@") # ['date', 'job_id']
            out_dir = os.path.join(RES_TABLES_PATH, date, job_id, fold_id, f"exp_{exp_num}")
        
        return out_dir, exp_num



def convert_uniprot_ids(dataset, mapping_df):
    # Create a dictionary from the mapping dataframe
    mapping_dict = mapping_df.set_index('From')['Entry'].to_dict()

    # Map the uniprot_id1 and uniprot_id2 columns to their respective Entry values
    dataset['uniprot_id1'] = dataset['uniprot_id1'].map(mapping_dict)
    dataset['uniprot_id2'] = dataset['uniprot_id2'].map(mapping_dict)
    return dataset.drop_duplicates()

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


def plot_train_val_auc(
    train_values,
    val_values,
    matric='AUC',
    save_path=None,
    title="Loss Curve",
    xlabel="Training Steps",
    ylabel="Loss"
):
    steps = range(len(train_values))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_values, label=f"Training {matric}", linewidth=2, color="blue")
    plt.plot(steps, val_values, label=f"Validation {matric}", linewidth=2, color="green")

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


class custom_self_attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(custom_self_attention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward(self, embeddings_mat):
        # Apply self-attention for PPI
        embeddings_mat = embeddings_mat.permute(1, 0, 2)  # Change to (num_modalities, batch_size, embed_dim) for MultiheadAttention
        attn_output, attn_weights = self.self_attention(embeddings_mat, embeddings_mat, embeddings_mat)
        attn_output = attn_output.permute(1, 0, 2)  # shape ->> (batch_size, num_modalities, embed_dim)

        # Add & Norm
        embeddings_mat = embeddings_mat.permute(1, 0, 2)  # Back to original shape for residual connection
        attn_output = (0.5*attn_output) + (0.5*embeddings_mat)  # Add (residual connection) & apply weighted residual connection 
        attn_output = self.norm_layer(attn_output)  # Apply LayerNorm
 
        return attn_output # shape -> (batch_size, num_modalities, embed_dim), as in CAT-DTI paper
    


class FeatureReducer_(nn.Module):
    # Feature reducer for joint attention in PPI structure feature - in order to reduce tensors dim for math operations
    # Use this class if |UniProt_NumOfAminoAcidComp| < 128
    def __init__(self, in_channels, out_channels):
        super(FeatureReducer_, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, in_channels]
        x = x.transpose(1, 2)  # Change shape to [batch_size, in_channels, sequence_length]
        x = self.conv(x)       
        x = x.transpose(1, 2)  # Change shape back to [batch_size, target_length, out_channels]
        return x
        
class FeatureReducer(nn.Module):
    # Feature reducer for joint attention in PPI structure feature - in order to reduce tensors dim for math operations
    # Use this class if |UniProt_NumOfAminoAcidComp| >= 128
    def __init__(self, in_channels, out_channels, target_length):
        super(FeatureReducer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(target_length)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, in_channels]
        x = x.transpose(1, 2)  # Change shape to [batch_size, in_channels, sequence_length]
        x = self.conv(x)    
        x = self.pool(x) 
        x = x.transpose(1, 2)  # Change shape back to [batch_size, target_length, out_channels]
        return x