
import os
import logging

import torch

import torch.nn as nn

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np


if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"

def convert_uniprot_ids(dataset, mapping_df):
    # Create a dictionary from the mapping dataframe
    mapping_dict = mapping_df.set_index('From')['Entry'].to_dict()

    # Map the uniprot_id1 and uniprot_id2 columns to their respective Entry values
    dataset['uniprot_id1'] = dataset['uniprot_id1'].map(mapping_dict)
    dataset['uniprot_id2'] = dataset['uniprot_id2'].map(mapping_dict)
    return dataset.drop_duplicates()

def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def scaffold_split(df, smiles_col="smiles", frac_train=0.8, seed=0):
    # group indices by scaffold
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col].values):
        scaf = murcko_scaffold(smi)
        scaffolds.setdefault(scaf, []).append(i)

    # sort scaffold groups by size (largest first), deterministic shuffle on ties
    rng = np.random.RandomState(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)
    scaffold_sets.sort(key=len, reverse=True)

    n_total = len(df)
    n_train = int(frac_train * n_total)

    train_idx, val_idx = [], []
    for sset in scaffold_sets:
        if len(train_idx) + len(sset) <= n_train:
            train_idx.extend(sset)
        else:
            val_idx.extend(sset)

    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()



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