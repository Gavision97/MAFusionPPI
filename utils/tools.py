
import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def plot_train_val_auc(
    train_values,
    val_values,
    matric='AUC',
    save_path=None,
    title="Loss Curve"
):
    steps = range(len(train_values))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_values, label=f"Training {matric}", linewidth=2, color="blue")
    plt.plot(steps, val_values, label=f"Validation {matric}", linewidth=2, color="green")

    plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

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