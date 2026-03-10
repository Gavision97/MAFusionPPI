
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from utils.tools import custom_self_attention
from MAFusionPPI.ABSMAFusionPPI import ABSMAFusionPPI

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DROPOUT = 0.3


class MAFusionPPI_(ABSMAFusionPPI):
    def __init__(self):
        super(MAFusionPPI, self).__init__()

        logging.info('MultiPPIMI model with self, with mlps after flatten')
        self.ppi_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.smiles_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1280),
            nn.ReLU(),
            nn.BatchNorm1d(1280),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=1280, out_features=640),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=640, out_features=320),
            nn.ReLU(),
            nn.BatchNorm1d(320),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=320, out_features=256)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=578),
            nn.ReLU(),
            nn.BatchNorm1d(578),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=578, out_features=256)
        )        

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=500, out_features=256)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=512, out_features=256)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=1200, out_features=600), 
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=600, out_features=300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=300, out_features=256)
        )

        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 609),
            nn.ReLU(),
            nn.BatchNorm1d(609),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=609, out_features=300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=300, out_features=256)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(in_features=384, out_features= 256)
        )

        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3 , out_features= 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=512, out_features=256)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=256 + 256, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=128, out_features=1)
        )
        
    #def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
    def forward(self, **inputs):
        cp_fingerprints = self.fp_mlp(inputs.get("cpe"))
        cbae = self.chemberta_mlp(cbae)
        
        mfp_chem_descriptors = torch.cat([inputs.get("morgan_fingerprints"), inputs.get("chemical_descriptors")], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        smiles_embeddings = torch.stack([cp_fingerprints, cbae, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        smiles_embeddings = self.smiles_self_attention(smiles_embeddings)
        
        # Pass all PPI features  through MLP layers, and then pass them all together into another MLP layer

        esm_embeddings = self.esm_mlp(inputs.get("esm"))
        fegs_embeddings = self.fegs_mlp(inputs.get("fegs"))
        gae_embeddings = self.gae_mlp(inputs.get("gae"))

        # Stack all 3 ppi embeddings along a new dimension (3x256) 
        ppi_embeddings = torch.stack([esm_embeddings, fegs_embeddings, gae_embeddings], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        ppi_embeddings = self.ppi_self_attention(ppi_embeddings)

        flatten_smiles_embed = smiles_embeddings.flatten(start_dim=1)
        flatten_ppi_embed = ppi_embeddings.flatten(start_dim=1)

        smiles_embed = self.smiles_mlp(flatten_smiles_embed)
        ppi_embed = self.ppi_mlp(flatten_ppi_embed)
        
        combined_embeddings = torch.cat([smiles_embed, ppi_embed], dim=1)
        output = self.additional_layers(combined_embeddings)
        
        return output
    



class MAFusionPPI(ABSMAFusionPPI):
    def __init__(self):
        super(MAFusionPPI, self).__init__()
        logging.info('MultiPPIMI model with self, with mlps after flatten & Structure Features')
        self.ppi_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.smiles_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=1280, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )        

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 4, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=1200, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(in_features=384, out_features= 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3 , out_features= 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=256 + 256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=128, out_features=1)
        )
        
    def forward(self, **inputs):
        
        cp_fingerprints = self.fp_mlp(inputs.get("cpe"))
        cbae = self.chemberta_mlp(inputs.get("cbae"))
        
        mfp_chem_descriptors = torch.cat([inputs.get("morgan_fingerprints"), inputs.get("chemical_descriptors")], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        smiles_embeddings = torch.stack([cp_fingerprints, cbae, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        smiles_embeddings = self.smiles_self_attention(smiles_embeddings)
        
        # Pass all PPI features  through MLP layers, and then pass them all together into another MLP layer

        esm_embeddings = self.esm_mlp(inputs.get("esm"))
        fegs_embeddings = self.fegs_mlp(inputs.get("fegs"))
        gae_embeddings = self.gae_mlp(inputs.get("gae"))
        ppi_former_embeddings = torch.cat([inputs.get("ppi_former_a"), inputs.get("ppi_former_b")], dim=1) # (B, 128+128=256)

        # Stack all 3 ppi embeddings along a new dimension (3x256) 
        ppi_embeddings = torch.stack([esm_embeddings, fegs_embeddings, gae_embeddings, ppi_former_embeddings], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        ppi_embeddings = self.ppi_self_attention(ppi_embeddings)

        flatten_smiles_embed = smiles_embeddings.flatten(start_dim=1)
        flatten_ppi_embed = ppi_embeddings.flatten(start_dim=1)

        smiles_embed = self.smiles_mlp(flatten_smiles_embed)
        ppi_embed = self.ppi_mlp(flatten_ppi_embed)
        
        combined_embeddings = torch.cat([smiles_embed, ppi_embed], dim=1)
        output = self.additional_layers(combined_embeddings)
        
        return output