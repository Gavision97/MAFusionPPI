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


class MFusionPPI(ABSMAFusionPPI):
    def __init__(self):
        super(MFusionPPI, self).__init__()

        logging.info('MFusionPPI model with self, with mlps after flatten @ w/o structure features')
        self.ppi_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.smiles_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=1280, out_features=640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=640, out_features=320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=320, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=578),
            nn.BatchNorm1d(578),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=578, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )        

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=500, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=1200, out_features=600),
            nn.BatchNorm1d(600), 
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=600, out_features=300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=300, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT)
        )

        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 609),
            nn.BatchNorm1d(609),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=609, out_features=300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=300, out_features=256),
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
        
    #def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
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


'''
# With Attention & structure features (second version)
class AUVG_PPI_v2_2(AbstractModel_v2_2):
    def __init__(self, pretrained_chemprop_model, chemberta_model, dropout):
        
        super(AUVG_PPI_v2_2, self).__init__()
        self.pretrained_chemprop_model = pretrained_chemprop_model
        self.chemberta_model = chemberta_model
        self.dropout = dropout
        self.ppi_self_attention = custom_self_attention(512, 8, 0.2)
        self.smiles_self_attention = custom_self_attention(384, 4, 0.2)
        self.cross_attention = nn.MultiheadAttention(512, 8, 0.2)
        self.max_pool = nn.MaxPool1d(2)
        self.compound_dim = 512
        self.W_p1, self.W_p2 = nn.Linear(self.compound_dim, self.compound_dim), nn.Linear(self.compound_dim, self.compound_dim)

        
        # PPI Features MLP layers: (esm, custom, fegs, gae)
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1750),
            nn.ReLU(),
            nn.BatchNorm1d(1750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1750, out_features=1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1000, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )        

        self.custom_mlp = nn.Sequential(
            nn.Linear(in_features=4700 + 4700 , out_features=8000),
            nn.ReLU(),
            nn.BatchNorm1d(8000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=8000, out_features=6500),
            nn.ReLU(),
            nn.BatchNorm1d(6500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=6500, out_features=5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=5000, out_features=3500),
            nn.ReLU(),
            nn.BatchNorm1d(3500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=3500, out_features=2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2000, out_features=1028),
            nn.ReLU(),
            nn.BatchNorm1d(1028),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1028, out_features=512)
        )

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=512 * 5 , out_features= 1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=2100, out_features=1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024), 
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # Morgan fingerprints & chemical descriptors MLP layers
        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # MLP for smiles_embeddings
        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=384 * 3 , out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=256 + 256, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=1)
        )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        #self.sigmoid = nn.Sigmoid()
    # bptfs -> batch protein tuple feature structure
    def forward(self, bmg, bpsf1, bpsf2, esm, custom, fegs, gae,
                input_ids, attention_mask,
                morgan_fingerprints, chemical_descriptors):
        # Forward pass batch mol graph through pretrained chemprop model in order to get fingerprints embeddings
        # Afterwards, pass the fingerprints through MLP layer
        cp_fingerprints = self.pretrained_chemprop_model(bmg)
        cp_fingerprints = self.fp_mlp(cp_fingerprints)

        chemberta_embeddings = self.chemberta_model(input_ids, attention_mask)
        #chemberta_embeddings = self.chemberta_mlp(chemberta_embeddings)
        mfp_chem_descriptors = torch.cat([morgan_fingerprints, chemical_descriptors], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        # Concatenate all 3 smiles embeddings along a new dimension (3x384) & pass them throw self-attention layer
        smiles_embeddings = torch.stack([cp_fingerprints, chemberta_embeddings, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3, 384)
        smiles_features = self.smiles_self_attention(smiles_embeddings)
        smiles_embeddings = self.smiles_mlp(smiles_features).unsqueeze(1)

        # Pass all PPI features  through MLP layers, and then pass them all together into another MLP layer
        esm_embeddings = self.esm_mlp(esm)
        custom_embeddings = self.custom_mlp(custom)
        fegs_embeddings = self.fegs_mlp(fegs)
        gae_embeddings = self.gae_mlp(gae)
        
        # Structure features
        if bpsf1.shape[1] > 128: feature_reducer_p1 = FeatureReducer(in_channels=722, out_channels=512, target_length=128).to(device)
        else: feature_reducer_p1 = FeatureReducer_(in_channels=722, out_channels=512).to(device)
        if bpsf2.shape[1] > 128: feature_reducer_p2 = FeatureReducer(in_channels=722, out_channels=512, target_length=128).to(device)
        else: feature_reducer_p2 = FeatureReducer_(in_channels=722, out_channels=512).to(device)
        bpsf1 = feature_reducer_p1(bpsf1)
        bpsf2 = feature_reducer_p2(bpsf2)
        #print(f'bpsf1 -> {bpsf1.shape}, bpsf2 -> {bpsf2.shape}')
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.W_p1(self.relu(bpsf1)), self.W_p2(self.relu(bpsf2))))
        #print(f'inter_comp_prot -> {inter_comp_prot.shape}')
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)
        #print(f'after, inter_comp_prot -> {inter_comp_prot.shape}')
        
        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', bpsf1, bpsf2))
        #print(cp_embedding.shape)
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)
        #print(f'end, cp_embedding -> {cp_embedding.shape}')
        
        # Concatenate all 4 ppi embeddings along a new dimension (4x512) & pass them throw self-attention layer
        ppi_embeddings = torch.stack([cp_embedding, esm_embeddings, custom_embeddings, fegs_embeddings, gae_embeddings], dim=1).to(device)  # shape ->> (batch_size, 4, 320)
        ppi_features = self.ppi_self_attention(ppi_embeddings)
        ppi_features = self.ppi_mlp(ppi_features).unsqueeze(1)

        #Cross-attention between smiles and PPI to capture the interaction relationships
        ppi_QKV = ppi_features.permute(1, 0, 2)
        smiles_QKV = smiles_embeddings.permute(1, 0, 2)
        
        smiles_att, _ = self.cross_attention(smiles_QKV, ppi_QKV, ppi_QKV)
        ppi_att, _ = self.cross_attention(ppi_QKV, smiles_QKV, smiles_QKV)

        # permute attention outputrs to match (batch_size, embed_dim, num_heads) shape
        smiles_attn_output = (0.5* smiles_att.permute(1, 2, 0)) + (0.5* smiles_embeddings.permute(0, 2, 1))  # Add (residual connection) & apply weighted residual connection 
        ppi_attn_output = (0.5* ppi_att.permute(1, 2, 0)) + (0.5* ppi_features.permute(0, 2, 1))  # Add (residual connection) & apply weighted residual connection 

        # Drop the last dim in order to get (batch_size, embed_dim) & 
        # Pass cross-attention norm outputs throw max-pool layer before passing throw MLP layers
        smiles_att = self.max_pool(smiles_attn_output.squeeze(2))
        ppi_att = self.max_pool(ppi_attn_output.squeeze(2)) 
        combined_embeddings = torch.cat([smiles_att, ppi_att], dim=1)
        output = self.additional_layers(combined_embeddings)
        
        return output
        #return self.sigmoid(output)



'''

'''

compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(protein_feats)), self.joint_attn_comp(self.relu(compound_feats))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', protein_feats, compound_feats))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)


'''