
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from utils.tools import custom_self_attention, GatedFeatureFusion
from MAFusionPPI.ABSMAFusionPPI import ABSMAFusionPPI

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def choose_model_setting(exclude_modalities=None, mlp_dropout=0.3, head_dropout=0.3, self_attn_dropout=0.1,
                         join_attn_feat='both', compound_dim=850, compound_proj_dim=256, ppi_fuse_setting='cat',
                         head_fuse='cat', proj_feat=False):
    """
    Initialize MAFusionPPI model with configurable architecture and dropout settings.

    Parameters
    ----------
    exclude_modalities : list[str] or None, optional
        PPI sequence modalities to exclude. Options: ['esm', 'fegs', 'gae'].
    mlp_dropout : float, optional
        Dropout rate used in modality-specific MLP encoders.
    head_dropout : float, optional
        Dropout rate used in the final prediction head.
    self_attn_dropout : float, optional
        Dropout rate used in self-attention layers.
    join_attn_feat : str, optional
        Structure features used in joint attention. Options: ['both', 'ppiformer', 'omega'].
    compound_dim : int, optional
        Input dimension of structure embeddings before projection.
    compound_proj_dim : int, optional
        Projected dimension used in joint attention.
    ppi_fuse_setting : str, optional
        Method for fusing sequence and structure PPI embeddings.
        Options: ['cat', 'gate', 'self_attn'].
    head_fuse : str, optional
        Method for final fusion between PPI and SMILES embeddings.
        Options: ['cat', 'gate'].
    proj_feat : bool, optional
        If True, use lightweight projection layers for feature encoders
        instead of deeper MLP stacks.

    Returns
    -------
    MAFusionPPI
        Initialized model with the specified configuration.
    """

    model = MAFusionPPI(
        exclude_modalities=exclude_modalities,
        mlp_dropout=mlp_dropout,
        head_dropout=head_dropout,
        self_attn_dropout=self_attn_dropout,
        join_attn_feat=join_attn_feat,
        compound_dim=compound_dim,
        compound_proj_dim=compound_proj_dim,
        ppi_fuse_setting=ppi_fuse_setting,
        head_fuse = head_fuse,
        proj_feat=proj_feat
    )

    logger.info(
        f"Initialized model | "
        f"excluded={exclude_modalities} | "
        f"mlp_dropout={mlp_dropout} | "
        f"head_dropout={head_dropout} | "
        f"self_attn_dropout={self_attn_dropout} | "
        f"join_attn_feat={join_attn_feat} | "
        f"compound_dim={compound_dim} | "
        f"compound_proj_dim={compound_proj_dim} | "
        f"ppi_fuse_setting={ppi_fuse_setting} | "
        f"head_fuse={head_fuse} | "
        f"proj_feat={proj_feat}"
    )
    return model


class MAFusionPPI(ABSMAFusionPPI):
    def __init__(
        self,
        exclude_modalities=None,
        mlp_dropout=0.3,
        head_dropout=0.3,
        self_attn_dropout=0.1,
        join_attn_feat = 'both', # ['both', 'ppiformer', 'omega']
        compound_dim = 850, # ppi former -> 128, ifeature omega = 722, both = 128 + 722
        compound_proj_dim = 256,
        ppi_fuse_setting = 'cat', # ['cat', 'gate', 'self_attn']
        head_fuse = 'cat', # ['cat', 'gate']
        proj_feat = False,
    ):
        super().__init__()

        if exclude_modalities is None:
            exclude_modalities = []

        self.exclude_modalities = set(exclude_modalities)
        self.mlp_dropout = mlp_dropout
        self.head_dropout = head_dropout
        self.self_attn_dropout = self_attn_dropout
        self.ppi_fuse_setting = ppi_fuse_setting  # ['cat', 'gate', 'self_attn']
        self.head_fuse = head_fuse # ['cat', 'gate']
        self.proj_feat = proj_feat
        # structure joint-attention hyperparameters & variables #
        self.join_attn_feat = join_attn_feat
        self.compound_dim = compound_dim
        self.compound_proj_dim = compound_proj_dim
        self.W_p1, self.W_p2 = nn.Linear(self.compound_proj_dim, self.compound_proj_dim), nn.Linear(self.compound_proj_dim, self.compound_proj_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.ppi_gate = GatedFeatureFusion(compound_proj_dim)
        self.fuse_gate = GatedFeatureFusion(compound_proj_dim)
        self.ppi_self_attention = custom_self_attention(embed_dim=compound_proj_dim, num_heads=4, dropout=self_attn_dropout)
        self.smiles_self_attention = custom_self_attention(embed_dim=compound_proj_dim, num_heads=4, dropout=self_attn_dropout)


        valid_modalities = {"esm", "fegs","gae"} # TODO -> later, check also small molecule features
        invalid = self.exclude_modalities - valid_modalities
        if invalid:
            raise ValueError(f"Invalid excluded modalities: {invalid}")

        self.use_esm = "esm" not in self.exclude_modalities
        self.use_fegs = "fegs" not in self.exclude_modalities
        self.use_gae = "gae" not in self.exclude_modalities

        logger.info(f"Excluded modalities: {sorted(self.exclude_modalities)}")


        self.join_attn_proj_a = nn.Sequential(
            nn.Linear(in_features=compound_dim, out_features=compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )
        self.join_attn_proj_b = nn.Sequential(
            nn.Linear(in_features=compound_dim, out_features=compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )
   
        # ---------- Small molecule branch ----------
        if self.proj_feat:
            self.fp_mlp = nn.Sequential(
                nn.Linear(1200, compound_proj_dim),
                nn.BatchNorm1d(compound_proj_dim)
            )
        else:
            self.fp_mlp = nn.Sequential(
                nn.Linear(1200, 600),
                nn.ReLU(),
                nn.BatchNorm1d(600),
                nn.Dropout(mlp_dropout),
                nn.Linear(600, 300),
                nn.ReLU(),
                nn.BatchNorm1d(300),
                nn.Dropout(mlp_dropout),
                nn.Linear(300, compound_proj_dim)
            )

        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(1024 + 194, 609),
            nn.ReLU(),
            nn.BatchNorm1d(609),
            nn.Dropout(mlp_dropout),
            nn.Linear(609, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(mlp_dropout),
            nn.Linear(300, compound_proj_dim)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )

        self.smiles_mlp = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, compound_proj_dim)
        )

        if self.use_esm:
            if self.proj_feat:
                self.esm_mlp = nn.Sequential(
                    nn.Linear(1280 + 1280, compound_proj_dim),
                    nn.BatchNorm1d(compound_proj_dim)
            )
            else:
                self.esm_mlp = nn.Sequential(
                    nn.Linear(1280 + 1280, 1280),
                    nn.ReLU(),
                    nn.BatchNorm1d(1280),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(1280, 640),
                    nn.ReLU(),
                    nn.BatchNorm1d(640),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(640, 320),
                    nn.ReLU(),
                    nn.BatchNorm1d(320),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(320, compound_proj_dim)
                )

        if self.use_fegs:
            if self.proj_feat:
                self.esfegs_mlpm_ml = nn.Sequential(
                    nn.Linear(578 + 578, compound_proj_dim),
                    nn.BatchNorm1d(compound_proj_dim)
            )
            else:
                self.fegs_mlp = nn.Sequential(
                    nn.Linear(578 + 578, 578),
                    nn.ReLU(),
                    nn.BatchNorm1d(578),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(578, compound_proj_dim)
                )

        if self.use_gae:
            if self.proj_feat:
                self.esfegs_mlpm_ml = nn.Sequential(
                    nn.Linear(500 + 500, compound_proj_dim),
                    nn.BatchNorm1d(compound_proj_dim)
            )
            else:
                self.gae_mlp = nn.Sequential(
                    nn.Linear(500 + 500, 500),
                    nn.ReLU(),
                    nn.BatchNorm1d(500),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(500, compound_proj_dim)
                )
    

        self.active_modalities = []
        if self.use_esm:
            self.active_modalities.append("esm")
        if self.use_fegs:
            self.active_modalities.append("fegs")
        if self.use_gae:
            self.active_modalities.append("gae")

        n_ppi_tokens = len(self.active_modalities)
        if n_ppi_tokens == 0:
            raise ValueError("At least one PPI modality must remain active.")

        # in -> 256 * num_of_sequence_features + structure_dim (ppiformer 128, omega 722, and both 850)
        self.ppi_seq_proj = nn.Sequential(
            nn.Linear(compound_proj_dim * n_ppi_tokens, compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )
        
        # in case we cat(seq_emb, struct_emb) -> 512 -> 256 using proj
        self.ppi_proj = nn.Sequential(
            nn.Linear(compound_proj_dim + compound_proj_dim, compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )

        # in case we fuse by concat(smiles_emb, ppi_emb), project first to 128
        self.head_proj = nn.Sequential(
            nn.Linear(compound_proj_dim*2, compound_proj_dim),
            nn.BatchNorm1d(compound_proj_dim)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(compound_proj_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1)
        )

        logger.info(f"Active modalities: {self.active_modalities}")

    def forward(self, **inputs):
        # ---------- Small molecule module ----------
        cp_fingerprints = self.fp_mlp(inputs["cpe"])
        cbae = self.chemberta_mlp(inputs["cbae"])

        mfp_chem_descriptors = torch.cat(
            [inputs["morgan_fingerprints"], inputs["chemical_descriptors"]],
            dim=1
        )
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)

        smiles_embeddings = torch.stack([cp_fingerprints, cbae, mfp_chem_descriptors], dim=1).to(device)

        smiles_embeddings = self.smiles_self_attention(smiles_embeddings)
        flatten_smiles_embed = smiles_embeddings.flatten(start_dim=1)
        smiles_embed = self.smiles_mlp(flatten_smiles_embed)

        # ---------- PPI module ----------
        ppi_tokens_emb = []

        if self.use_esm:
            ppi_tokens_emb.append(self.esm_mlp(inputs["esm"]))

        if self.use_fegs:
            ppi_tokens_emb.append(self.fegs_mlp(inputs["fegs"]))

        if self.use_gae:
            ppi_tokens_emb.append(self.gae_mlp(inputs["gae"]))

        sequence_ppi_embeddings = self.ppi_seq_proj(torch.cat(ppi_tokens_emb, dim=-1)) # concate all sequence-based language models -> proj to (B, 256)
        
        # structure embeddings
        if self.join_attn_feat == "both":
            ppi_former_a_emb, ppi_former_b_emb = inputs.get("ppi_former_a"), inputs.get("ppi_former_b")
            ppi_omega_a_emb, ppi_omega_b_emb = inputs.get("ppi_omega_a"), inputs.get("ppi_omega_b")
            join_emb_a = self.join_attn_proj_a(torch.cat([ppi_former_a_emb, ppi_omega_a_emb], dim=-1)) # (B, 256, 128+722) - > (B, 256, compound_proj_dim)
            join_emb_b = self.join_attn_proj_b(torch.cat([ppi_former_b_emb, ppi_omega_b_emb], dim=-1)) # (B, 256, 128+722) - > (B, 256, compound_proj_dim)
        elif self.join_attn_feat == "ppiformer":
            ppi_former_a_emb, ppi_former_b_emb = inputs.get("ppi_former_a"), inputs.get("ppi_former_b")
            join_emb_a = self.join_attn_proj_a(ppi_former_a_emb) # (B, 256, 128) - > (B, 256, compound_proj_dim)
            join_emb_b = self.join_attn_proj_b(ppi_former_b_emb) # (B, 256, 128) - > (B, 256, compound_proj_dim)
        else:
            ppi_omega_a_emb, ppi_omega_b_emb = inputs.get("ppi_omega_a"), inputs.get("ppi_omega_b")
            join_emb_a = self.join_attn_proj_a(ppi_omega_a_emb) # (B, 256, 722) - > (B, 256, compound_proj_dim)
            join_emb_b = self.join_attn_proj_b(ppi_omega_b_emb) # (B, 256, 722) - > (B, 256, compound_proj_dim)


        #print(f'bpsf1 -> {join_emb_a.shape}, bpsf2 -> {join_emb_b.shape}') #  bpsf1 -> torch.Size([16, 256, 850]), bpsf2 -> torch.Size([16, 256, 850])
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.W_p1(self.relu(join_emb_a)), self.W_p2(self.relu(join_emb_b))))
        #print(f'inter_comp_prot -> {inter_comp_prot.shape}') # inter_comp_prot -> torch.Size([16, 256, 256])
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)
        #print(f'after, inter_comp_prot -> {inter_comp_prot.shape}') # after, inter_comp_prot -> torch.Size([16, 256, 256])
        
        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', join_emb_a, join_emb_b))
        #print(cp_embedding.shape) # torch.Size([16, 256, 256, 850])
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)
        #print(f'end, cp_embedding -> {cp_embedding.shape}') # end, cp_embedding -> torch.Size([16, 850]

        # -------- PPI & Small molecule fusion module ---------- #
        # fuse PPI features -> sequence-based 1D & structure-based 3D
        # in all cases, final fused PPI embedding vector is (B, 256)
        if self.ppi_fuse_setting == 'cat':
            ppi_embed = self.ppi_proj(torch.cat([cp_embedding, sequence_ppi_embeddings], dim=-1)) # cat and then proj to (B, 256)
        elif self.ppi_fuse_setting == 'gate':
            ppi_embed = self.ppi_gate(cp_embedding, sequence_ppi_embeddings) # gated, already (B, 256)
        else: # self-attention fuse
            ppi_emb = torch.stack([cp_embedding, sequence_ppi_embeddings], dim=1).to(device) 
            ppi_emb = self.ppi_self_attention(ppi_emb)
            ppi_emb = self.ppi_proj(ppi_emb.flatten(start_dim=1)) # flatten to (B, 256 *2) -> proj to (B, 256)

        #print(f'PPI embeddings shape after fuse -> {ppi_embed.shape}') # PPI embeddings shape after cat -> torch.Size([16, 1618])
        
        #---------- Final fusion ----------
        if self.head_fuse == 'cat':
            combined_embeddings = self.head_proj(torch.cat([smiles_embed, ppi_embed], dim=1)) # (256)
        else: # gate
            combined_embeddings = self.fuse_gate(smiles_embed, ppi_embed)
            
        output = self.additional_layers(combined_embeddings)
        return output