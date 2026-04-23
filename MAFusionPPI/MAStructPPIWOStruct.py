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


def choose_model_setting_wo_structure(
    exclude_features=None,
    mlp_dropout=0.3,
    head_dropout=0.3,
    self_attn_dropout=0.1,
):
    """
    Initialize MFusionPPI without structure features, while excluding
    selected input feature groups.

    Parameters
    ----------
    exclude_features : list[str] or None
        Features to exclude.
        Supported:
        ['chemprop', 'expert', 'chemberta', 'esm', 'fegs', 'gae']

        where:
        - chemprop  -> Chemprop embedding branch (input key: "cpe")
        - expert    -> Morgan + chemical descriptors branch
        - chemberta -> ChemBERTa branch
        - esm/fegs/gae -> PPI branches

    mlp_dropout : float
        Dropout used in feature-specific MLPs.

    head_dropout : float
        Dropout used in final prediction head.

    self_attn_dropout : float
        Dropout used in self-attention blocks.

    Returns
    -------
    MFusionPPI
    """
    logger.info(
        f"Initialized MFusionPPI w/o structure | "
        f"exclude_features={exclude_features} | "
        f"mlp_dropout={mlp_dropout} | "
        f"head_dropout={head_dropout} | "
        f"self_attn_dropout={self_attn_dropout}"
    )

    model = MFusionPPI(
        exclude_features=exclude_features,
        mlp_dropout=mlp_dropout,
        head_dropout=head_dropout,
        self_attn_dropout=self_attn_dropout,
    )
    return model


class MFusionPPI(ABSMAFusionPPI):
    def __init__(
        self,
        exclude_features=None,
        mlp_dropout=0.3,
        head_dropout=0.3,
        self_attn_dropout=0.1,
        proj_dim=256,
    ):
        super().__init__()

        if exclude_features is None:
            exclude_features = []

        self.exclude_features = set(exclude_features)
        self.mlp_dropout = mlp_dropout
        self.head_dropout = head_dropout
        self.self_attn_dropout = self_attn_dropout
        self.proj_dim = proj_dim

        valid_features = {"chemprop", "expert", "chemberta", "esm", "fegs", "gae"}
        invalid = self.exclude_features - valid_features
        if invalid:
            raise ValueError(f"Invalid excluded features: {invalid}")

        # molecule-side branches
        self.use_chemprop = "chemprop" not in self.exclude_features
        self.use_expert = "expert" not in self.exclude_features
        self.use_chemberta = "chemberta" not in self.exclude_features

        # ppi-side branches
        self.use_esm = "esm" not in self.exclude_features
        self.use_fegs = "fegs" not in self.exclude_features
        self.use_gae = "gae" not in self.exclude_features

        logger.info(f"Excluded features: {sorted(self.exclude_features)}")

        self.ppi_self_attention = custom_self_attention(
            embed_dim=proj_dim,
            num_heads=4,
            dropout=self_attn_dropout
        )
        self.smiles_self_attention = custom_self_attention(
            embed_dim=proj_dim,
            num_heads=4,
            dropout=self_attn_dropout
        )

        # ---------- Molecule branches ----------
        if self.use_chemprop:
            self.fp_mlp = nn.Sequential(
                nn.Linear(1200, 600),
                nn.ReLU(),
                nn.BatchNorm1d(600),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(600, 300),
                nn.ReLU(),
                nn.BatchNorm1d(300),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(300, proj_dim)
            )

        if self.use_expert:
            self.mfp_cd_mlp = nn.Sequential(
                nn.Linear(1024 + 194, 609),
                nn.ReLU(),
                nn.BatchNorm1d(609),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(609, 300),
                nn.ReLU(),
                nn.BatchNorm1d(300),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(300, proj_dim)
            )

        if self.use_chemberta:
            self.chemberta_mlp = nn.Sequential(
                nn.Linear(384, proj_dim)
            )

        # ---------- PPI branches ----------
        if self.use_esm:
            self.esm_mlp = nn.Sequential(
                nn.Linear(1280 + 1280, 1280),
                nn.ReLU(),
                nn.BatchNorm1d(1280),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(1280, 640),
                nn.ReLU(),
                nn.BatchNorm1d(640),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(640, 320),
                nn.ReLU(),
                nn.BatchNorm1d(320),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(320, proj_dim)
            )

        if self.use_fegs:
            self.fegs_mlp = nn.Sequential(
                nn.Linear(578 + 578, 578),
                nn.ReLU(),
                nn.BatchNorm1d(578),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(578, proj_dim)
            )

        if self.use_gae:
            self.gae_mlp = nn.Sequential(
                nn.Linear(500 + 500, 500),
                nn.ReLU(),
                nn.BatchNorm1d(500),
                nn.Dropout(p=mlp_dropout),
                nn.Linear(500, proj_dim)
            )

        # ---------- Active features bookkeeping ----------
        self.active_smiles_features = []
        if self.use_chemprop:
            self.active_smiles_features.append("chemprop")
        if self.use_chemberta:
            self.active_smiles_features.append("chemberta")
        if self.use_expert:
            self.active_smiles_features.append("expert")

        self.active_ppi_features = []
        if self.use_esm:
            self.active_ppi_features.append("esm")
        if self.use_fegs:
            self.active_ppi_features.append("fegs")
        if self.use_gae:
            self.active_ppi_features.append("gae")

        n_smiles_tokens = len(self.active_smiles_features)
        n_ppi_tokens = len(self.active_ppi_features)

        if n_smiles_tokens == 0:
            raise ValueError("At least one molecule feature must remain active.")
        if n_ppi_tokens == 0:
            raise ValueError("At least one PPI feature must remain active.")

        # ---------- Fusion MLPs with dynamic input sizes ----------
        self.smiles_mlp = nn.Sequential(
            nn.Linear(proj_dim * n_smiles_tokens, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(512, proj_dim)
        )

        self.ppi_mlp = nn.Sequential(
            nn.Linear(proj_dim * n_ppi_tokens, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(512, proj_dim)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(proj_dim + proj_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=head_dropout),
            nn.Linear(128, 1)
        )

        logger.info(f"Active molecule features: {self.active_smiles_features}")
        logger.info(f"Active PPI features: {self.active_ppi_features}")

    def forward(self, **inputs):
        # ---------- Small molecule module ----------
        smiles_tokens = []

        if self.use_chemprop:
            chemprop_embeddings = self.fp_mlp(inputs["cpe"])
            smiles_tokens.append(chemprop_embeddings)

        if self.use_chemberta:
            chemberta_embeddings = self.chemberta_mlp(inputs["cbae"])
            smiles_tokens.append(chemberta_embeddings)

        if self.use_expert:
            mfp_chem_descriptors = torch.cat(
                [inputs["morgan_fingerprints"], inputs["chemical_descriptors"]],
                dim=1
            )
            expert_embeddings = self.mfp_cd_mlp(mfp_chem_descriptors)
            smiles_tokens.append(expert_embeddings)

        smiles_embeddings = torch.stack(smiles_tokens, dim=1).to(device)
        smiles_embeddings = self.smiles_self_attention(smiles_embeddings)
        flatten_smiles_embed = smiles_embeddings.flatten(start_dim=1)
        smiles_embed = self.smiles_mlp(flatten_smiles_embed)

        # ---------- PPI module ----------
        ppi_tokens = []

        if self.use_esm:
            esm_embeddings = self.esm_mlp(inputs["esm"])
            ppi_tokens.append(esm_embeddings)

        if self.use_fegs:
            fegs_embeddings = self.fegs_mlp(inputs["fegs"])
            ppi_tokens.append(fegs_embeddings)

        if self.use_gae:
            gae_embeddings = self.gae_mlp(inputs["gae"])
            ppi_tokens.append(gae_embeddings)

        ppi_embeddings = torch.stack(ppi_tokens, dim=1).to(device)
        ppi_embeddings = self.ppi_self_attention(ppi_embeddings)
        flatten_ppi_embed = ppi_embeddings.flatten(start_dim=1)
        ppi_embed = self.ppi_mlp(flatten_ppi_embed)

        # ---------- Final fusion ----------
        combined_embeddings = torch.cat([smiles_embed, ppi_embed], dim=1)
        output = self.additional_layers(combined_embeddings)

        return output