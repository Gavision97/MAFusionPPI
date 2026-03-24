import os
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import h5py
import random
import torch
from torch.utils.data import Dataset, DataLoader


PPI_EMB_PATH = 'embeddings/ppi/'
SMI_EMB_PATH = 'embeddings/smi/'
PPI_INFO_PATH = 'datasets/ppi_strcuture_information_with_id.csv'


PPI_STRUCT_EMB_H5_PATH = 'embeddings/ppi struct/'




class BaseMoleculeDataset(Dataset):
    """
    Loads & preprocesses all PPI + SMILES features once.
    Subclasses control what __getitem__ returns (train vs eval).
    """
    def __init__(self, ds_, use_struct=True, struct_dataset='dataset1', strategy = "conditional", aug_train=False, eval_all_confs=False):
       
        if aug_train and eval_all_confs:
            raise ValueError("aug_train and eval_all_confs should not both be True.")
        
        logger.info(f'MolecularDataset hyperparamets -> dataset={struct_dataset}, trategy={strategy}, aug train={aug_train}, aug eval={eval_all_confs}')
        self.data = ds_.reset_index(drop=True)
        self.use_struct = use_struct
        # ----------------- PPI features ----------------- #
        if use_struct:
            ppi_struct_emb_path = os.path.join(PPI_STRUCT_EMB_H5_PATH, f"{struct_dataset}_processed_matrix_data_with_padding.h5")
            self.ppi_struct_emb = h5py.File(ppi_struct_emb_path, "r") # PPI structure embeddings stored in h5py object
            self.ppi_stuct_strategy = strategy if strategy in ['conditional', 'full_mean', 'subset_mean'] else 'conditional'
            self.aug_train = aug_train
            self.eval_all_confs = eval_all_confs

            self.ppi_info_df = pd.read_csv(PPI_INFO_PATH)
            self.af3_model2indx = {'model0': '1', 'model1': '2', 'model2': '3', 'model3': '4', 'model4': '5'}
            self.best_conf_by_ppi = self._build_best_conf_map(struct_dataset)


        self.esm  = pd.read_csv(os.path.join(PPI_EMB_PATH, "esm_features.csv"))
        self.fegs = pd.read_csv(os.path.join(PPI_EMB_PATH, "fegs_features.csv"))
        self.gae  = pd.read_csv(os.path.join(PPI_EMB_PATH, "gae_features.csv"))

        self.gae_features_ppi  = self._merge_ppi(self.data, self.gae).drop(columns=["smiles", "label"]).to_numpy(np.float32)
        self.esm_features_ppi  = self._merge_ppi(self.data, self.esm).drop(columns=["smiles", "label"]).to_numpy(np.float32)
        self.fegs_features_ppi = self._merge_ppi(self.data, self.fegs).drop(columns=["smiles", "label"]).to_numpy(np.float32)

        # --------------------- SMILES features ------------------
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join(SMI_EMB_PATH, "smiles_morgan_fingerprints_dataset.csv"))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join(SMI_EMB_PATH, "smiles_chem_descriptors_mapping_dataset.csv"))
        self.chemprop = pd.read_csv(os.path.join(SMI_EMB_PATH, "chemprop_features.csv"))
        self.chemberta = pd.read_csv(os.path.join(SMI_EMB_PATH, "chemBERTa_features.csv"))

        # fast lookup maps
        self.morgan_map = self.smiles_morgan_fingerprints.set_index("SMILES")
        self.desc_map = self.smiles_chemical_descriptors.set_index("SMILES")
        self.chemprop_map = self.chemprop.set_index("SMILES")
        self.chemberta_map = self.chemberta.set_index("SMILES")

    def _pad_to_256(self, x):
        """
        Pad (or trim) matrix to 256 rows.
        x: Tensor [L, D]
        return: Tensor [256, D]
        """
        L, D = x.shape
    
        if L > 256:
            return x[:256]
    
        if L < 256:
            pad = torch.zeros(256 - L, D, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        return x

    def _merge_ppi(self, dataset, features_df):
        dataset_ = dataset.drop(columns=['ppi_id']) if 'ppi_id' in list(dataset.columns) else dataset
        out = dataset_.merge(features_df, how="left", left_on="uniprot_id1", right_on="UniProt_ID",
                            suffixes=("", "_id1")).drop(columns=["UniProt_ID"])

        features_df_renamed = features_df.add_suffix("_id2").rename(columns={"UniProt_ID_id2": "UniProt_ID"})
        out = out.merge(features_df_renamed, how="left", left_on="uniprot_id2", right_on="UniProt_ID",
                        suffixes=("", "_id2")).drop(columns=["UniProt_ID", "uniprot_id1", "uniprot_id2"])
        
        out.fillna(0, inplace=True) # fill missing values (na, None etc..) with zero
        return out

    def __len__(self):
        return len(self.data)


    def _build_best_conf_map(self, struct_dataset: str) -> dict:
        """
        Build mapping:
            ppi_id -> h5 conformer index ('1'..'5')

        based on columns like:
            dataset1_model_0_ranking, ..., dataset1_model_4_ranking
        """

        ranking_cols = [f"{struct_dataset}_model_{i}_ranking" for i in range(5)]

        # make sure all needed columns exist
        missing_cols = [c for c in ranking_cols if c not in self.ppi_info_df.columns]
        if missing_cols:
            raise ValueError(f"Missing ranking columns in ppi_info_df: {missing_cols}")

        if "ppi_id" not in self.ppi_info_df.columns:
            raise ValueError("ppi_info_df must contain a 'ppi_id' column")

        best_conf_by_ppi = {}

        for _, row in self.ppi_info_df.iterrows():
            ppi_id = row["ppi_id"]

            ranks = row[ranking_cols]
            ranks = pd.to_numeric(ranks, errors="coerce")

            # if all missing -> fallback to first conformation
            if ranks.isna().all():
                best_conf_by_ppi[ppi_id] = '1'
                continue

            best_col = ranks.idxmax()   # higher is better (e.g. "dataset1_model_3_ranking")
            # extract model number from column name; "dataset1_model_3_ranking" -> "model3"
            model_token = best_col.split("_model_")[1].split("_ranking")[0]   # "3"
            model_name = f"model{model_token}"                                 # "model3"
    
            best_conf_by_ppi[ppi_id] = self.af3_model2indx[model_name] # map to h5 conformer index

        return best_conf_by_ppi

    def _get_inputs_y_meta(self, idx):
        smiles = self.data.loc[idx, "smiles"]
        uniprot1 = self.data.loc[idx, "uniprot_id1"]
        uniprot2 = self.data.loc[idx, "uniprot_id2"]
        meta = (smiles, uniprot1, uniprot2, 'w/o ppi_id')
        y = torch.tensor(self.data.loc[idx, "label"], dtype=torch.float32)

        # ---------- structure embeddings ---------- #
        if self.use_struct:
            ppi_id = self.data.loc[idx, "ppi_id"]
            meta = (smiles, uniprot1, uniprot2, ppi_id)

            if self.eval_all_confs:
                ppi_former_a, ppi_former_b, ppi_omega_a, ppi_omega_b, ppi_progress_vec = \
                    self._get_all_conformation_struct_features(ppi_id)
            else:
                if self.aug_train:
                    ppi_conformer_id = random.choice(list(self.ppi_struct_emb[self.ppi_stuct_strategy].keys()))
                else:
                    ppi_conformer_id = self.best_conf_by_ppi.get(ppi_id, '1')

                grp = self.ppi_struct_emb[self.ppi_stuct_strategy][ppi_conformer_id][ppi_id]

                ppi_former_a = self._pad_to_256(torch.tensor(grp['ppi_former_a'][:], dtype=torch.float32))
                ppi_former_b = self._pad_to_256(torch.tensor(grp['ppi_former_b'][:], dtype=torch.float32))
                ppi_omega_a = self._pad_to_256(torch.tensor(grp['ifeature_omega_a'][:], dtype=torch.float32))
                ppi_omega_b = self._pad_to_256(torch.tensor(grp['ifeature_omega_b'][:], dtype=torch.float32))
    
                ppi_progress_vec = torch.tensor(grp['progres_vector'][:], dtype=torch.float32)

        # ---------- sequence / graph / smiles features ---------- #
        esm_features = torch.from_numpy(self.esm_features_ppi[idx])
        fegs_features = torch.from_numpy(self.fegs_features_ppi[idx])
        gae_features = torch.from_numpy(self.gae_features_ppi[idx])

        morgan = torch.from_numpy(self.morgan_map.loc[smiles].to_numpy(np.float32))
        chem_desc = torch.from_numpy(self.desc_map.loc[smiles].to_numpy(np.float32))
        chemprop = torch.from_numpy(self.chemprop_map.loc[smiles].to_numpy(np.float32))
        chemberta = torch.from_numpy(self.chemberta_map.loc[smiles].to_numpy(np.float32))

        if self.use_struct:
            inputs = {
                "ppi_former_a": ppi_former_a,
                "ppi_former_b": ppi_former_b,
                "ppi_omega_a": ppi_omega_a,
                "ppi_omega_b": ppi_omega_b,
                "ppi_progress_vec": ppi_progress_vec,
                "cpe": chemprop,
                "esm": esm_features,
                "fegs": fegs_features,
                "gae": gae_features,
                "cbae": chemberta,
                "morgan_fingerprints": morgan,
                "chemical_descriptors": chem_desc,
            }
        else:
            inputs = {
                "cpe": chemprop,
                "esm": esm_features,
                "fegs": fegs_features,
                "gae": gae_features,
                "cbae": chemberta,
                "morgan_fingerprints": morgan,
                "chemical_descriptors": chem_desc,
            }

        return inputs, y, meta

    def _get_all_conformation_struct_features(self, ppi_id):
        former_a_list = []
        former_b_list = []
        omega_a_list = []
        omega_b_list = []
        progress_list = []

        conf_ids = sorted(list(self.ppi_struct_emb[self.ppi_stuct_strategy].keys()), key=int)

        for conf_id in conf_ids:
            grp = self.ppi_struct_emb[self.ppi_stuct_strategy][conf_id][ppi_id]

            ppi_former_a = self._pad_to_256(torch.tensor(grp['ppi_former_a'][:], dtype=torch.float32))
            ppi_former_b = self._pad_to_256(torch.tensor(grp['ppi_former_b'][:], dtype=torch.float32))
            ppi_omega_a = self._pad_to_256(torch.tensor(grp['ifeature_omega_a'][:], dtype=torch.float32))
            ppi_omega_b = self._pad_to_256(torch.tensor(grp['ifeature_omega_b'][:], dtype=torch.float32))

            ppi_progress_vec = torch.tensor(grp['progres_vector'][:], dtype=torch.float32)

            former_a_list.append(former_a)
            former_b_list.append(former_b)
            omega_a_list.append(omega_a)
            omega_b_list.append(omega_b)
            progress_list.append(progress)

        return (
            torch.stack(former_a_list, dim=0),
            torch.stack(former_b_list, dim=0),
            torch.stack(omega_a_list, dim=0),
            torch.stack(omega_b_list, dim=0),
            torch.stack(progress_list, dim=0),
        )
class TrainMoleculeDataset(BaseMoleculeDataset):
    def __getitem__(self, idx):
        inputs, y, _ = self._get_inputs_y_meta(idx)
        return inputs, y

    @staticmethod
    def collate_train(batch):
        """
        Collate training batch into:
            inputs_dict, y
        where each tensor in inputs_dict is stacked over batch dimension.
        """
        inputs_list, ys = zip(*batch)

        keys = inputs_list[0].keys()
        stacked_inputs = {
            k: torch.stack([sample[k] for sample in inputs_list], dim=0)
            for k in keys
        }
        y = torch.stack(ys, dim=0)
        return stacked_inputs, y


class EvalMoleculeDataset(BaseMoleculeDataset):
    def __getitem__(self, idx):
        return self._get_inputs_y_meta(idx)

    @staticmethod
    def collate_eval(batch):
        """
        Collate evaluation batch into:
            inputs_dict, y, (smiles, uniprot1, uniprot2)
        """
        inputs_list, ys, metas = zip(*batch)

        keys = inputs_list[0].keys()
        stacked_inputs = {
            k: torch.stack([sample[k] for sample in inputs_list], dim=0)
            for k in keys
        }
        y = torch.stack(ys, dim=0)

        smiles, uniprot1, uniprot2, ppi_id = zip(*metas)
        return stacked_inputs, y, (smiles, uniprot1, uniprot2, ppi_id)

class MoleculeDataset(Dataset):
    """
    custom dataset; loads & preprocess all PPI & small molecule features
    """
    def __init__(self, ds_):
        logger.info('Initializing MoleculeDataset ...')
        self.data = ds_.reset_index(drop=True)  # cols -> smiles, uniprot1, uniprot2, label

        # PPI features
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter="\t")
        self.esm = pd.read_csv(os.path.join(PPI_EMB_PATH, 'esm_features.csv'))
        self.fegs = pd.read_csv(os.path.join(PPI_EMB_PATH, 'fegs_features.csv'))
        self.gae = pd.read_csv(os.path.join(PPI_EMB_PATH, 'gae_features.csv'))

        
        self.gae.loc[self.gae['predicted'] == 1, self.gae.columns[9:509]] = 0 
        gae_features_columns = self.gae.iloc[:, 9:509]
        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)

        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)

        # SMILES features
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join(SMI_EMB_PATH, 'smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join(SMI_EMB_PATH, 'smiles_chem_descriptors_mapping_dataset.csv'))
        self.chemprop = pd.read_csv(os.path.join(SMI_EMB_PATH, 'chemprop_features.csv'))
        self.chemberta = pd.read_csv(os.path.join(SMI_EMB_PATH, 'chemBERTa_features.csv'))

        # build indexes for fast lookup in __getittem__(...)
        self.morgan_map   = self.smiles_morgan_fingerprints.set_index("SMILES")
        self.desc_map     = self.smiles_chemical_descriptors.set_index("SMILES")
        self.chemprop_map = self.chemprop.set_index("SMILES")
        self.chemberta_map= self.chemberta.set_index("SMILES")

    def merge_datasets(self, dataset, features_df):
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID',
                                suffixes=('', '_id1')).drop(columns=['UniProt_ID'])

        features_df_renamed = features_df.add_suffix('_id2').rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID',
                                suffixes=('', '_id2')).drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])

        dataset.fillna(0, inplace=True)
        return dataset.drop(columns=['zero_count'])

    @staticmethod
    def collate_train(batch):
        inputs_list, ys, _metas = zip(*batch)
        features_by_type = list(zip(*inputs_list))
        stacked_inputs = [torch.stack([torch.as_tensor(f) for f in feats], 0) for feats in features_by_type]
        y = torch.stack(ys, 0)
        return stacked_inputs, y

    @staticmethod
    def collate_eval(batch):
        inputs_list, ys, metas = zip(*batch)
        features_by_type = list(zip(*inputs_list))
        stacked_inputs = [torch.stack([torch.as_tensor(f) for f in feats], 0) for feats in features_by_type]
        y = torch.stack(ys, 0)
        smiles, uniprot1, uniprot2 = zip(*metas)
        return stacked_inputs, y, (smiles, uniprot1, uniprot2)

    def make_loader(self, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=MoleculeDataset.collate_fn
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # smiles, uniprot_id1,uniprot_id2,label
        smiles = self.data.loc[idx, "smiles"]
        uniprot1, uniprot2 = self.data.loc[idx, "uniprot_id1"], self.data.loc[idx, "uniprot_id2"]
        meta = (smiles, uniprot1, uniprot2)
        y = torch.tensor(self.data.loc[idx, "label"], dtype=torch.float32)

        # PPI features (already aligned by idx because you built *_features_ppi using self.data order)
        esm_features = torch.from_numpy(self.esm_features_ppi.iloc[idx].values.astype(np.float32))
        fegs_features = torch.from_numpy(self.fegs_features_ppi.iloc[idx].values.astype(np.float32))
        gae_features = torch.from_numpy(self.gae_features_ppi.iloc[idx].values.astype(np.float32))

        # used predefined indeces for fast lookup 
        morgan   = self.morgan_map.loc[smiles].to_numpy(np.float32)
        chem_desc= self.desc_map.loc[smiles].to_numpy(np.float32)
        chemprop = self.chemprop_map.loc[smiles].to_numpy(np.float32)
        chemberta= self.chemberta_map.loc[smiles].to_numpy(np.float32)
        
        inputs = [chemprop, esm_features, fegs_features, gae_features, chemberta, morgan, chem_desc]
        return inputs, y, meta


