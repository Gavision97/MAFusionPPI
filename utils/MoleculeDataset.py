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

PPI_STRUCT_EMB_H5_PATH = 'embeddings/ppi struct/'

class BaseMoleculeDataset(Dataset):
    """
    Loads & preprocesses all PPI + SMILES features once.
    Subclasses control what __getitem__ returns (train vs eval).
    """
    def __init__(self, ds_, struct_dataset='dataset1', strategy = "conditional", aug=False):
        self.data = ds_.reset_index(drop=True)

        # ---- PPI features ----
        ppi_struct_emb_path = os.path.join(PPI_STRUCT_EMB_H5_PATH, f"{struct_dataset}_processed_matrix_data.h5")
        self.ppi_struct_emb = h5py.File(ppi_struct_emb_path, "r") # PPI structure embeddings stored in h5py object
        self.ppi_stuct_strategy = strategy if strategy in ['conditional', 'full_mean', 'subset_mean'] else 'conditional'
        self.aug = aug

        self.esm  = pd.read_csv(os.path.join(PPI_EMB_PATH, "esm_features.csv"))
        self.fegs = pd.read_csv(os.path.join(PPI_EMB_PATH, "fegs_features.csv"))
        self.gae  = pd.read_csv(os.path.join(PPI_EMB_PATH, "gae_features.csv"))

        self.gae_features_ppi  = self._merge_ppi(self.data, self.gae).drop(columns=["smiles", "label"]).to_numpy(np.float32)
        self.esm_features_ppi  = self._merge_ppi(self.data, self.esm).drop(columns=["smiles", "label"]).to_numpy(np.float32)
        self.fegs_features_ppi = self._merge_ppi(self.data, self.fegs).drop(columns=["smiles", "label"]).to_numpy(np.float32)

        # ---- SMILES features ----
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join(SMI_EMB_PATH, "smiles_morgan_fingerprints_dataset.csv"))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join(SMI_EMB_PATH, "smiles_chem_descriptors_mapping_dataset.csv"))
        self.chemprop = pd.read_csv(os.path.join(SMI_EMB_PATH, "chemprop_features.csv"))
        self.chemberta = pd.read_csv(os.path.join(SMI_EMB_PATH, "chemBERTa_features.csv"))

        # fast lookup maps
        self.morgan_map = self.smiles_morgan_fingerprints.set_index("SMILES")
        self.desc_map = self.smiles_chemical_descriptors.set_index("SMILES")
        self.chemprop_map = self.chemprop.set_index("SMILES")
        self.chemberta_map = self.chemberta.set_index("SMILES")

    def _merge_ppi(self, dataset, features_df):
        dataset_ = dataset.drop(columns=['ppi_id'])
        out = dataset_.merge(features_df, how="left", left_on="uniprot_id1", right_on="UniProt_ID",
                            suffixes=("", "_id1")).drop(columns=["UniProt_ID"])

        features_df_renamed = features_df.add_suffix("_id2").rename(columns={"UniProt_ID_id2": "UniProt_ID"})
        out = out.merge(features_df_renamed, how="left", left_on="uniprot_id2", right_on="UniProt_ID",
                        suffixes=("", "_id2")).drop(columns=["UniProt_ID", "uniprot_id1", "uniprot_id2"])
        
        out.fillna(0, inplace=True) # fill missing values (na, None etc..) with zero
        return out

    def __len__(self):
        return len(self.data)

    def _get_inputs_y_meta(self, idx):
        """Shared sample construction. Returns tensors + meta tuple."""
        smiles = self.data.loc[idx, "smiles"]
        uniprot1 = self.data.loc[idx, "uniprot_id1"]
        uniprot2 = self.data.loc[idx, "uniprot_id2"]
        ppi_id = self.data.loc[idx, "ppi_id"] # get PPI ID in order to extract structure embedding 
        meta = (smiles, uniprot1, uniprot2, ppi_id)
        y = torch.tensor(self.data.loc[idx, "label"], dtype=torch.float32)

        # --------- PPI features --------- #
        # strcutre embeddings
        ppi_conformer_id = '1' # w/o data augmentation, take first PPI conformation
        if self.aug:
            # get PPI conformer ID from conformation pool (5 conformers for each PPI)
            ppi_conformer_id = random.choice(list(self.ppi_struct_emb[self.ppi_stuct_strategy].keys())) 

        # ['mean_ifeature_omega_a', 'mean_ifeature_omega_b', 'mean_ppi_former_a', 'mean_ppi_former_b', 'progres_vector']
        ppi_former_a_features = torch.from_numpy(self.ppi_struct_emb[self.ppi_stuct_strategy][ppi_conformer_id][ppi_id]['mean_ppi_former_a'][:]).float()
        ppi_former_b_features = torch.from_numpy(self.ppi_struct_emb[self.ppi_stuct_strategy][ppi_conformer_id][ppi_id]['mean_ppi_former_b'][:]).float()

        #ean_ifeatures_omega_a = torch.from_numpy(self.ppi_struct_emb[self.ppi_stuct_strategy][ppi_conformer_id][ppi_id]['mean_ifeatures_omega_a'][:]).float()
        #mean_ifeatures_omega_b = torch.from_numpy(self.ppi_struct_emb[self.ppi_stuct_strategy][ppi_conformer_id][ppi_id]['mean_ifeatures_omega_b'][:]).float()


        esm_features  = torch.from_numpy(self.esm_features_ppi[idx])
        fegs_features = torch.from_numpy(self.fegs_features_ppi[idx])
        gae_features  = torch.from_numpy(self.gae_features_ppi[idx])

        # SMILES features
        morgan   = torch.from_numpy(self.morgan_map.loc[smiles].to_numpy(np.float32))
        chem_desc= torch.from_numpy(self.desc_map.loc[smiles].to_numpy(np.float32))
        chemprop = torch.from_numpy(self.chemprop_map.loc[smiles].to_numpy(np.float32))
        chemberta= torch.from_numpy(self.chemberta_map.loc[smiles].to_numpy(np.float32))

        inputs = {
            "ppi_former_a": ppi_former_a_features,
            "ppi_former_b": ppi_former_b_features,
            "cpe": chemprop,
            "esm": esm_features,
            "fegs": fegs_features,
            "gae": gae_features,
            "cbae": chemberta,
            "morgan_fingerprints": morgan,
            "chemical_descriptors": chem_desc,
        }

        return inputs, y, meta


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
        return stacked_inputs, y, (smiles, uniprot1, uniprot2)

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










#####################################################333
class MoleculeDataset__(Dataset):
    def __init__(self, ds_):
        logger.info('Initializing MoleculeDataset !')
        self.data = ds_ # cols -> smiles, uniprot1, uniprot2, label
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
        self.esm = pd.read_csv(os.path.join('datasets', 'esm_features.csv'))
        self.fegs = pd.read_csv(os.path.join('datasets', 'fegs_features.csv'))
        self.gae = pd.read_csv(os.path.join('datasets', 'gae_features.csv'))
        
        self.gae.loc[self.gae['predicted'] == 1, self.gae.columns[9:509]] = 0
        gae_features_columns = self.gae.iloc[:, 9:509]

        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)
        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)

         # SMILES RDKit features - Morgan Fingerprints (r=4, nbits=1024)  chemical descriptors, chemprop & chemBERTa
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join('datasets', 'smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join('datasets', 'smiles_chem_descriptors_mapping_dataset.csv'))
        self.chemprop = pd.read_csv(os.path.join('datasets', 'chemprop_features.csv'))
        self.chemberta = pd.read_csv(os.path.join('datasets', 'chemBERTa_features.csv'))

    def merge_datasets(self, dataset, features_df):
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
        dataset = dataset.drop(columns=['UniProt_ID'])
        
        features_df_renamed = features_df.add_suffix('_id2')
        features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
        dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
        
        # In order to avoid dropping duplicated rows that holds only zeros (in gae when there is zero vectors), which can be represents embeddings of ppi vector when
        # specifying to reset the rows to hold only zeros
        dataset['zero_count'] = (dataset == 0).any(axis=1).astype(int)
        count = 1
        for index in dataset.index:
            if dataset.at[index, 'zero_count'] == 1:
                dataset.at[index, 'zero_count'] = count
                count += 1
                
        # Fill null values with 0
        dataset.fillna(0, inplace=True)

        # Adam: make sure when you train_test on cold start ppi to uncomment that:
        #dataset.drop_duplicates(inplace=True)

        return dataset.drop(columns=['zero_count'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, -1], dtype=np.float32)  
        esm_features = np.array(self.esm_features_ppi.iloc[idx].values, dtype=np.float32)
        fegs_features = np.array(self.fegs_features_ppi.iloc[idx].values, dtype=np.float32)
        gae_features = np.array(self.gae_features_ppi.iloc[idx].values, dtype=np.float32)

        # Retrieve precomputed RDKit Morgan fingerprints
        morgan_fingerprint = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemical_descriptors = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemprop_features = self.chemprop.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemberta_features = self.chemberta.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        
        return (chemprop_features, esm_features, fegs_features, gae_features, 
                chemberta_features, morgan_fingerprint, chemical_descriptors, label)


# 27/12 - dataset for MC dropout evaluation
class MCDMoleculeDataset(Dataset):
    def __init__(self, ds_):
        logger.info('Initialized MoleculeDataset -> For MC Dropout evaluation')
        self.data = ds_
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
        self.esm = pd.read_csv(os.path.join('datasets', 'esm_features.csv'))
        self.fegs = pd.read_csv(os.path.join('datasets', 'fegs_features.csv'))
        self.gae = pd.read_csv(os.path.join('datasets', 'gae_features.csv'))
        
        self.gae.loc[self.gae['predicted'] == 1, self.gae.columns[9:509]] = 0
        gae_features_columns = self.gae.iloc[:, 9:509]

        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)
        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)

         # SMILES RDKit features - Morgan Fingerprints (r=4, nbits=1024)  chemical descriptors, chemprop & chemBERTa
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join('datasets', 'enamine_mcd_smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join('datasets', 'enamine_mcd_smiles_chem_descriptors_mapping_dataset.csv'))
        self.chemprop = pd.read_csv(os.path.join('datasets', 'enamine_chemprop_features_mcd.csv'))
        self.chemberta = pd.read_csv(os.path.join('datasets', 'enamine_chemBERTa_features_mcd.csv'))

    def merge_datasets(self, dataset, features_df):
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
        dataset = dataset.drop(columns=['UniProt_ID'])
        
        features_df_renamed = features_df.add_suffix('_id2')
        features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
        dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
        
        # In order to avoid dropping duplicated rows that holds only zeros (in gae when there is zero vectors), which can be represents embeddings of ppi vector when
        # specifying to reset the rows to hold only zeros
        dataset['zero_count'] = (dataset == 0).any(axis=1).astype(int)
        count = 1
        for index in dataset.index:
            if dataset.at[index, 'zero_count'] == 1:
                dataset.at[index, 'zero_count'] = count
                count += 1
                
        # Fill null values with 0
        dataset.fillna(0, inplace=True)

        # Adam: make sure when you train_test on cold start ppi to uncomment that:
        #dataset.drop_duplicates(inplace=True)

        return dataset.drop(columns=['zero_count'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, -1], dtype=np.float32)  
        esm_features = np.array(self.esm_features_ppi.iloc[idx].values, dtype=np.float32)
        fegs_features = np.array(self.fegs_features_ppi.iloc[idx].values, dtype=np.float32)
        gae_features = np.array(self.gae_features_ppi.iloc[idx].values, dtype=np.float32)

        # Retrieve precomputed RDKit Morgan fingerprints
        morgan_fingerprint = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemical_descriptors = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemprop_features = self.chemprop.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemberta_features = self.chemberta.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        
        return (chemprop_features, esm_features, fegs_features, gae_features, 
                chemberta_features, morgan_fingerprint, chemical_descriptors, label)


'''
# In case we want to drop duplicates inside
# MolecularDataset(...), we'll create unique_id 
# column for rows that have full-zero-vector,
# in order to not drop them, because they might
# not be duplicated (same smiles, uniprot_id1, uniprot_id2)
# jsut same zero-vecrtor-feature

dataset['zero_count'] = (dataset == 0).any(axis=1).astype(int)
count = 1
for index in dataset.index:
    if dataset.at[index, 'zero_count'] == 1:
        dataset.at[index, 'zero_count'] = count
        count += 1
dataset.drop_duplicates(inplace=True)
'''

