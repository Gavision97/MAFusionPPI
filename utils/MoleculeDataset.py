import os
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PPI_EMB_PATH = 'embeddings/ppi/'
SMI_EMB_PATH = 'embeddings/smi/'

class MoleculeDataset(Dataset):
    """
    Returns:
      inputs (list of tensors), y (tensor)

    inputs = [
        chemprop_features,      # (F1,)
        esm_features_ppi,       # (F2,)
        fegs_features_ppi,      # (F3,)
        gae_features_ppi,       # (F4,)
        chemberta_features,     # (F5,)
        morgan_fingerprint,     # (1024,)
        chemical_descriptors    # (194,)
    ]
    y = label (float32)
    """
    def __init__(self, ds_):
        logger.info('Initializing MoleculeDataset !')
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

    def merge_datasets(self, dataset, features_df):
        dataset = dataset.merge(
            features_df, how='left',
            left_on='uniprot_id1', right_on='UniProt_ID',
            suffixes=('', '_id1')
        ).drop(columns=['UniProt_ID'])

        features_df_renamed = features_df.add_suffix('_id2').rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(
            features_df_renamed, how='left',
            left_on='uniprot_id2', right_on='UniProt_ID',
            suffixes=('', '_id2')
        ).drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])

        # keep your "zero_count" trick
        dataset['zero_count'] = (dataset == 0).any(axis=1).astype(int)
        count = 1
        for index in dataset.index:
            if dataset.at[index, 'zero_count'] == 1:
                dataset.at[index, 'zero_count'] = count
                count += 1

        dataset.fillna(0, inplace=True)
        return dataset.drop(columns=['zero_count'])

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of (inputs, y)
        inputs is list of tensors. Stack each position across batch.
        """
        inputs_list, ys = zip(*batch)          # tuple of lists, tuple of scalars/tensors
        # transpose: from list-per-sample to list-per-feature
        features_by_type = list(zip(*inputs_list))

        stacked_inputs = [torch.stack(feats, dim=0) for feats in features_by_type]
        y = torch.stack(ys, dim=0)             # (B,)
        return stacked_inputs, y

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
        smiles = self.data.loc[idx, "smiles"]
        y = torch.tensor(self.data.loc[idx, "label"], dtype=torch.float32)

        # PPI features (already aligned by idx because you built *_features_ppi using self.data order)
        esm_features = torch.from_numpy(self.esm_features_ppi.iloc[idx].values.astype(np.float32))
        fegs_features = torch.from_numpy(self.fegs_features_ppi.iloc[idx].values.astype(np.float32))
        gae_features = torch.from_numpy(self.gae_features_ppi.iloc[idx].values.astype(np.float32))

        morgan = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chem_desc = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemprop = self.chemprop.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemberta = self.chemberta.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)

        inputs = [chemprop, esm_features, fegs_features, gae_features, chemberta, morgan, chem_desc]
        return inputs, y


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
