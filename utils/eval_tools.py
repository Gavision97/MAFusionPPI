import os
import logging
logger = logging.getLogger(__name__) # get logger name

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


def calc_metrics(ys_true, ys_pred, true_threshold=0.5):
    metric_dict = {'AUC': None, 'AUPR': None, 'Precision': None, 'Sensitivity': None, 'Specificity': None}
    metric_dict['AUC'] = roc_auc_score(ys_true, ys_pred)
    metric_dict['AUPR'] = average_precision_score(ys_true, ys_pred)
    metric_dict['Precision'] = precision_score(ys_true, (np.array(ys_pred) > true_threshold).astype(int))
    metric_dict['Sensitivity'] = recall_score(ys_true, (np.array(ys_pred) > true_threshold).astype(int))
    
    tn, fp, _, _ = confusion_matrix(ys_true, (np.array(ys_pred) > true_threshold).astype(int)).ravel()
    metric_dict['Specificity'] = tn / (tn + fp)

    return metric_dict


class EarlyStopping:
    '''
    custom early stopping implementation (for that research, hardcoded with score setting => AUC)
    '''
    def __init__(self, mode="min", patience=20, warm_up_epochs=30, delta=0.0, verbose=False):
        """
        mode: "min" for RMSE/loss, "max" for Pearson/AUC
        """
        self.mode = mode
        self.patience = patience
        self.warm_up_epochs = warm_up_epochs
        self.delta = delta
        self.verbose = verbose

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = -1
        self.no_improvement = 0
        self.stop_training = False
        
        # auc, aurp, precision, sensitivity, specificity 
        self.best_metrics = [float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")] 

    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.delta
        else:  # "max"
            return score > self.best_score + self.delta

    def check_early_stop(self, score, metrics, train_score, epoch):
        if self._is_improvement(score):
            self.best_score = score
            # store best metrics (auc, aupr, precision, sensitivity & specificity)
            self.best_metrics[0], self.best_metrics[1], self.best_metrics[2], self.best_metrics[3], self.best_metrics[4] = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]
            self.best_epoch = epoch
            self.no_improvement = 0
            logger.info(f"NEW BEST @ epoch {epoch+1}: train_auc={train_score:.4f}, val_auc={self.best_score:.4f}, "
                    f"val_aupr={self.best_metrics[1]:.4f}, val_precision={self.best_metrics[2]:.4f}, val_sensitivity={self.best_metrics[3]:.4f}; val_specificity={self.best_metrics[4]:.4f}")
        else:
            if epoch < self.warm_up_epochs:
                return
            self.no_improvement += 1
            if self.no_improvement >= self.patience:
                self.stop_training = True
                if self.verbose:
                    logger.info(f"Early stop at {epoch+1}. Best AUC={self.best_score} @ {self.best_epoch+1}")


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df, smiles_col="smiles", frac_train=0.8, seed=42):
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

