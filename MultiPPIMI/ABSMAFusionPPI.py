import time
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from utils.tools import scaffold_split
from utils.MoleculeDataset import MoleculeDataset



class ABSMAFusionPPI(ABC, nn.Module):
    def __init__(self):
        super(ABSMAFusionPPI, self).__init__()
        self.early_stopping_patience = 5
        self.delta = 0.0001
        self.true_threshold = 0.5 # same as state-of-the-art frameworks
    @abstractmethod
    def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
        pass
        
    def train_model(self, fold, num_epochs, dataset, optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5):
        
        train_dataset = MoleculeDataset(dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        logger.info(f'Start training {fold} for {num_epochs} epochs !')
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()
            train_loss = 0.0
            all_labels, all_outputs = [], []

            for (inputs), y in train_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)

                optimizer.zero_grad()
                outputs = self(*inputs) 
                loss = criterion(outputs.squeeze(), y)
                
                train_loss += loss.item()
                all_labels.extend(y.cpu().numpy())
                all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
        
                loss.backward()
                optimizer.step()
                        
            # Calculate metrics
            train_loss /= len(train_loader)
            train_auc = roc_auc_score(all_labels, all_outputs)
            train_aupr = average_precision_score(all_labels, all_outputs)
            precision = precision_score(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int))
            sensitivity = recall_score(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int))
            
            tn, fp, _, _ = confusion_matrix(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int)).ravel()
            specificity = tn / (tn + fp)
            
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            logger.info(f"Epoch {epoch+1} Time: {epoch_time:.2f} min, Train Loss: {train_loss:.5f}, Train AUC: {train_auc:.5f}, "
                  f"Train AUPR: {train_aupr:.5f}, Precision: {precision:.5f}, Sensitivity: {sensitivity:.5f}, Specificity: {specificity:.5f}")

        
    def train_val_model(self, fold, num_epochs, dataset, optimizer, criterion, custom_splits=False,
                    batch_size=32, device='cuda', num_workers=5):
        best_val_auc = float('-inf')
        no_improve_epochs = 0

        if custom_splits: 
            # splits are given as inputs from cv_cold_neg_smoo(...) method
            train_subset, val_subset = dataset[0], dataset[1]
        else:                
            # split using custim scaffold splitter rather than deepchem version
            train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=0.8, seed=fold)

            train_indices = train_dataset.ids
            val_indices = val_dataset.ids        
            train_subset = dataset[dataset["smiles"].isin(train_indices)].copy()
            val_subset = dataset[dataset["smiles"].isin(val_indices)].copy()

        train_dataset = MoleculeDataset(train_subset)
        val_dataset = MoleculeDataset(val_subset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for epoch in range(num_epochs):
            all_preds = []
            all_labels = []
            start_time = time.time()
            self.train()
            running_loss = 0.0
            for (inputs), y in train_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)

                optimizer.zero_grad()
                outputs = self(*inputs)                 
                loss = criterion(outputs.squeeze(), y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(outputs.squeeze().detach().cpu().numpy())
            
            train_auc = roc_auc_score(all_labels, all_preds)
            _, _, val_auc = self.validate_model(val_loader, criterion, device)
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {(running_loss/len(train_loader)):.4f}, Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}, Epoch Time: {epoch_time:.4f}')
            # Check whether val_auc > best_val_auc + delta
            if val_auc > best_val_auc + self.delta:
                best_val_auc = val_auc
                train_epoch = epoch+1
                no_improve_epochs = 0 
                logger.info(f"Current best val_auc -> {val_auc:.5f}, at epoch {epoch+1}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.early_stopping_patience:
                    logger.info(f"Stopping early at epoch {epoch+1}")
                    break

        logger.info(f'Train the model for -> {train_epoch}, best validation auc: {best_val_auc:.5f}')
        return train_epoch, round(best_val_auc, 5)   

    def test_model(self, test_dataset, criterion, batch_size, device, num_workers):
        test_dataset = MoleculeDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.eval()
    
        test_loss, correct, total = 0.0, 0, 0
        all_labels, all_outputs = [], []
    
        with torch.no_grad():
            for (inputs), y in test_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)
    
                outputs = self(*inputs)
                loss = criterion(outputs.squeeze(), y)
                test_loss += loss.item()
    
                all_labels.extend(y.cpu().numpy())
                all_outputs.extend(outputs.squeeze().cpu().numpy())
    
  
        test_loss /= len(test_loader)
        test_auc = roc_auc_score(all_labels, all_outputs)
        test_aupr = average_precision_score(all_labels, all_outputs)
        precision = precision_score(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int))
        sensitivity = recall_score(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int))
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(all_labels, (np.array(all_outputs) > self.true_threshold).astype(int)).ravel()
        specificity = tn / (tn + fp)
    
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Test AUPR: {test_aupr:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Test Specificity: {specificity:.4f}")

        return round(test_loss, 5), round(test_auc, 5), round(test_aupr, 5), round(precision, 5), round(sensitivity, 5), round(specificity, 5)

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for (inputs), y in val_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)

                outputs = self(*inputs)                
                loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()

                all_labels.extend(y.cpu().numpy())
                all_outputs.extend(outputs.squeeze().cpu().numpy())

                predicted = (outputs.squeeze() > self.true_threshold).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total
        val_auc = roc_auc_score(all_labels, all_outputs)
        return val_loss, accuracy, val_auc