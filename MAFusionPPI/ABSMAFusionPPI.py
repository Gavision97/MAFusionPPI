import os
import time
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score


from utils.eval_tools import EarlyStopping, calc_metrics
from utils.splitters import scaffold_split
from utils.MoleculeDataset import MoleculeDataset

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


class ABSMAFusionPPI(ABC, nn.Module):
    def __init__(self):
        super(ABSMAFusionPPI, self).__init__()
        self.true_threshold = 0.5 # same as state-of-the-art frameworks

        # early stopping hyperparameters
        self.early_stopping_patience = 20
        self.warm_up_epochs = 10
        self.delta = 0.0001

    @abstractmethod
    def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
        pass
        
    def train_model(self, fold, num_epochs, dataset, optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5):
        
        train_dataset = MoleculeDataset(dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        logger.info(f'Start training {fold} for {num_epochs} epochs !')
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()
            train_loss = 0.0
            all_labels, all_outputs = [], []

            for (inputs), y in train_loader:
                inputs = [inp.to(device) for inp in inputs] # move all features & labels to device
                y = y.to(device)

                optimizer.zero_grad()
                outputs = self(*inputs)
                logits, targets = outputs.view(-1), y.view(-1).float()
                loss = criterion(logits, targets)

                train_loss += loss.item()
                all_labels.extend(targets.detach().cpu().numpy())
                all_outputs.extend(logits.detach().cpu().numpy())
                        
                loss.backward()
                optimizer.step()
                        
            train_loss /= len(train_loader)
            train_metric_dict = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)
            
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            logger.info(f"Epoch {epoch+1} Time: {epoch_time:.2f} min, Train Loss: {train_loss:.5f}, Train AUC: {train_metric_dict['AUC']:.5f}, "
                  f"Train AUPR: {train_metric_dict['AUPR']:.5f}, Precision: {train_metric_dict['Precision']:.5f}, Sensitivity: {train_metric_dict['Sensitivity']:.5f}, Specificity: {train_metric_dict['Specificity']:.5f}")

        
    def train_val_model(self, fold, num_epochs, dataset, optimizer, criterion,
                    batch_size=32, device='cuda', num_workers=5):
        
        # we track train & val AUC score in order to plot train vs. auc curves
        train_aucs, val_aucs = [], [] 

        # split using custim scaffold splitter rather than deepchem version
        train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=0.8)

        train_dataset = MoleculeDataset(train_subset)
        val_dataset = MoleculeDataset(val_subset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        logger.info(f'---- train loader size -> {len(train_dataset)}, val loader size {len(val_dataset)} ----')

        # custom early stopping object; in order to reduce risk of overfitting
        # on the validation set and improve model generalization ability
        early_stopping = EarlyStopping(mode='max', patience=self.early_stopping_patience,
                                       warm_up_epochs=self.warm_up_epochs)

        for epoch in range(num_epochs):
            all_preds, all_labels = [], []

            self.train()
            #running_loss = 0.0
            for (inputs), y in train_loader:
                inputs = [inp.to(device) for inp in inputs] # move all features & labels to device
                y = y.to(device)

                optimizer.zero_grad()
                outputs = self(*inputs)
                logits, targets = outputs.view(-1), y.view(-1).float()
                loss = criterion(logits, targets)

                #running_loss += loss.item()
                all_labels.extend(targets.detach().cpu().numpy())
                all_preds.extend(logits.detach().cpu().numpy())

                loss.backward()
                optimizer.step()
            
            train_auc = roc_auc_score(all_labels, all_preds)
            val_metrics_dict, _ = self.validate_model(val_loader, criterion, device)
            curr_val_metrics = [val_metrics_dict['AUC'], val_metrics_dict['AUPR'], val_metrics_dict['Precision'],
                                val_metrics_dict['Sensitivity'], val_metrics_dict['Specificity']]
            
            train_aucs.append(train_auc)
            val_aucs.append(val_metrics_dict['AUC'])
            
            # early stopping step
            early_stopping.check_early_stop(val_metrics_dict['AUC'], curr_val_metrics
                                            ,train_auc, epoch)   
            if early_stopping.stop_training:
                logger.info(f"Early stopping at epoch {epoch}. "
                            f"Best AUC={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch}")
                train_epoch = early_stopping.best_epoch 
                break
        
        # extract the best epoch from early stopping process
        train_epoch = early_stopping.best_epoch 
        logger.info(f"Train the model for -> {train_epoch}, best validation auc: {early_stopping.best_metrics[0]:.5f}")
        
        # return number of epoch to train the model based on CV with scaffold split,
        # best epoch metrics (auc, aupr, etc ..), train aucs and val aucs
        # in order to plot train vs. val auc curves.
        return train_epoch, early_stopping.best_metrics, train_aucs, val_aucs   

    def test_model(self, test_dataset, criterion, batch_size, device, num_workers):

        test_dataset = MoleculeDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        self.eval()
    
        test_loss, all_labels, all_outputs = 0.0, [], []
    
        with torch.no_grad():
            for (inputs), y in test_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)
    
                outputs = self(*inputs)
                logits, targets = outputs.view(-1), y.view(-1).float()
                loss = criterion(logits, targets)

                test_loss += loss.item()
                all_labels.extend(targets.detach().cpu().numpy())
                all_outputs.extend(logits.detach().cpu().numpy())
    
  
        test_loss /= len(test_loader)


        # get test auc, aupr, precision, sensitivity, & specificity values as dictionary
        # that maps each metric name to its corresponding value
        test_metrics_dict = calc_metrics(ys_true=all_labels, ys_pred=all_outputs)

        # log all test results (loss, auc, aupr, etc ..)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test AUC: {test_metrics_dict['AUC']:.4f}")
        logger.info(f"Test AUPR: {test_metrics_dict['AUPR']:.4f}")
        logger.info(f"Test Precision: {test_metrics_dict['Precision']:.4f}")
        logger.info(f"Test Sensitivity (Recall): {test_metrics_dict['Sensitivity']:.4f}")
        logger.info(f"Test Specificity: {test_metrics_dict['Specificity']:.4f}")


        return test_metrics_dict, round(test_loss, 5)

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for (inputs), y in val_loader:
                # Move tensors to the configured device
                inputs = [inp.to(device) for inp in inputs]
                y = y.to(device)

                outputs = self(*inputs)
                logits, targets = outputs.view(-1), y.view(-1).float()
                loss = criterion(logits, targets)

                val_loss += loss.item()
                all_labels.extend(targets.detach().cpu().numpy())
                all_outputs.extend(logits.detach().cpu().numpy())

        val_loss /= len(val_loader)

        # get validation auc, aupr, precision, sensitivity, & specificity values as dictionary
        # that maps each metric name to its corresponding value
        val_metrics = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)
        
        return val_metrics, round(val_loss, 5)