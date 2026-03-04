import os
import time
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import pandas as pd
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

RES_TABLES_PATH = 'results/result_tables'

class ABSMAFusionPPI(ABC, nn.Module):
    def __init__(self):
        super(ABSMAFusionPPI, self).__init__()
        self.true_threshold = 0.5 # same as state-of-the-art frameworks

        # early stopping hyperparameters
        self.early_stopping_patience = 10
        self.warm_up_epochs = 10
        self.delta = 0.003

    @abstractmethod
    def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
        pass
    
    
    def _train_one_epoch(self, train_loader, optimizer, criterion, device):

        self.train()
        train_loss = 0.0
        all_labels, all_outputs = [], []

        for inputs, y, _ in train_loader:
            inputs = [inp.to(device) for inp in inputs]
            y = y.to(device)

            optimizer.zero_grad()

            outputs = self(*inputs)
            logits = outputs.view(-1)
            targets = y.view(-1).float()

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.extend(targets.detach().cpu().numpy())
            all_outputs.extend(probs.detach().cpu().numpy())

        avg_loss = train_loss / max(1, len(train_loader))
        metrics = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)
        
        return metrics, round(avg_loss, 5), all_labels, all_outputs


    def train_model(self, fold, num_epochs, dataset, optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5):
        
        train_dataset = MoleculeDataset(dataset)

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=MoleculeDataset.collate_fn)
        
        logger.info(f'Start training {fold} for {num_epochs} epochs !')
        for epoch in range(num_epochs):
            start_time = time.time()
            train_metric_dict, train_loss, _, _ = self._train_one_epoch(train_loader, optimizer, criterion, device)
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60

            logger.info(f"Epoch {epoch+1} Time: {epoch_time:.2f} min, Train Loss: {train_loss:.5f}, Train AUC: {train_metric_dict['AUC']:.5f}, "
                  f"Train AUPR: {train_metric_dict['AUPR']:.5f}, Precision: {train_metric_dict['Precision']:.5f}, Sensitivity: {train_metric_dict['Sensitivity']:.5f}, Specificity: {train_metric_dict['Specificity']:.5f}")
    

    def heldout_val_model(self, fold, num_epochs, dataset, optimizer, 
                          criterion, batch_size=32, device="cuda", num_workers=5, seed=42):
        """
        Heldout-validation using scaffold-splitter and early-stopping.

        Trains with early stopping (score = val AUC), restores the BEST model weights,
        then runs validation ONCE with the best model and saves ALL validation rows to:

        results/results_tables/cold_neg_{neg_factor}_smoo_{smoo_factor}/fold{K}/exp_{exp_num}/val_predictions.csv

        CSV columns:
        [smiles, uniprot1, uniprot2, label, predicted_prob]
        """

        # fold format -> f'{fold_id}_{neg_factor}_{smoo_factor}_{exp_num}'
        fold_splitted_name = fold.split("_")
        fold_id, neg_factor, smoo_factor, exp_num = (fold_splitted_name[0], fold_splitted_name[1], 
                                                     fold_splitted_name[2], fold_splitted_name[3])

        train_aucs, val_aucs = [], []
        # split using custim scaffold splitter rather than deepchem version; set seed for reproducibility
        # between same N experiments (i.e., all experiments i will use seed=i)
        train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=0.05, seed=seed)

        train_dataset = MoleculeDataset(train_subset)
        val_dataset = MoleculeDataset(val_subset)

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=MoleculeDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,  collate_fn=MoleculeDataset.collate_fn)

        early_stopping = EarlyStopping(mode="max", patience=self.early_stopping_patience,
                                       warm_up_epochs=self.warm_up_epochs, delta=self.delta)

        for epoch in range(num_epochs):
            train_metrics_dict, _, _, _ = self._train_one_epoch(train_loader, optimizer, criterion, device)

            # metrics on val (no need to collect per-row each epoch)
            val_metrics_dict, _ = self.validate_model(val_loader, criterion, device, return_rows=False)
            curr_val_metrics = [val_metrics_dict["AUC"], val_metrics_dict["AUPR"], val_metrics_dict["Precision"],
                                val_metrics_dict["Sensitivity"], val_metrics_dict["Specificity"]]

            # track train & val AUC in order to plot them for performance analysis over time (i.e., epochs)
            train_aucs.append(train_metrics_dict["AUC"])
            val_aucs.append(val_metrics_dict["AUC"])

            # store best model weights inside early_stopping
            early_stopping.check_early_stop(score=val_metrics_dict["AUC"], metrics=curr_val_metrics, 
                                            train_score=train_metrics_dict["AUC"], epoch=epoch, model=self)

            if early_stopping.stop_training:
                break

        # restore best weights
        if getattr(early_stopping, "best_state_dict", None) is not None:
            self.load_state_dict(early_stopping.best_state_dict)

        # evaluate best model on the validation set in order to retrive validation results
        # and save all validation predictions in pd.DataFrame() in the next strcuture:
        # [smiles, uniprot_id1, uniprot_id2, label, predicted_prob]
        best_val_metrics_dict, _, val_rows = self.validate_model(val_loader, criterion, device, return_rows=True)
        
        # save evaluation results in results/results_tables/cold_neg_{i}_smoo_{j}/{fold_k}/exp_{x}/val_exp_{x}.csv
        out_dir = os.path.join(RES_TABLES_PATH, f"cold_neg_{neg_factor}_smoo_{smoo_factor}", fold_id, f"exp_{exp_num}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"val_exp_{exp_num}.csv")
        pd.DataFrame(val_rows).to_csv(out_path, index=False)

        # return best_model, best model validation metrics (AUC, AUPR ..), 
        # train & val AUCs for plot comparison
        return self, best_val_metrics_dict, train_aucs, val_aucs
        
    def train_val_model(self, fold, num_epochs, dataset, optimizer, criterion,
                    batch_size=32, device='cuda', num_workers=5, seed=42):
        
        # we track train & val AUC score in order to plot train vs. auc curves
        train_aucs, val_aucs = [], [] 

        # split using custim scaffold splitter rather than deepchem version; set seed for reproducibility
        # between same N experiments (i.e., all experiments i will use seed=i)
        train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=0.8, seed=seed)

        train_dataset = MoleculeDataset(train_subset)
        val_dataset = MoleculeDataset(val_subset)

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=MoleculeDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=MoleculeDataset.collate_fn)
        logger.info(f'---- train loader size -> {len(train_dataset)}, val loader size {len(val_dataset)} ----')

        # custom early stopping object; in order to reduce risk of overfitting
        # on the validation set and improve model generalization ability
        early_stopping = EarlyStopping(mode='max', patience=self.early_stopping_patience,
                                       warm_up_epochs=self.warm_up_epochs, delta=self.delta)

        for epoch in range(num_epochs):
            train_metrics_dict, _, _, _ = self._train_one_epoch(train_loader, optimizer, criterion, device)            


            val_metrics_dict, _ = self.validate_model(val_loader, criterion, device)
            curr_val_metrics = [val_metrics_dict['AUC'], val_metrics_dict['AUPR'], val_metrics_dict['Precision'],
                                val_metrics_dict['Sensitivity'], val_metrics_dict['Specificity']]
            

            # track train & val AUC in order to plot them for performance analysis over time (i.e., epochs)
            train_aucs.append(train_metrics_dict['AUC'])
            val_aucs.append(val_metrics_dict['AUC'])
            
            # early stopping step
            early_stopping.check_early_stop(val_metrics_dict['AUC'], curr_val_metrics
                                            ,train_metrics_dict['AUC'], epoch)   
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


    def test_model(self, fold, dataset, criterion, batch_size, device, num_workers, save=False):

        test_dataset = MoleculeDataset(dataset)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, drop_last=False, collate_fn=MoleculeDataset.collate_fn)

        if not save:
            test_metrics_dict, test_loss = self.validate_model(test_loader, criterion, device, return_rows=False)
        else:
            test_metrics_dict, test_loss, test_rows = self.validate_model(test_loader, criterion, device, return_rows=True)
            
            # fold format -> f'{fold_id}_{neg_factor}_{smoo_factor}_{exp_num}'
            fold_splitted_name = fold.split("_")
            fold_id, neg_factor, smoo_factor, exp_num = (fold_splitted_name[0], fold_splitted_name[1], 
                                                         fold_splitted_name[2], fold_splitted_name[3])
            
            # save evaluation results in results/results_tables/cold_neg_{i}_smoo_{j}/{fold_k}/exp_{x}/test_exp_{x}.csv
            out_dir = os.path.join(RES_TABLES_PATH, f"cold_neg_{neg_factor}_smoo_{smoo_factor}", fold_id, f"exp_{exp_num}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"test_exp_{exp_num}.csv")
            pd.DataFrame(test_rows).to_csv(out_path, index=False)


        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test AUC: {test_metrics_dict['AUC']:.4f}")
        logger.info(f"Test AUPR: {test_metrics_dict['AUPR']:.4f}")
        logger.info(f"Test Precision: {test_metrics_dict['Precision']:.4f}")
        logger.info(f"Test Sensitivity (Recall): {test_metrics_dict['Sensitivity']:.4f}")
        logger.info(f"Test Specificity: {test_metrics_dict['Specificity']:.4f}")

        return test_metrics_dict, test_loss

    @torch.no_grad()
    def validate_model(self, val_loader, criterion, device, return_rows=False):
        self.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []
        val_rows = [] if return_rows else None

        for inputs, y, meta in val_loader:
            inputs = [inp.to(device) for inp in inputs]
            y = y.to(device)

            outputs = self(*inputs)
            logits = outputs.view(-1)
            targets = y.view(-1).float()

            loss = criterion(logits, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            labels_np = targets.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_labels.extend(labels_np)
            all_outputs.extend(probs_np)

            if return_rows:
                smiles, uniprot1, uniprot2 = meta
                for s, u1, u2, lab, pr in zip(smiles, uniprot1, uniprot2, labels_np, probs_np):
                    val_rows.append({"smiles": s, "uniprot1": u1, "uniprot2": u2, "label": float(lab), "predicted_prob": float(pr)})

        val_loss /= max(1, len(val_loader))
        val_metrics = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)

        if return_rows:
            return val_metrics, round(val_loss, 5), val_rows
        return val_metrics, round(val_loss, 5)