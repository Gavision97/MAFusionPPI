import os
import time
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tools import seed_worker, get_out_dir
from utils.eval_tools import EarlyStopping, calc_metrics
from utils.splitters import scaffold_split
from utils.MoleculeDataset import TrainMoleculeDataset, EvalMoleculeDataset

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"

# TODO -> change 'date' every time we want to choose different sub-directory
RES_TABLES_PATH = 'results/result_tables/1403/'
FRAC_TRAIN = 0.95

# structure hyperparameters
#STRUCT_DATASET = "dataset1"
#STRATEGY = "subset_mean"
#AUG = False

class ABSMAFusionPPI(ABC, nn.Module):
    def __init__(self):
        super(ABSMAFusionPPI, self).__init__()
        self.true_threshold = 0.5 # same as state-of-the-art frameworks

        # early stopping hyperparameters
        self.early_stopping_patience = 15
        self.warm_up_epochs = 10
        self.delta = 0.002

    @abstractmethod
    def forward(self, **inputs):
        pass
    
    def _train_one_epoch(self, train_loader, optimizer, criterion, device):
        self.train()
        train_loss = 0.0
        all_labels, all_outputs = [], []

        for inputs, y in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)

            optimizer.zero_grad()

            outputs = self(**inputs)
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


    def train_model(self, fold, num_epochs, dataset, strct_dataset,
                    strct_strategy, strct_aug_train,optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5, seed=42):
        
        train_dataset = TrainMoleculeDataset(ds_=dataset, struct_dataset=strct_dataset, strategy=strct_strategy, aug_train=strct_aug_train)
        val_dataset_es = EvalMoleculeDataset(ds_=val_subset, use_struct=use_struct, struct_dataset=strct_dataset, strategy=strct_strategy, eval_all_confs=False) 

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, generator=g, worker_init_fn=seed_worker)
        
        logger.info(f'Start training {fold} for {num_epochs} epochs !')
        for epoch in range(num_epochs):
            start_time = time.time()
            train_metric_dict, train_loss, _, _ = self._train_one_epoch(train_loader, optimizer, criterion, device)
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60

            logger.info(f"Epoch {epoch+1} Time: {epoch_time:.2f} min, Train Loss: {train_loss:.5f}, Train AUC: {train_metric_dict['AUC']:.5f}, "
                  f"Train AUPR: {train_metric_dict['AUPR']:.5f}, Precision: {train_metric_dict['Precision']:.5f}, Sensitivity: {train_metric_dict['Sensitivity']:.5f}, Specificity: {train_metric_dict['Specificity']:.5f}")
    

    def heldout_val_model(self, fold, use_struct, num_epochs, dataset, strct_dataset, strct_strategy,
                          strct_aug_train, strct_aug_eval, optimizer, criterion, is_neg_smoo=False, save_probs=False,
                          batch_size=32, device="cuda", num_workers=5, seed=42):
        """
        Heldout-validation using scaffold-splitter and early-stopping.

        Trains with early stopping (score = val AUC), restores the BEST model weights,
        then runs validation ONCE with the best model and saves ALL validation rows to:

        results/results_tables/cold_neg_{neg_factor}_smoo_{smoo_factor}/fold{K}/exp_{exp_num}/val_predictions.csv

        CSV columns:
        [smiles, uniprot1, uniprot2, label, predicted_prob]
        """

        # fold name format -> f'{job_id}_{fold_id}_{neg_factor}_{smoo_factor}_{exp_num}'
        #fold_splitted_name = fold.split("_")
        #job_id, fold_id, neg_factor, smoo_factor, exp_num = (fold_splitted_name[0], fold_splitted_name[1], 
         #                                            fold_splitted_name[2], fold_splitted_name[3], fold_splitted_name[4])
        
        train_aucs, val_aucs = [], []
        # split using custim scaffold splitter rather than deepchem version; set seed for reproducibility
        # between same N experiments (i.e., all experiments i will use seed=i)
        train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=FRAC_TRAIN, seed=seed)

        train_dataset = TrainMoleculeDataset(ds_=train_subset, use_struct=use_struct, struct_dataset=strct_dataset, strategy=strct_strategy, aug_train=strct_aug_train)

        # val_dataset_es -> for early stopping (w/o evaluate all confs), val_dataset -> for evaluating best model  
        # on the validatoin set after early stopping fired (if specified by the user with eval_all_confs=True)
        val_dataset_es = EvalMoleculeDataset(ds_=val_subset, use_struct=use_struct, struct_dataset=strct_dataset, strategy=strct_strategy, eval_all_confs=False) 
        val_dataset = EvalMoleculeDataset(ds_=val_subset, use_struct=use_struct, struct_dataset=strct_dataset, strategy=strct_strategy, eval_all_confs=strct_aug_eval)
 

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, generator=g, worker_init_fn=seed_worker)
        val_loader_es = DataLoader(val_dataset_es, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,  collate_fn=EvalMoleculeDataset.collate_eval)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,  collate_fn=EvalMoleculeDataset.collate_eval)
        logger.info(f'---- train loader size -> {len(train_dataset)}, val loader size {len(val_dataset)} ----')

        early_stopping = EarlyStopping(mode="max", patience=self.early_stopping_patience,
                                       warm_up_epochs=self.warm_up_epochs, delta=self.delta)

        for epoch in range(num_epochs):
            train_metrics_dict, _, _, _ = self._train_one_epoch(train_loader, optimizer, criterion, device)

            # metrics on val (no need to collect per-row each epoch)
            val_metrics_dict, _ = self.validate_model(val_loader_es, criterion, device, return_rows=False)
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
        validate_fn = self.validate_with_aug_model if strct_aug_eval else self.validate_model
        best_val_metrics_dict, _, val_rows = validate_fn(val_loader, criterion, device, return_rows=True)
        

        if save_probs: 
            out_dir, exp_num = get_out_dir(fold_name=fold, is_neg_smoo=is_neg_smoo)
            # save evaluation results in results/results_tables/date/job_id/cold_neg_{i}_smoo_{j}/{fold_k}/exp_{x}/val_exp_{x}.csv
            #date, job_id = job_id.split('@') # ['date', 'job_id']
            #out_dir = os.path.join(RES_TABLES_PATH, date, job_id, f"cold_neg_{neg_factor}_smoo_{smoo_factor}", fold_id, f"exp_{exp_num}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"val_exp_{exp_num}.csv")
            pd.DataFrame(val_rows).to_csv(out_path, index=False)
            logger.info(f'Saved validation probabilities to -> {out_path} successfully')
        # return best_model, best model validation metrics (AUC, AUPR ..), 
        # train & val AUCs for plot comparison
        return self, best_val_metrics_dict, train_aucs, val_aucs
        
    def train_val_model(self, fold, num_epochs, dataset, optimizer, criterion,
                    batch_size=32, device='cuda', num_workers=5, seed=42):
        
        # we track train & val AUC score in order to plot train vs. auc curves
        train_aucs, val_aucs = [], [] 

        # split using custim scaffold splitter rather than deepchem version; set seed for reproducibility
        # between same N experiments (i.e., all experiments i will use seed=i)
        train_subset, val_subset = scaffold_split(dataset, smiles_col="smiles", frac_train=FRAC_TRAIN, seed=seed)

        train_dataset = TrainMoleculeDataset(train_subset)
        val_dataset = EvalMoleculeDataset(val_subset)

        # drop_last=True in order to avoid bug when batch_size=1 in training phase (BatchNorn1d crashes when batch_size=1)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=EvalMoleculeDataset.collate_eval)
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


    def test_model(self, fold, use_struct, dataset, strct_dataset, strct_strategy, criterion,
                   is_neg_smoo=False, batch_size=32, device='cuda', num_workers=0, save_probs=False, eval_all_confs=False):

        test_dataset = EvalMoleculeDataset(ds_=dataset, use_struct=use_struct, struct_dataset=strct_dataset, strategy=strct_strategy, eval_all_confs=eval_all_confs)
        # in evaluation, keep drop_last=False in order to retrieve all rows for tracking & analysing performance
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=0, drop_last=False, collate_fn=EvalMoleculeDataset.collate_eval)
        
        validate_fn = self.validate_with_aug_model if eval_all_confs and use_struct else self.validate_model

        if not save_probs:
            test_metrics_dict, test_loss = validate_fn(test_loader, criterion, device, return_rows=False,)
        else:
            test_metrics_dict, test_loss, test_rows = validate_fn(test_loader, criterion, device, return_rows=True)
            out_dir, exp_num = get_out_dir(fold_name=fold, is_neg_smoo=is_neg_smoo)
            # fold format -> f'{job_id}_{fold_id}_{neg_factor}_{smoo_factor}_{exp_num}'
            #fold_splitted_name = fold.split("_")
            #job_id, fold_id, neg_factor, smoo_factor, exp_num = (fold_splitted_name[0], fold_splitted_name[1], 
            #                                             fold_splitted_name[2], fold_splitted_name[3], fold_splitted_name[4])
            
            # save evaluation results in results/date/job_id/results_tables/cold_neg_{i}_smoo_{j}/{fold_k}/exp_{x}/test_exp_{x}.csv
            #date, job_id = job_id.split("@") # ['date', 'job_id']
            #out_dir = os.path.join(RES_TABLES_PATH, date, job_id, f"cold_neg_{neg_factor}_smoo_{smoo_factor}", fold_id, f"exp_{exp_num}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"test_exp_{exp_num}.csv")
            pd.DataFrame(test_rows).to_csv(out_path, index=False)
            logger.info(f'Saved test probabilities to -> {out_path} successfully')

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
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)

            outputs = self(**inputs)
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
                smiles, uniprot1, uniprot2, ppi_id = meta
                for s, u1, u2, ppiid, lab, pr in zip(smiles, uniprot1, uniprot2, ppi_id, labels_np, probs_np):
                    val_rows.append({"smiles": s, "uniprot1": u1, "uniprot2": u2, "ppi_id":ppiid, "label": float(lab), "predicted_prob": float(pr)})

        val_loss /= max(1, len(val_loader))
        val_metrics = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)

        if return_rows:
            return val_metrics, round(val_loss, 5), val_rows
        return val_metrics, round(val_loss, 5)

    # only while using structure feature
    @torch.no_grad()
    def validate_with_aug_model(self, val_loader, criterion, device, return_rows=False):
        self.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []
        val_rows = [] if return_rows else None

        for inputs, y, meta in val_loader:
            y = y.to(device)

            # move fixed features
            fixed_inputs = {}
            for k, v in inputs.items():
                fixed_inputs[k] = v.to(device)

            ppi_former_all = fixed_inputs["ppi_former"]       
            ppi_omega_all = fixed_inputs["ppi_omega"]         
            ppi_progress_all = fixed_inputs["ppi_progress_vec"] 

            conf_preds = []

            for conf_idx in range(ppi_former_all.size(1)): # iterate over all PPI conformations (AF3 default=5)
                curr_inputs = {
                    "ppi_former": ppi_former_all[:, conf_idx, :],          
                    "ppi_omega": ppi_omega_all[:, conf_idx, :], 
                    "ppi_progress_vec": ppi_progress_all[:, conf_idx, :],          
                    "cpe": fixed_inputs["cpe"],
                    "esm": fixed_inputs["esm"],
                    "fegs": fixed_inputs["fegs"],
                    "gae": fixed_inputs["gae"],
                    "cbae": fixed_inputs["cbae"],
                    "morgan_fingerprints": fixed_inputs["morgan_fingerprints"],
                    "chemical_descriptors": fixed_inputs["chemical_descriptors"],
                }

                outputs = self(**curr_inputs)
                logits = outputs.view(-1)   # (B,)
                conf_preds.append(logits)

            # shape: (num_confs, B) -> mean over conformations
            mean_logits = torch.stack(conf_preds, dim=0).mean(dim=0)

            targets = y.view(-1).float()
            loss = criterion(mean_logits, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(mean_logits)
            labels_np = targets.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_labels.extend(labels_np)
            all_outputs.extend(probs_np)

            if return_rows:
                smiles, uniprot1, uniprot2, ppi_id = meta
                for s, u1, u2, ppiid, lab, pr in zip(smiles, uniprot1, uniprot2, ppi_id, labels_np, probs_np):
                    val_rows.append({"smiles": s, "uniprot1": u1, "uniprot2": u2, "ppi_id": ppiid, "label": float(lab), "predicted_prob": float(pr)})

        val_loss /= max(1, len(val_loader))
        val_metrics = calc_metrics(ys_true=all_labels, ys_pred=all_outputs, true_threshold=self.true_threshold)
        if return_rows:
            return val_metrics, round(val_loss, 5), val_rows
        return val_metrics, round(val_loss, 5)