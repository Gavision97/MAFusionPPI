import argparse
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import json
import pickle
import itertools
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

from utils.parser import get_args
from evaluate.train_test_cold_start_ppi import train_test_cold_start_ppi
from evaluate.cold_neg_smoo_eval import cold_neg_smoo_eval
from evaluate.cv_cold_eval import cv_cold_eval
from evaluate.mcd_eval import mcd_eval

PPI_DICT_PATH = 'saved_obj/ppi_dict.pkl'
PPI_DFS_DICT_PATH = "saved_obj/enamine_ppi_dfs_dict.pkl"

#DATA_PATH = 'datasets/cold_both_folds'
#DATA_PATH = 'datasets/multi_ppimi_cold_both'
DATA_PATH = 'datasets/multi_ppimi_s4_tests_splits_with_my_train'



def hyperparam_search(
    res_file_name="hyperparam_search_result",
    save_probs=False,
    use_struct=True,
    strct_dataset="dataset1",
    strct_strategy="conditional",
    strct_aug_train=False,
    strct_aug_eval=False,
    job_id="date@j1",
    n=1,
    device="cuda",
    model_kwargs=None,
    training_kwargs=None,
    hyperparam_kwargs=None
) -> None:
    """
    Run hyperparameter search for one job.

    Each job receives a dictionary whose values are lists of candidate values.
    The function expands that dictionary into concrete hyperparameter combinations,
    runs one CV evaluation per combination, and appends the results to a CSV file.

    Notes
    -----
    - Each combination triggers one call to `cv_cold_eval`.
    - Each `cv_cold_eval` internally performs one evaluation over 5 folds.
    - `hyperparam_kwargs` is expected to already represent only the subset
      assigned to the current job.
    """

    if model_kwargs is None:
        raise ValueError("model_kwargs must not be None")
    if training_kwargs is None:
        raise ValueError("training_kwargs must not be None")
    if hyperparam_kwargs is None:
        raise ValueError("hyperparam_kwargs must not be None")

    keys = list(hyperparam_kwargs.keys())
    values = [hyperparam_kwargs[k] for k in keys]

    hyperparam_sets = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    logger.info(f"Job {job_id} started with {len(hyperparam_sets)} hyperparameter combinations")

    res_path = os.path.join("results", "result_tables", "final_results")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")
    print(f'HYPERPARAM SEARCH - BEFORE LOOP')
    for exp_idx, curr_set in enumerate(hyperparam_sets, start=1):
        print(f'HYPERPARAM SEARCH - INSIDE THE LOOP')

        current_model_kwargs = {
            "exclude_modalities": model_kwargs["exclude_modalities"],
            "mlp_dropout": curr_set["mlp_dropout"],
            "head_dropout": curr_set["head_dropout"],
            "self_attn_dropout": curr_set["self_attn_dropout"],
            "join_attn_feat": model_kwargs["join_attn_feat"],
            "compound_dim": model_kwargs["compound_dim"],
            "compound_proj_dim": curr_set["compound_proj_dim"],
            "ppi_fuse_setting": curr_set["ppi_fuse_setting"],
            "head_fuse": curr_set["head_fuse"],
            "proj_feat": curr_set["proj_feat"],
        }

        current_train_kwargs = {
            "lr": curr_set["lr"],
            "weight_decay": curr_set["weight_decay"],
            "batch_size": training_kwargs["batch_size"]
        }

        logger.info(
            f"[{job_id}] Running hyperparameter set {exp_idx}/{len(hyperparam_sets)} | "
            f"lr={curr_set['lr']} | "
            f"weight_decay={curr_set['weight_decay']} | "
            f"mlp_dropout={curr_set['mlp_dropout']} | "
            f"head_dropout={curr_set['head_dropout']} | "
            f"self_attn_dropout={curr_set['self_attn_dropout']} | "
            f"compound_proj_dim={curr_set['compound_proj_dim']} | "
            f"ppi_fuse_setting={curr_set['ppi_fuse_setting']} | "
            f"head_fuse={curr_set['head_fuse']} | "
            f"proj_feat={curr_set['proj_feat']}"
        )

        res_dict, val_metrics_dict = cv_cold_eval(
            use_struct=use_struct,
            save_probs=save_probs,
            strct_dataset=strct_dataset,
            strct_strategy=strct_strategy,
            strct_aug_train=strct_aug_train,
            strct_aug_eval=strct_aug_eval,
            n=n,
            job_id=job_id,
            device=device,
            model_kwargs=current_model_kwargs,
            train_kwargs=current_train_kwargs
        )

        hyperparameters_str = json.dumps(curr_set)

        fold_keys = [f"fold{i}" for i in range(1, 6)]
        row = {
            "job_id": job_id,
            "exp_in_job": exp_idx,
            "hyperparameters": hyperparameters_str,
            "val_res": json.dumps(val_metrics_dict or {}, default=list),
        }

        for fk in fold_keys:
            row[fk] = json.dumps(res_dict.get(fk, []), default=list)

        summary_df = pd.DataFrame([row])

        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
        else:
            updated_df = summary_df

        updated_df.to_csv(summary_path, index=False)
        logger.info(f"Saved/updated summary to: {summary_path}")

def cv_cold_eval_(
    res_file_name="cv_cold_results",
    save_probs=False,
    use_struct=True,
    strct_dataset="dataset1",
    strct_strategy="conditional",
    strct_aug_train=False,
    strct_aug_eval=False,
    job_id="date@j1",
    n=10,
    device="cuda",
    model_kwargs=None,
) -> None:
    """
    Run cross-validation cold evaluation and append one summary row
    to results/result_tables/final_results/<res_file_name>.csv.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    exp_name = job_id
    logger.info(f"--- Start Cold Start Experiments for Dataset -> {DATA_PATH} ---")

    if use_struct:
        logger.info(
            f"Job hyperparameters -> "
            f"use_struct={use_struct}, "
            f"strct_dataset={strct_dataset}, "
            f"strct_strategy={strct_strategy}, "
            f"strct_aug_train={strct_aug_train}, "
            f"strct_aug_eval={strct_aug_eval}"
        )

    logger.info(f"Model kwargs -> {model_kwargs}")

    res_dict, val_metrics_dict = cv_cold_eval(
        use_struct=use_struct,
        save_probs=save_probs,
        strct_dataset=strct_dataset,
        strct_strategy=strct_strategy,
        strct_aug_train=strct_aug_train,
        strct_aug_eval=strct_aug_eval,
        n=n,
        job_id=job_id,
        device=device,
        model_kwargs=model_kwargs,
    )

    fold_keys = [f"fold{i}" for i in range(1, 6)]
    row = {
        "exp": exp_name,
        "val_res": json.dumps(val_metrics_dict or {}, default=list),
    }

    for fk in fold_keys:
        row[fk] = json.dumps(res_dict.get(fk, []))

    summary_df = pd.DataFrame([row], columns=["exp", "val_res"] + fold_keys)

    res_path = os.path.join("results", "result_tables", "final_results")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)
    logger.info(f"Saved/updated summary to: {summary_path}")


def cold_neg_smoo_eval_(
    res_file_name="cv_cold_neg_smoo_results",
    save_probs=False,
    use_struct=True,
    neg_factor="1",
    smoo_factor="1",
    strct_dataset="dataset1",
    strct_strategy="conditional",
    strct_aug_train=False,
    strct_aug_eval=False,
    job_id="date@j1",
    n=10,
    device="cuda",
    model_kwargs=None,
) -> None:
    """
    Run cold negative-sampling / smoothing evaluation and append one
    summary row to results/result_tables/final_results/<res_file_name>.csv.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    exp_name = f"dataset_neg_fct_{neg_factor}_smoo_fct_{smoo_factor}_{job_id}"
    logger.info(f"--- Start Cold Start Experiments for Dataset -> {exp_name} ---")
    logger.info(
        f"Job hyperparameters -> "
        f"strct_dataset={strct_dataset}, "
        f"strct_strategy={strct_strategy}, "
        f"strct_aug_train={strct_aug_train}, "
        f"strct_aug_eval={strct_aug_eval}"
    )
    logger.info(f"Model kwargs -> {model_kwargs}")

    res_dict, val_metrics_dict = cold_neg_smoo_eval(
        use_struct=use_struct,
        save_probs=save_probs,
        neg_factor=neg_factor,
        smoo_factor=smoo_factor,
        strct_dataset=strct_dataset,
        strct_strategy=strct_strategy,
        strct_aug_train=strct_aug_train,
        strct_aug_eval=strct_aug_eval,
        n=n,
        job_id=job_id,
        device=device,
        model_kwargs=model_kwargs,
    )

    fold_keys = [f"fold{i}" for i in range(1, 6)]
    row = {
        "exp": exp_name,
        "val_res": json.dumps(val_metrics_dict or {}, default=list),
    }

    for fk in fold_keys:
        row[fk] = json.dumps(res_dict.get(fk, []))

    summary_df = pd.DataFrame([row], columns=["exp", "val_res"] + fold_keys)

    res_path = os.path.join("results", "result_tables", "final_results")
    os.makedirs(res_path, exist_ok=True)
    summary_path = os.path.join(res_path, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)
    logger.info(f"Saved/updated summary to: {summary_path}")


def mcd_eval_(
    exp_name="Enamine",
    smiles_column="smiles",
    model_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs

    with open(PPI_DICT_PATH, "rb") as f:
        ppi_partiton_dict = pickle.load(f)

    with open(PPI_DFS_DICT_PATH, "rb") as f:
        ppi_dict = pickle.load(f)

    logger.info(f"PPI pair set length -> {len(list(ppi_dict.keys()))}")
    logger.info(f"Model kwargs -> {model_kwargs}")

    df = pd.read_csv(f"datasets/mcd/{exp_name}.csv")
    smiles_df = df[[smiles_column]]
    logging.info(f"number of smiles -> {smiles_df.shape[0]}")

    output_directory = "./mc_dropout_results_enamine"

    mcd_eval(
        ppi_partition_number=1,
        ppi_dict=ppi_dict,
        ppi_partiton_dict=ppi_partiton_dict,
        smiles_df=smiles_df,
        output_dir=output_directory,
        model_kwargs=model_kwargs,
    )


def main():

    # get all arguments
    args = get_args()


    # general arguments -
    log_msg = args.log_msg
    res_file_name = args.res_file_name
    log_file_name = args.log_file_name
    eval_method = args.eval_method
    exp_log_dir = args.exp_log_dir
    job_id = args.job_id
    n_exp = args.n_exp
    device = args.device

    use_struct = True if args.use_struct == "True" else False
    save_probs = True if args.save_probs == "True" else False # whether to save model probs for val & test

    # evaluation-specific arguments 
    neg_factor = args.neg_factor # ["1", "2", "3", "4", "5"]
    smoo_factor = args.smoo_factor # ["0.75", "0.8", "0.9", "0.95", "1.0"]
    strct_dataset = args.strct_dataset # ["dataset1", "dataset2", "dataset3"]
    strct_strategy = args.strct_strategy # ["conditional", "full_mean", "subset_mean"]
    strct_aug_train = True if args.strct_aug_train == "True" else False
    strct_aug_eval = True if args.strct_aug_eval == "True" else False
    
    epo_f1 = args.epo_f1
    epo_f2 = args.epo_f2
    epo_f3 = args.epo_f3
    epo_f4 = args.epo_f4
    epo_f5 = args.epo_f5
    n_epochs = args.n_epochs
    folds = args.folds

    device = args.device

    # MAFusionPPI architecture arguments
    model_kwargs = {
        "exclude_modalities": args.exclude_modalities,
        "mlp_dropout": args.mlp_dropout[0],
        "head_dropout": args.head_dropout[0],
        "self_attn_dropout": args.self_attn_dropout[0],
        "join_attn_feat": args.join_attn_feat,
        "compound_dim": args.compound_dim,
        "compound_proj_dim": args.compound_proj_dim[0],
        "ppi_fuse_setting": args.ppi_fuse_setting[0],
        "head_fuse": args.head_fuse[0],
        "proj_feat": True if args.proj_feat[0] == "True" else False
    }

    train_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size
    }

    hyperparameters_kwargs = {
        "lr": args.lr, # [1e-5, 1e-6, 5e-5]
        "weight_decay": args.weight_decay, # [1e-3, 1e-4, 1e-5]
        "mlp_dropout": args.mlp_dropout, # [0.3, 0.1]
        "head_dropout": args.head_dropout, # [0.3, 0.1, 0.5]
        "self_attn_dropout": args.self_attn_dropout, # [0.1, 0.3]
        "compound_proj_dim": args.compound_proj_dim, # [0.1, 0.3]
        "ppi_fuse_setting": args.ppi_fuse_setting, # ['cat', 'gate', 'self_attn']
        "head_fuse": args.head_fuse, # ['cat', 'fuse']
        "proj_feat": [p == "True" for p in args.proj_feat] # ['True', 'False']
    }

    logger.info(log_msg)
    # constract log file in order to log the results
    log_dir = os.path.join("results", "logs", exp_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_file_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', force=True)
    
    # run selected evaluation 
    if eval_method == "cold":
        nel = [epo_f1, epo_f2, epo_f3, epo_f4, epo_f5]
        train_test_cold_start_ppi(nel=nel, n=n_exp)
    elif eval_method == "hyperparam_search":
        hyperparam_search(
          res_file_name=res_file_name,
          save_probs=save_probs,
          use_struct=use_struct,
          strct_dataset=strct_dataset,
          strct_strategy=strct_strategy,
          strct_aug_train=strct_aug_train,
          strct_aug_eval=strct_aug_eval,
          job_id=job_id,
          n=1,
          device=device,
          model_kwargs=model_kwargs,
          training_kwargs=train_kwargs,
          hyperparam_kwargs=hyperparameters_kwargs
        )
    elif eval_method == "cv_neg_smoo":
        cold_neg_smoo_eval_(
            res_file_name=res_file_name,
            save_probs=save_probs,
            use_struct=use_struct,
            neg_factor=neg_factor,
            smoo_factor=smoo_factor,
            strct_dataset=strct_dataset,
            strct_strategy=strct_strategy,
            strct_aug_train=strct_aug_train,
            strct_aug_eval=strct_aug_eval,
            job_id=job_id,
            n=n_exp,
            device=device,
            model_kwargs=model_kwargs,
        )

    elif eval_method == "cv_cold":
        cv_cold_eval_(
            res_file_name=res_file_name,
            save_probs=save_probs,
            use_struct=use_struct,
            strct_dataset=strct_dataset,
            strct_strategy=strct_strategy,
            strct_aug_train=strct_aug_train,
            strct_aug_eval=strct_aug_eval,
            job_id=job_id,
            n=n_exp,
            device=device,
            model_kwargs=model_kwargs,
        )

    elif eval_method == "mcd":
        pass  # TODO: finish building mcd pipeline & test it

    else:
        raise ValueError(f"Unsupported eval_method: {eval_method}")


if __name__ == "__main__":
    main()


