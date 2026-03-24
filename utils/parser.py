import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_msg",
        type=str,
        default="log message",
        help="First message to display in the log file; for experiment description"
    )
    parser.add_argument(
        "--res_file_name",
        type=str,
        default="res",
        help="name of the result; pandas.DataFrame() object"
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="cv_neg_smoo",
        choices=["cv_cold", "cv", "cv_neg_smoo", "cold", "mcd"],
        help="evaluation method"
    )
    parser.add_argument(
        "--cv_method",
        type=str,
        default="all",
        choices=["all", "per_fam"],
        help="The split for 10-fold cross-validation - all data or per RNA subtype"
    )
    parser.add_argument(
        "--neg_factor",
        type=str,
        default="5",
        choices=["1", "2", "3", "4", "5"],
        help="negative sampling hyperparameter"
    )
    parser.add_argument(
        "--smoo_factor",
        type=str,
        default="1.0",
        choices=["0.75", "0.8", "0.9", "0.95", "1.0"],
        help="smoothing factor"
    )
    parser.add_argument(
        "--save_probs",
        type=str,
        default="False",
        choices=["True", "False"],
        help="whether to use save model probabilities (for validation & test)"
    )

    parser.add_argument(
        "--use_struct",
        type=str,
        default="True",
        choices=["True", "False"],
        help="whether to use structure features or not"
    )
    parser.add_argument(
        "--strct_dataset",
        type=str,
        default="dataset1",
        choices=["dataset1", "dataset2", "dataset3"],
        help="structure dataset type"
    )
    parser.add_argument(
        "--strct_strategy",
        type=str,
        default="conditional",
        choices=["conditional", "full_mean", "subset_mean"],
        help="amino acid sampling for sequence construction"
    )
    parser.add_argument(
        "--strct_aug_train",
        type=str,
        default="False",
        choices=["True", "False"],
        help="whether to use conformational data augmentation during training phase"
    )

    parser.add_argument(
        "--strct_aug_eval",
        type=str,
        default="False",
        choices=["True", "False"],
        help="whether to use conformational data augmentation during evaluation phase"
    )
    parser.add_argument(
        "--n_exp",
        type=int,
        default=10,
        help="number of experiments for statistically significant evaluation"
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default="date@j1",
        help="job unique id (in order to differentiate between different experiment settings & jobs); format=date@job_num (example: 1203@j3)"
    )
    parser.add_argument(
        "--exp_log_dir",
        type=str,
        default="log_dir",
        help="experiments sub-directory name; located at results/logs/exp_log_dir"
    )
    parser.add_argument(
        "--epo_f1",
        type=int,
        default=50,
        help="number of epochs to train fold 1 (cold eval)"
    )
    parser.add_argument(
        "--epo_f2",
        type=int,
        default=50,
        help="number of epochs to train fold 2 (cold eval)"
    )
    parser.add_argument(
        "--epo_f3",
        type=int,
        default=50,
        help="number of epochs to train fold 3 (cold eval)"
    )
    parser.add_argument(
        "--epo_f4",
        type=int,
        default=50,
        help="number of epochs to train fold 4 (cold eval)"
    )
    parser.add_argument(
        "--epo_f5",
        type=int,
        default=50,
        help="number of epochs to train fold 5 (cold eval)"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        nargs=5,
        default=[50, 50, 50, 50, 50],
        help="List of 5 epoch values for cold eval"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of folds for cold eval (1-5)"
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default="res",
        help="name of the logger file (e.g., result.log)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to use - cuda/cpu"
    )

    # ---- Model architecture arguments ----
    parser.add_argument(
        "--exclude_modalities",
        nargs="+",
        default=None,
        choices=["esm", "fegs", "gae"],
        help="Sequence modalities to exclude"
    )

    parser.add_argument(
        "--mlp_dropout",
        nargs="+",
        type=float,
        default=[0.3],
        help="Dropout used in modality MLPs"
    )

    parser.add_argument(
        "--head_dropout",
        nargs="+",
        type=float,
        default=[0.3],
        help="Dropout used in prediction head"
    )

    parser.add_argument(
        "--self_attn_dropout",
        nargs="+",
        type=float,
        default=[0.1],
        help="Dropout used in self-attention layers"
    )

    parser.add_argument(
        "--join_attn_feat",
        type=str,
        default="both",
        choices=["both", "ppiformer", "omega"],
        help="Structure features used in joint attention"
    )

    parser.add_argument(
        "--compound_dim",
        type=int,
        default=850,
        help="Input dimension of structure embeddings"
    )

    parser.add_argument(
        "--compound_proj_dim",
        nargs="+",
        type=int,
        default=[256],
        help="Projected dimension used in joint attention"
    )

    parser.add_argument(
        "--ppi_fuse_setting",
        nargs="+",
        type=str,
        default=["cat"],
        choices=["cat", "gate", "self_attn"],
        help="PPI fusion mechanism"
    )

    parser.add_argument(
        "--head_fuse",
        nargs="+",
        type=str,
        default=["cat"],
        choices=["cat", "gate"],
        help="Final head fusion method"
    )

    parser.add_argument(
        "--proj_feat",
        nargs="+",
        type=str,
        default=["False"],
        choices=["True", "False"],
        help="Use lightweight projection layers instead of full MLP encoders"
    )

    parser.add_argument(
        "--lr",
        nargs="+",
        type=float,
        default=[1e-5],
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        nargs="+",
        type=float,
        default=[1e-3],
        help="Weight decay for optimizer"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )

    return parser.parse_args()