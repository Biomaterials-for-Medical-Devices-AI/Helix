from helix.options.choices.ui import SVM_KERNELS

LINEAR_MODEL_GRID = {
    "fit_intercept": [True, False],
}

RANDOM_FOREST_GRID = {
    "n_estimators": [100, 300, 500],
    "min_samples_split": [2, 0.05, 0.1],
    "min_samples_leaf": [1, 0.05, 0.1],
    "max_depth": [None, 3, 6],
}

XGB_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 3, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.15, 0.20, 0.25],
}

SVM_GRID = {
    "kernel": [k.lower() for k in SVM_KERNELS],
    "degree": [2, 3, 4],
    "C": [1.0, 10.0, 100],
}

BRNN_GRID = {
    "batch_size": [32],
    "epochs": [10],
    "hidden_dim": [64],
    "output_dim": [1],
    "lr": [0.0003],
    "prior_mu": [0],
    "prior_sigma": [1],
    "lambda_reg": [0.01],
}
