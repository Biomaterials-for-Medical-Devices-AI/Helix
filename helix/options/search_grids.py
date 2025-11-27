from helix.options.choices.ui import SVM_KERNELS
from helix.options.enums import ActivationFunctions
import numpy as np


def _create_KAN_width():
    max_num_neurons = 8
    max_num_hidden_layers = 3

    widths = [
        [neurons for _ in range(hid_layer)]
        for neurons in range(3, max_num_neurons, 2)
        for hid_layer in range(1, max_num_hidden_layers)
    ]

    return widths


def get_KAN_grid():

    KAN_GRID = {
        "width": _create_KAN_width(),
        "grid": [5, 7],
        "k": [1, 3],
        "epochs": [100, 200],
        "lr": [1.0, 0.1, 0.01],
        "batch": [-1, 16, 32],
    }

    return KAN_GRID


LINEAR_MODEL_GRID = {
    "fit_intercept": [True, False],
}

LASSO_GRID = {"fit_intercept": [True, False], "alpha": [0.05, 0.1, 0.5, 0.8, 1.0]}

RIDGE_GRID = {"fit_intercept": [True, False], "alpha": [0.05, 0.1, 0.5, 0.8, 1.0]}

ELASTICNET_GRID = {
    "fit_intercept": [True, False],
    "alpha": [0.05, 0.1, 0.5, 0.8, 1.0],
    "l1_ratio": [0, 0.5, 1],
}

KNN_GRID = {"n_neighbors": [5, 10, 15], "leaf_size": [10, 15, 30, 45, 60], "p": [1, 2]}

GNB_GRID = {"var_smoothing": [1e-09, 1e-08]}

MLP_GRID = {
    "hidden_layer_sizes": [(25,), (50,), (75,)],
    "activation": ["logistic", "relu"],
    "solver": ["adam"],
    "learning_rate": ["constant"],
    "learning_rate_init": [0.001, 0.01],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "max_iter": [200],
    "early_stopping": [True, False],
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


MLREM_GRID = {
    "alpha": [0.05, 0.1, 0.5, 0.8],
    "max_beta": [40.0],
    "weight_threshold": [1e-3],
    "max_iterations": [300],
    "tolerance": [0.01],
}

BRNN_GRID = {
    "hidden_layer_sizes": [32, 64, 128],
    "activation": [
        ActivationFunctions.Logistic,
        ActivationFunctions.Tanh,
        ActivationFunctions.ReLU,
    ],
    "batch_size": [64, 128],
    "learning_rate_init": [0.01, 0.005, 0.001],
    "max_iter": [200, 300],
}
