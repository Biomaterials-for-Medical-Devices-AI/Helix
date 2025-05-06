from helix.options.choices.ui import SVM_KERNELS
from helix.options.enums import ActivationFunctions

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
