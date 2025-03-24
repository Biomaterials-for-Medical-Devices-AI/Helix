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

MLREM_GRID = {
    "alpha": [0.0, 0.1, 0.5, 1.0],
    "beta": [10],
    "weight_threshold": [1e-4, 1e-3, 1e-2],
    "max_iterations": [200, 300, 400],
    "tolerance": [0.01, 0.001],
}
