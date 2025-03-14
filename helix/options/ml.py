from dataclasses import dataclass


@dataclass
class MachineLearningOptions:
    model_types: dict
    save_actual_pred_plots: bool = True
    ml_log_dir: str = "ml"
    save_models: bool = True
    ml_plot_dir: str = "ml"
    use_hyperparam_search: bool = True

@dataclass
class TabPFNOptions:
    """
    A class containing the parameters for TabPFN models
    """
    n_estimators: int = 100
    random_state: int = 42

# ----- Bayesian Regularised Neural Network Parameters ----- >>>>>>>>


@dataclass
class BrnnOptions:
    """
    This class contains the parameters as an options
    for the Bayesian Regularised Neural Network.
    """

    batch_size: int = 32
    epochs: int = 10
    hidden_dim: int = 64
    output_dim: int = 1
    lr: float = 0.0003
    prior_mu: int = 0
    prior_sigma: int = 1
    lambda_reg: float = 0.01
    classification_cutoff: float = 0.5
