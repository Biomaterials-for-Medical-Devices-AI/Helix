from dataclasses import dataclass


@dataclass
class MachineLearningOptions:
    model_types: dict
    save_actual_pred_plots: bool = True
    ml_log_dir: str = "ml"
    save_models: bool = True
    ml_plot_dir: str = "ml"
    use_hyperparam_search: bool = True


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


# ----- Maximum Likelihood Regularised EM Parameters ----- >>>>>>>>


@dataclass
class MLREMOptions:
    """
    This class contains the parameters as options
    for the Maximum Likelihood Regularised EM model.
    """

    alpha: float = 0.0  # Regularisation parameter
    beta: float = 1.0  # Scaling parameter
    weight_threshold: float = 1e-3  # Threshold for weight pruning
    max_iterations: int = 300  # Maximum iterations for EM
    tolerance: float = 0.01  # Convergence tolerance
