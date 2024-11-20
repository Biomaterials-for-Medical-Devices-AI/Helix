from dataclasses import dataclass


@dataclass
class MachineLearningOptions:
    model_types: dict
    save_actual_pred_plots: bool = True
    ml_log_dir: str = "ml"
    save_models: bool = True
    ml_plot_dir: str = "ml"
