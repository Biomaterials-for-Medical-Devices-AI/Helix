from dataclasses import dataclass
from typing import Any

@dataclass
class MlOptions:
    data_path: str
    data_split: Any
    n_bootstraps: int
    save_actual_pred_plots: bool
    model_types: Any
    normalization: str
    ml_log_dir: str