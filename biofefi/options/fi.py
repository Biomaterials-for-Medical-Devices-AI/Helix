from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureImportanceOptions:
    global_importance_methods: Any = {
        "Permutation Importance": {"type": "global", "value": True},
        "SHAP": {"type": "global", "value": False},
    }
    feature_importance_ensemble: Any = {"Mean": True, "Majority Vote": True}
    local_importance_methods: Any = {
        "LIME": {"type": "local", "value": True},
        "SHAP": {"type": "local", "value": False},
    }
    save_feature_importance_results: bool = True
    save_feature_importance_options: bool = True
    save_feature_importance_plots: bool = True
    num_features_to_plot: int = 5
    angle_rotate_yaxis_labels: int = 60
    angle_rotate_xaxis_labels: int = 10
    plot_axis_font_size: int = 8
    plot_axis_tick_size: int = 8
    plot_title_font_size: int = 20
    plot_colour_scheme: str = "classic"
    permutation_importance_scoring: str = "neg_mean_absolute_error"
    permutation_importance_repeat: int = 10
    shap_reduce_data: int = 50
    fi_log_dir: str = "fi"
