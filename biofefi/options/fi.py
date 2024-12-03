from dataclasses import dataclass


@dataclass
class FeatureImportanceOptions:
    global_importance_methods: dict
    feature_importance_ensemble: dict
    local_importance_methods: dict
    save_feature_importance_results: bool = True
    save_feature_importance_options: bool = True
    save_feature_importance_plots: bool = True
    num_features_to_plot: int = 5
    permutation_importance_scoring: str = "neg_mean_absolute_error"
    permutation_importance_repeat: int = 10
    shap_reduce_data: int = 50
    fi_log_dir: str = "fi"
