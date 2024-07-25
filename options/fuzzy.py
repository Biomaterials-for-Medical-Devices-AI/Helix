from dataclasses import dataclass
from typing import Any


@dataclass
class FuzzyOptions:
    fuzzy_feature_selection: bool = True
    number_fuzzy_features: int = 10
    is_granularity: bool = True
    number_clusters: int = 5
    names_clusters: Any = ["very low", "low", "medium", "high", "very high"]
    number_rules: int = 5
    save_fuzzy_set_plots: bool = True
    fuzzy_log_dir: str = "fuzzy"
