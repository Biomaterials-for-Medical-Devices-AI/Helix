import ast
import os
import sys

from feature_importance.feature_importance_options import FeatureImportanceOptions


class FuzzyOptions(FeatureImportanceOptions):
    """
    Feature Importance options
    Parent class: BaseOptions
    """

    def __init__(self) -> None:
        super().__init__()


    def initialize(self) -> None:
        """Initialize train options"""
        FeatureImportanceOptions.initialize(self)        

        self.parser.add_argument(
            "--fuzzy_feature_selection",
            type=bool,
            default=True,
            help="Flag for fuzzy feature selection",
        ),  
        self.parser.add_argument(
            "--number_fuzzy_features",
            type=int,
            default=10,  
            help="Number of features selected for fuzzy interpretation",
        ),
        self.parser.add_argument(
            "--is_granularity",
            type=bool,
            default=True,
            help="Flag for granularity of features",
        ),
        self.parser.add_argument(
            "--number_clusters",
            type=int,
            default=5,
            help="Number of clusters for target variable in fuzzy interpretation",
        ),
        self.parser.add_argument(
            "--names_clusters",
            type=ast.literal_eval,
            default=['very low', 'low', 'medium', 'high', 'very high'],
            help="Names of the clusters for target variable in fuzzy interpretation",
        ),
        self.parser.add_argument(
            "--number_rules",
            type=int,
            default=5,
            help="Number of top occuring rules to consider for fuzzy synergy analysis",
        ),
        self.parser.add_argument(
            "--save_fuzzy_set_plots",
            type=bool,
            default=True,
            help="Flag for saving fuzzy set plots",
        ),
        self.parser.add_argument(
            "--fuzzy_log_dir",
            type=str,
            default="fuzzy",
            help="Path to the directory to store logs",
            required=False,
        ),
        