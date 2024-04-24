

import ast
import os
import sys

print(os.getcwd())

from options.feature_importance_options import FeatureImportanceOptions


class PermutationImportanceOptions(FeatureImportanceOptions):
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
            "--permutation_importance_scoring",
            type=str,
            default='neg_mean_absolute_error',
            choices=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'accuracy', 'f1'],
            help="Scoring function for permutation importance",
        ),
        self.parser.add_argument(
            "--permutation_importance_repeat",
            type=int,
            default=10,
            help="Number of repetitions for permutation importance",
        ),

        
        self.parser.add_argument(
            "--num_features_to_plot_pi",
            type=int,
            default=5,
            help="Number of top important features to plot for Permutation Importance",
        ),
        self.parser.add_argument(
            "--angle_rotate_xaxis_labels",
            type=int,
            default=10,
            help="Angle to rotate x-axis labels for better readability",
        )
