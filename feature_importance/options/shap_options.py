

import ast
import os
import sys


from options.feature_importance_options import FeatureImportanceOptions


class SHAPOptions(FeatureImportanceOptions):
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
            "--num_features_to_plot_shap",
            type=int,
            default=5,
            help="Number of top important features to plot for SHAP",
        ),
        self.parser.add_argument(
            "--angle_rotate_yaxis_labels",
            type=int,
            default=60,
            help="Angle to rotate y-axis labels for better readability",
        )
