

import ast
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_options import BaseOptions


class FeatureImportanceOptions(BaseOptions):
    """
    Feature Importance options
    Parent class: BaseOptions
    """

    def __init__(self) -> None:
        super().__init__()


    def initialize(self) -> None:
        """Initialize train options"""
        BaseOptions.initialize(self)

        self.parser.add_argument(
            "--feature_importance_methods",
            type=lambda x: ast.literal_eval(x),
            default={'Permutation Importance': True,
                     'SHAP': True,},
            help="Feature importance methods to use",
        ),

        

        self.parser.add_argument(
            "--save_feature_importance_results",
            type=bool,
            default=True,
            help="Flag to save feature importance results",
        ),

        self.parser.add_argument(
            "--save_feature_importance_metrics",
            type=bool,
            default=True,
            help="Flag to save feature importance metrics",
        ),

        self.parser.add_argument(
            "--save_feature_importance_plots",
            type=bool,
            default=True,
            help="Flag to save feature importance plots",
        ),

        

        # Update --is_feature_importance to True
        self.parser.set_defaults(is_feature_importance=True),