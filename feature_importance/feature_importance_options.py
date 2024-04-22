

import ast
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_options import BaseOptions


class FIOptions(BaseOptions):
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
            "--save_feature_importance_results",
            type=bool,
            default=True,
            help="Flag to save feature importance results",
        ),

        self.parser.add_argument(
            "--save_feature_importance_plots",
            type=bool,
            default=True,
            help="Flag to save feature importance plots",
        ),

        # Update --is_feature_importance to True
        self.parser.set_defaults(is_feature_importance=True),