

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
            "--fi_methods",
            type=lambda x: ast.literal_eval(x),
            default={'Permutation Importance': False,
                     'SHAP': False,},
            help="Feature importance methods to use",
        ),

        self.parser.add_argument(
            "--save_fi_results",
            type=bool,
            default=True,
            help="Flag to save feature importance results",
        ),

        self.parser.add_argument(
            "--save_fi_plots",
            type=bool,
            default=True,
            help="Flag to save feature importance plots",
        ),

        # Update --is_feature_importance to True
        self.parser.set_defaults(is_feature_importance=True),