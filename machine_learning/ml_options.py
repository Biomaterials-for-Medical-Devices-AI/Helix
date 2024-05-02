import ast
import os
import sys

from base_options import BaseOptions


class MLOptions(BaseOptions):
    """Machine Learning options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize ML options"""
        BaseOptions.initialize(self)

        self.parser.add_argument(
            "--data_path",
            type=str,
            default="machine_learning/data/granular_surface_macrophage.csv",
            help="Path to the data",
            required=False,
        )

        self.parser.add_argument(
            "--problem_type",
            type=str,
            default="regression",
            help="Problem type: classification or regression",
            choices=["classification", "regression"],
            required=False,
        )

        self.parser.add_argument(
            "--data_split",
            type=lambda x: ast.literal_eval(x),
            default={"type": "holdout", "test_size": 0.2},
            help="Data split method: holdout or kfold",
            choices=[
                {"type": "holdout", "test_size": 0.2},
                {"type": "kfold", "n_splits": 5},
                {"type": "LOOCV"},
            ],
        )

        self.parser.add_argument(
            "--model_types",
            type=lambda x: ast.literal_eval(x),
            default={
                "Linear Model": {"use": True, "params": {"fit_intercept": False}},
                "Random Forest": {"use": True, "params": {"fit_intercept": True}},
            },
            help="Model types to use",
        )

        self.parser.add_argument(
            "--normalization",
            type=str,
            default="MinMax",
            help="Normalization method: Standardization or MinMax",
            choices=["Standardization", "MinMax", "None"],
            required=False,
        )

        self.parser.add_argument(
            "--log_dir",
            type=str,
            default="ml",
            help="Path to the directory to store logs",
            required=False,
        )

        ######## New Parameters to be added above this line ########
        self._is_train = True
