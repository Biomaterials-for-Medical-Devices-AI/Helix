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
            default="machine_learning/data_clean.csv",
            help="Path to the data",
            required=False,
        )

        self.parser.add_argument(
            "--problem_type",
            type=str,
            default="classification",
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
                "Logistic Regression": False, 
                "Random Forest": False,
                "XGBoost": False,
                "Neural Network": False,
                },
            help="Model types to use",
        )


        ######## New Parameters to be added above this line ########
        self._is_train = True
