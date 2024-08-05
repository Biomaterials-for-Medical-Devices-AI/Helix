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
            default="machine_learning/data/Topo_surface_5_features_macrophage.csv",
            help="Path to the data",
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
            "--n_bootstraps",
            type=int,
            default=3,
            help="Number of bootstraps to use",
        )

        self.parser.add_argument(
            "--save_actual_pred_plots",
            type=bool,
            default=True,
            help="Flag to save actual vs predicted plots",
        )

        self.parser.add_argument(
            "--model_types",
            type=lambda x: ast.literal_eval(x),
            default={
                "Linear Model": {"use": False, "params": {"fit_intercept": False}},
                "Random Forest": {
                    "use": True,
                    "params": {
                        "n_estimators": 300,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_depth": 6,
                    },
                },
                "XGBoost": {
                    "use": True,
                    "params": {
                        "kwargs": {
                            "n_estimators": 300,
                            "max_depth": 6,
                            "learning_rate": 0.01,
                            "subsample": 0.5,
                        }
                    },
                },
            },
            help="Model types to use",
        )

        self.parser.add_argument(
            "--normalization",
            type=str,
            default="None",
            help="Normalization method: Standardization or MinMax",
            choices=["Standardization", "MinMax", "None"],
            required=False,
        )

        self.parser.add_argument(
            "--ml_log_dir",
            type=str,
            default="ml",
            help="Path to the directory to store logs",
            required=False,
        ),

        ######## New Parameters to be added above this line ########

        # self.parser.set_defaults(is_machine_learning=True),

    def fuzzy_reset_bootstraps(self):
        """Set the number of bootstraps to 1"""
        if not self.initialized:
            self.initialize()
        self.parser.set_defaults(n_bootstraps=1)
        self.parser.set_defaults(is_machine_learning=False)
        self._opt = self.parser.parse_args()

        return self._opt
