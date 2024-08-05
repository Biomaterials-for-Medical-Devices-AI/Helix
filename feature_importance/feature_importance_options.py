import ast

from machine_learning.ml_options import MLOptions


class FeatureImportanceOptions(MLOptions):
    """
    Feature Importance options
    Parent class: BaseOptions
    """

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize train options"""
        MLOptions.initialize(self)

        self.parser.add_argument(
            "--global_importance_methods",
            type=lambda x: ast.literal_eval(x),
            default={
                "Permutation Importance": {"type": "global", "value": True},
                "SHAP": {"type": "global", "value": False},
            },
            help="Global feature importance methods to use",
        ),

        self.parser.add_argument(
            "--feature_importance_ensemble",
            type=lambda x: ast.literal_eval(x),
            default={"Mean": True, "Majority Vote": True},
            help="Feature importance ensemble methods to use",
        ),

        self.parser.add_argument(
            "--local_importance_methods",
            type=lambda x: ast.literal_eval(x),
            default={
                "LIME": {"type": "local", "value": True},
                "SHAP": {"type": "local", "value": False},
            },
            help="Local feature importance methods to use in fuzzy interpretation",
        ),
        self.parser.add_argument(
            "--save_feature_importance_results",
            type=bool,
            default=True,
            help="Flag to save feature importance results",
        ),
        self.parser.add_argument(
            "--save_feature_importance_options",
            type=bool,
            default=True,
            help="Flag to save feature importance options",
        ),
        self.parser.add_argument(
            "--save_feature_importance_plots",
            type=bool,
            default=False,
            help="Flag to save feature importance plots",
        ),

        self.parser.add_argument(
            "--num_features_to_plot",
            type=int,
            default=5,
            help="Number of top important features to plot for Permutation Importance",
        ),
        self.parser.add_argument(
            "--angle_rotate_yaxis_labels",
            type=int,
            default=60,
            help="Angle to rotate y-axis labels for better readability",
        ),
        self.parser.add_argument(
            "--angle_rotate_xaxis_labels",
            type=int,
            default=10,
            help="Angle to rotate x-axis labels for better readability",
        ),
        self.parser.add_argument(
            "--permutation_importance_scoring",
            type=str,
            default="neg_mean_absolute_error",
            choices=[
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "accuracy",
                "f1",
            ],
            help="Scoring function for permutation importance",
        ),
        self.parser.add_argument(
            "--permutation_importance_repeat",
            type=int,
            default=10,
            help="Number of repetitions for permutation importance",
        ),
        self.parser.add_argument(
            "--shap_reduce_data",
            type=int,
            default=50,
            choices=[20, 50, 70, 100],
            help="Percentage of data to consider when calculating SHAP",
        ),

        self.parser.add_argument(
            "--fi_log_dir",
            type=str,
            default="fi",
            help="Path to the directory to store logs",
            required=False,
        ),

        # Update --is_feature_importance to True
        # self.parser.set_defaults(is_feature_importance=True),
