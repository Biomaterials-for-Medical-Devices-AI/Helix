import os

import pandas as pd

from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.file_paths import fi_plot_dir, helix_experiments_base_dir
from helix.options.plotting import PlottingOptions
from helix.services.data import TabularData
from helix.services.feature_importance.ensemble_methods import (
    calculate_ensemble_majorityvote,
    calculate_ensemble_mean,
)
from helix.services.feature_importance.global_methods import (
    calculate_global_shap_values,
    calculate_permutation_importance,
)
from helix.services.feature_importance.local_methods import (
    calculate_lime_values,
    calculate_local_shap_values,
)
from helix.services.feature_importance.results import save_importance_results
from helix.services.plotting import (
    plot_global_shap_importance,
    plot_lime_importance,
    plot_local_shap_importance,
)
from helix.utils.logging_utils import Logger
from helix.utils.utils import create_directory


class Interpreter:
    """
    Interpreter class to interpret the model results.

    """

    def __init__(
        self,
        fi_opt: FeatureImportanceOptions,
        exec_opt: ExecutionOptions,
        plot_opt: PlottingOptions,
        logger: Logger | None = None,
    ) -> None:
        self._fi_opt = fi_opt
        self._logger = logger
        self._exec_opt = exec_opt
        self._plot_opt = plot_opt
        self._feature_importance_methods = self._fi_opt.global_importance_methods
        self._local_importance_methods = self._fi_opt.local_importance_methods
        self._feature_importance_ensemble = self._fi_opt.feature_importance_ensemble

    def interpret(self, models: dict, data: TabularData) -> tuple[dict, dict, dict]:
        """
        Interpret the model results using the selected feature importance methods
        and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            data (TabularData): The data to interpret.
        Returns:
            tuple[dict, dict, dict]:
            Global, local and ensemble feature importance votes.
        """
        # Load just the first fold of the data and the first models for interpretation
        X, y = data.X_train[0], data.y_train[0]
        self._logger.info("-------- Start of feature importance logging--------")
        global_importance_results = self._individual_feature_importance(models, X, y)
        local_importance_results = self._local_feature_importance(models, X)
        ensemble_results = self._ensemble_feature_importance(global_importance_results)
        self._logger.info("-------- End of feature importance logging--------")

        return global_importance_results, local_importance_results, ensemble_results

    def _individual_feature_importance(
        self, models: dict, X: pd.DataFrame, y: pd.Series
    ):
        """
        Calculate global feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        """
        feature_importance_results = {}
        if not any(
            sub_dict["value"] for sub_dict in self._feature_importance_methods.values()
        ):
            self._logger.info("No feature importance methods selected")
            self._logger.info("Skipping global feature importance methods")
        else:
            for model_type, model in models.items():
                self._logger.info(
                    f"Global feature importance methods for {model_type}..."
                )
                feature_importance_results[model_type] = {}

                # Run methods with TRUE values in the dictionary
                # of feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._feature_importance_methods.items():
                    if value["value"]:
                        # Select the first model in the list - model[0]
                        if feature_importance_type == "Permutation Importance":
                            # Run Permutation Importance -
                            permutation_importance_df = calculate_permutation_importance(
                                model=model[0],
                                X=X,
                                y=y,
                                permutation_importance_scoring=self._fi_opt.permutation_importance_scoring,
                                permutation_importance_repeat=self._fi_opt.permutation_importance_repeat,
                                random_state=self._exec_opt.random_state,
                                logger=self._logger,
                            )
                            save_importance_results(
                                feature_importance_df=permutation_importance_df,
                                model_type=model_type,
                                importance_type=value["type"],
                                feature_importance_type=feature_importance_type,
                                experiment_name=self._exec_opt.experiment_name,
                                fi_opt=self._fi_opt,
                                plot_opt=self._plot_opt,
                                logger=self._logger,
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = permutation_importance_df

                        if feature_importance_type == "SHAP":
                            # Run SHAP
                            shap_df, _ = calculate_global_shap_values(
                                model=model[0],
                                X=X,
                                shap_reduce_data=self._fi_opt.shap_reduce_data,
                                logger=self._logger,
                            )
                            fig = plot_global_shap_importance(
                                shap_values=shap_df,
                                plot_opts=self._plot_opt,
                                num_features_to_plot=self._fi_opt.num_features_to_plot,
                                title=f"{feature_importance_type} - {value['type']} - {model_type}",
                            )
                            save_dir = fi_plot_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(
                                save_dir
                            )  # will create the directory if it doesn't exist
                            fig.savefig(
                                save_dir
                                / f"{feature_importance_type}-{value['type']}-{model_type}-bar.png"
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = shap_df

        return feature_importance_results

    def _local_feature_importance(self, models, X):
        """
        Calculate local feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        """
        feature_importance_results = {}
        if not any(
            sub_dict["value"] for sub_dict in self._local_importance_methods.values()
        ):
            self._logger.info("No local feature importance methods selected")
            self._logger.info("Skipping local feature importance methods")
        else:
            for model_type, model in models.items():
                self._logger.info(
                    f"Local feature importance methods for {model_type}..."
                )
                feature_importance_results[model_type] = {}

                # Run methods with TRUE values in the dictionary of feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._local_importance_methods.items():
                    if value["value"]:
                        # Select the first model in the list - model[0]
                        if feature_importance_type == "LIME":
                            # Run Permutation Importance
                            lime_importance_df = calculate_lime_values(
                                model[0], X, self._exec_opt.problem_type, self._logger
                            )
                            fig = plot_lime_importance(
                                df=lime_importance_df,
                                plot_opts=self._plot_opt,
                                num_features_to_plot=self._fi_opt.num_features_to_plot,
                                title=f"{feature_importance_type} - {model_type}",
                            )
                            save_dir = fi_plot_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(
                                save_dir
                            )  # will create the directory if it doesn't exist
                            fig.savefig(
                                save_dir
                                / f"{feature_importance_type}-{model_type}-violin.png"
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = lime_importance_df

                        if feature_importance_type == "SHAP":
                            # Run SHAP
                            shap_df, shap_values = calculate_local_shap_values(
                                model=model[0],
                                X=X,
                                shap_reduce_data=self._fi_opt.shap_reduce_data,
                                logger=self._logger,
                            )
                            fig = plot_local_shap_importance(
                                shap_values=shap_values,
                                plot_opts=self._plot_opt,
                                num_features_to_plot=self._fi_opt.num_features_to_plot,
                                title=f"{feature_importance_type} - {value['type']} - {model_type}",
                            )
                            save_dir = fi_plot_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(
                                save_dir
                            )  # will create the directory if it doesn't exist
                            fig.savefig(
                                save_dir
                                / f"{feature_importance_type}-{value['type']}-{model_type}-beeswarm.png"
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = shap_df

        return feature_importance_results

    def _ensemble_feature_importance(self, feature_importance_results):
        """
        Calculate ensemble feature importance methods.
        Parameters:
            feature_importance_results (dict): Dictionary of feature importance results.
        Returns:
            dict: Dictionary of ensemble feature importance results.
        """
        ensemble_results = {}

        if not any(self._feature_importance_ensemble.values()):
            self._logger.info("No ensemble feature importance method selected")
            self._logger.info("Skipping ensemble feature importance analysis")
        else:
            self._logger.info("Ensemble feature importance methods...")
            for ensemble_type, value in self._feature_importance_ensemble.items():
                if value:
                    if ensemble_type == "Mean":
                        # Calculate mean of feature importance results
                        mean_results = calculate_ensemble_mean(
                            feature_importance_results, self._logger
                        )
                        save_importance_results(
                            feature_importance_df=mean_results,
                            model_type=f"Ensemble {ensemble_type}",
                            importance_type=None,
                            feature_importance_type=ensemble_type,
                            experiment_name=self._exec_opt.experiment_name,
                            fi_opt=self._fi_opt,
                            plot_opt=self._plot_opt,
                            logger=self._logger,
                        )
                        ensemble_results[ensemble_type] = mean_results

                    if ensemble_type == "Majority Vote":
                        # Calculate majority vote of feature importance results
                        majority_vote_results = calculate_ensemble_majorityvote(
                            feature_importance_results, self._logger
                        )
                        save_importance_results(
                            feature_importance_df=majority_vote_results,
                            model_type=f"Ensemble {ensemble_type}",
                            importance_type=None,
                            feature_importance_type=ensemble_type,
                            experiment_name=self._exec_opt.experiment_name,
                            fi_opt=self._fi_opt,
                            plot_opt=self._plot_opt,
                            logger=self._logger,
                        )
                        ensemble_results[ensemble_type] = majority_vote_results

            self._logger.info(
                f"Ensemble feature importance results: {os.linesep}{ensemble_results}"
            )

        return ensemble_results
