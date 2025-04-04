import json
import os

import pandas as pd

from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.file_paths import (
    fi_plot_dir,
    helix_experiments_base_dir,
    ml_metrics_mean_std_path,
)
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
from helix.services.metrics import find_mean_model_index
from helix.services.plotting import (
    plot_global_shap_importance,
    plot_lime_importance,
    plot_local_shap_importance,
)
from helix.utils.logging_utils import Logger
from helix.utils.utils import create_directory


class FeatureImportanceEstimator:
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
        global_importance_results = self._global_feature_importance(models, X, y)
        global_importance_df = self._stack_importances(global_importance_results)
        local_importance_results = self._local_feature_importance(models, X)
        ensemble_results = self._ensemble_feature_importance(global_importance_results)
        self._logger.info("-------- End of feature importance logging--------")

        return global_importance_df, local_importance_results, ensemble_results

    def _global_feature_importance(self, models: dict, data: TabularData):
        """
        Calculate global feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            data (TabularData): The data to interpret.
        Returns:
            dict: Dictionary of feature importance results.
        """
        feature_importance_results = {}
        if not any(
            sub_dict["value"] for sub_dict in self._feature_importance_methods.values()
        ):
            self._logger.info("No feature importance methods selected")
            self._logger.info("Skipping global feature importance methods")
            return feature_importance_results

        # Iterate through all data indices
        for idx in range(len(data.X_train)):
            X, y = data.X_train[idx], data.y_train[idx]

            # Iterate through all models
            for model_type, model_list in models.items():
                self._logger.info(
                    f"Global feature importance methods for {model_type} (fold {idx})..."
                )
                if model_type not in feature_importance_results:
                    feature_importance_results[model_type] = {}

                # Iterate through all feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._feature_importance_methods.items():
                    if not value["value"]:
                        continue

                    if (
                        feature_importance_type
                        not in feature_importance_results[model_type]
                    ):
                        feature_importance_results[model_type][
                            feature_importance_type
                        ] = []

                    if feature_importance_type == "Permutation Importance":
                        # Run Permutation Importance
                        permutation_importance_df = calculate_permutation_importance(
                            model=model_list[idx],
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
                            feature_importance_type=f"{feature_importance_type}_fold_{idx}",
                            experiment_name=self._exec_opt.experiment_name,
                            fi_opt=self._fi_opt,
                            plot_opt=self._plot_opt,
                            logger=self._logger,
                        )
                        feature_importance_results[model_type][
                            feature_importance_type
                        ].append(permutation_importance_df)

                    elif feature_importance_type == "SHAP":
                        # Run SHAP
                        shap_df, _ = calculate_global_shap_values(
                            model=model_list[idx],
                            X=X,
                            shap_reduce_data=self._fi_opt.shap_reduce_data,
                            logger=self._logger,
                        )
                        fig = plot_global_shap_importance(
                            shap_values=shap_df,
                            plot_opts=self._plot_opt,
                            num_features_to_plot=self._fi_opt.num_features_to_plot,
                            title=f"{feature_importance_type} - {value['type']} - {model_type} (fold {idx})",
                        )
                        save_dir = fi_plot_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(save_dir)
                        fig.savefig(
                            save_dir
                            / f"{feature_importance_type}-{value['type']}-{model_type}-fold_{idx}-bar.png"
                        )
                        feature_importance_results[model_type][
                            feature_importance_type
                        ].append(shap_df)

        return feature_importance_results

    def _local_feature_importance(self, models, data: pd.DataFrame):
        """
        Calculate local feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            data (pd.DataFrame): The data to interpret.
            For local interpretation, the entire data is used.
        Returns:
            dict: Dictionary of feature importance results.
        """
        # Get data features
        X = data.iloc[:, :-1]

        # Load the ml_metrics
        path_to_metrics = ml_metrics_mean_std_path(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        # Load the metrics from the file
        with open(path_to_metrics, "r") as f:
            metrics_dict = json.load(f)

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

                # Get the index for the model closest to the mean performance
                closest_index = find_mean_model_index(metrics_dict, model_type)

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
                                model[closest_index],
                                X,
                                self._exec_opt.problem_type,
                                self._logger,
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

    def _stack_importances(
        self, importances: dict[str, dict[str, list[pd.DataFrame]]]
    ) -> pd.DataFrame:
        """Stack and normalise feature importance results from different methods.

        This function processes feature importance results through these steps:
            - For each model:
           - For each importance type (e.g., SHAP, Permutation):
              - Concatenate all fold results vertically into a single DataFrame
              - Min-max normalise the importance scores to [0,1] range
           - Concatenate all normalised importance types horizontally

        Args:
            importances: Nested dictionary structure:
                - First level: Model name -> Dictionary of importance types
                - Second level: Importance type -> List of DataFrames (one per fold)
                Each DataFrame contains feature importance scores

        Returns:
            Dictionary mapping model names to their stacked importances.
            Each DataFrame has features as rows and importance methods as columns,
            with normalised importance scores as values.

        Example:
            Input structure:
            {
                'model1': {
                    'SHAP': [fold1_df, fold2_df],
                    'Permutation': [fold1_df, fold2_df]
                }
            }

            Output structure:
            {
                'model1': DataFrame(
                    columns=['SHAP', 'Permutation'],
                    index=[feature1, feature2, ...]
                )
            }
        """
        stack_importances = {}
        for model_name, importance_dict in importances.items():
            importance_type_df_list = []
            for importances_dfs in importance_dict.values():
                importance_df = pd.concat(importances_dfs, axis=0)
                importance_df = (importance_df - importance_df.min()) / (
                    importance_df.max() - importance_df.min()
                )
                importance_type_df_list.append(importance_df)

            stack_importances[model_name] = pd.concat(importance_type_df_list, axis=1)

        return stack_importances
