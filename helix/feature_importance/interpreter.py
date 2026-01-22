import json
import os
from pathlib import Path

import pandas as pd

from helix.options.enums import FeatureImportanceTypes, Metrics, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.file_paths import (
    fi_plot_dir,
    fi_result_dir,
    helix_experiments_base_dir,
    ml_metrics_full_path,
    ml_metrics_mean_std_path,
)
from helix.options.plotting import PlottingOptions
from helix.services.data import TabularData, read_data
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
from helix.services.metrics import find_mean_model_index
from helix.services.plotting import (
    plot_bar_chart,
    plot_global_shap_importance,
    plot_lime_importance,
    plot_local_shap_importance,
    plot_permutation_importance,
)
from helix.utils.logging_utils import Logger
from helix.utils.plotting import close_figure
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
        data_path: Path,
        logger: Logger | None = None,
    ) -> None:
        self._fi_opt = fi_opt
        self._logger = logger
        self._exec_opt = exec_opt
        self._plot_opt = plot_opt
        self._feature_importance_methods = self._fi_opt.global_importance_methods
        self._local_importance_methods = self._fi_opt.local_importance_methods
        self._feature_importance_ensemble = self._fi_opt.feature_importance_ensemble
        self._data_path = data_path

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
        self._logger.info("-------- Start of feature importance logging--------")
        global_importance_results = self._global_feature_importance(models, data)
        global_importance_df_dict = self._stack_importances(global_importance_results)
        # Compute average global importance across all folds for each model type
        self._calculate_mean_global_importance_of_folds(global_importance_results)

        # Load the total dataset for the local importance
        total_df = read_data(self._data_path, self._logger)
        local_importance_results = self._local_feature_importance(models, total_df)
        ensemble_results = self._ensemble_feature_importance(global_importance_df_dict)
        self._logger.info("-------- End of feature importance logging--------")

        return global_importance_df_dict, local_importance_results, ensemble_results

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
                    f"Global feature importance methods for {model_type} (fold {idx + 1})..."
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

                    if (
                        feature_importance_type
                        == FeatureImportanceTypes.PermutationImportance
                    ):
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
                        results_dir = fi_result_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(results_dir)
                        permutation_importance_df.to_csv(
                            results_dir
                            / f"global-{feature_importance_type}-{model_type}-fold-{idx + 1}.csv"
                        )
                        fig = plot_permutation_importance(
                            permutation_importance_df,
                            self._plot_opt,
                            self._fi_opt.num_features_to_plot,
                            f"{feature_importance_type} - {model_type} (fold {idx + 1})",
                        )
                        plot_dir = fi_plot_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(plot_dir)
                        fig.savefig(
                            plot_dir
                            / f"{feature_importance_type}-{value['type']}-{model_type}-fold-{idx + 1}-bar.png"
                        )
                        close_figure(fig)
                        feature_importance_results[model_type][
                            feature_importance_type
                        ].append(permutation_importance_df)

                    elif feature_importance_type == FeatureImportanceTypes.SHAP:
                        # Run SHAP
                        shap_df, _ = calculate_global_shap_values(
                            model=model_list[idx],
                            X=X,
                            logger=self._logger,
                        )
                        results_dir = fi_result_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(results_dir)
                        shap_df.to_csv(
                            results_dir
                            / f"global-{feature_importance_type}-{model_type}-fold-{idx + 1}.csv"
                        )
                        fig = plot_global_shap_importance(
                            shap_values=shap_df,
                            plot_opts=self._plot_opt,
                            num_features_to_plot=self._fi_opt.num_features_to_plot,
                            title=f"{feature_importance_type} - {value['type']} - {model_type} (fold {idx + 1})",
                        )
                        plot_dir = fi_plot_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(plot_dir)
                        fig.savefig(
                            plot_dir
                            / f"{feature_importance_type}-{value['type']}-{model_type}-fold-{idx + 1}-bar.png"
                        )
                        close_figure(fig)
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
        # Get the targets
        y = data.iloc[:, -1]

        # Determine which metric to use
        if self._exec_opt.problem_type == ProblemTypes.Regression:
            metric = Metrics.R2.value
        elif self._exec_opt.problem_type == ProblemTypes.Classification:
            metric = Metrics.ROC_AUC.value

        # Load the full ml_metrics
        path_to_metrics = ml_metrics_full_path(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        # Load the metrics mean and std from the file
        with open(path_to_metrics, "r") as f:
            metrics_full = json.load(f)

        # Load the ml_metrics mean std
        path_to_metrics = ml_metrics_mean_std_path(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        # Load the metrics mean and std from the file
        with open(path_to_metrics, "r") as f:
            metrics_mean_std = json.load(f)

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
                closest_index = find_mean_model_index(
                    metrics_full, metrics_mean_std, metric
                )

                # Run methods with TRUE values in the dictionary of feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._local_importance_methods.items():
                    if value["value"]:
                        # Select the first model in the list - model[0]
                        if feature_importance_type == FeatureImportanceTypes.LIME:
                            # Run Permutation Importance
                            lime_importance_df = calculate_lime_values(
                                model[closest_index],
                                X,
                                self._exec_opt.problem_type,
                                self._logger,
                            )
                            results_dir = fi_result_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(results_dir)
                            lime_importance_df.to_csv(
                                results_dir / f"local-{feature_importance_type}.csv"
                            )
                            fig = plot_lime_importance(
                                df=lime_importance_df,
                                plot_opts=self._plot_opt,
                                num_features_to_plot=self._fi_opt.num_features_to_plot,
                                title=f"{feature_importance_type} - {model_type}",
                            )
                            plot_dir = fi_plot_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(
                                plot_dir
                            )  # will create the directory if it doesn't exist
                            fig.savefig(
                                plot_dir
                                / f"local-{feature_importance_type}-{model_type}-violin.png"
                            )
                            close_figure(fig)

                            # Add class to local feature importance for fuzzy
                            lime_importance_df = pd.concat(
                                [lime_importance_df, y], axis=1
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = lime_importance_df

                        if feature_importance_type == FeatureImportanceTypes.SHAP:
                            # Run SHAP
                            shap_df, shap_values = calculate_local_shap_values(
                                model=model[closest_index],
                                X=X,
                                logger=self._logger,
                            )
                            results_dir = fi_result_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(results_dir)
                            shap_df.to_csv(
                                results_dir / f"local-{feature_importance_type}.csv"
                            )
                            fig = plot_local_shap_importance(
                                shap_values=shap_values,
                                plot_opts=self._plot_opt,
                                num_features_to_plot=self._fi_opt.num_features_to_plot,
                                title=f"{feature_importance_type} - {value['type']} - {model_type}",
                            )
                            plot_dir = fi_plot_dir(
                                helix_experiments_base_dir()
                                / self._exec_opt.experiment_name
                            )
                            create_directory(
                                plot_dir
                            )  # will create the directory if it doesn't exist
                            fig.savefig(
                                plot_dir
                                / f"local-{feature_importance_type}-{value['type']}-{model_type}-beeswarm.png"
                            )
                            close_figure(fig)

                            # Add class to local feature importance for fuzzy
                            shap_df = pd.concat([shap_df, y], axis=1)
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
                    if ensemble_type == FeatureImportanceTypes.Mean:
                        # Calculate mean of feature importance results
                        mean_results, mean_results_std = calculate_ensemble_mean(
                            feature_importance_results, self._logger
                        )
                        results_dir = fi_result_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(results_dir)
                        mean_results.to_csv(
                            results_dir / f"ensemble-{ensemble_type}.csv"
                        )
                        fig = plot_bar_chart(
                            df=mean_results,
                            sort_key="Mean Importance",
                            plot_opts=self._plot_opt,
                            title=f"Ensemble {ensemble_type}",
                            x_label="Feature",
                            y_label="Importance",
                            n_features=self._fi_opt.num_features_to_plot,
                            error_bars=mean_results_std,
                        )
                        plot_dir = fi_plot_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(
                            plot_dir
                        )  # will create the directory if it doesn't exist
                        fig.savefig(plot_dir / f"ensemble-{ensemble_type}.png")
                        close_figure(fig)
                        ensemble_results[ensemble_type] = mean_results

                    if ensemble_type == FeatureImportanceTypes.MajorityVote:
                        # Calculate majority vote of feature importance results
                        majority_vote_results, majority_vote_results_std = (
                            calculate_ensemble_majorityvote(
                                feature_importance_results, self._logger
                            )
                        )
                        results_dir = fi_result_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(results_dir)
                        majority_vote_results.to_csv(
                            results_dir / f"ensemble-{ensemble_type}.csv"
                        )
                        fig = plot_bar_chart(
                            df=majority_vote_results,
                            sort_key="Majority Vote Importance",
                            plot_opts=self._plot_opt,
                            title=f"Ensemble {ensemble_type}",
                            x_label="Feature",
                            y_label="Importance",
                            n_features=self._fi_opt.num_features_to_plot,
                            error_bars=majority_vote_results_std,
                        )
                        plot_dir = fi_plot_dir(
                            helix_experiments_base_dir()
                            / self._exec_opt.experiment_name
                        )
                        create_directory(
                            plot_dir
                        )  # will create the directory if it doesn't exist
                        fig.savefig(plot_dir / f"ensemble-{ensemble_type}.png")
                        close_figure(fig)
                        ensemble_results[ensemble_type] = majority_vote_results

            self._logger.info(
                f"Ensemble feature importance results: {os.linesep}{ensemble_results}"
            )

        return ensemble_results

    def _stack_importances(
        self, importances: dict[str, dict[str, list[pd.DataFrame]]]
    ) -> dict[str, pd.DataFrame]:
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

    def _calculate_mean_global_importance_of_folds(
        self, global_importances_dict: dict[str, dict[str, list[pd.DataFrame]]]
    ):
        """Calculate the mean global importance for all folds through which a model was trained.
        The all-folds mean for each model and importance type is saved along with a plot.

        Args:
            global_importances_dict (dict[str, dict[str, list[pd.DataFrame]]]):
                The global importance results containing the importance calculations for
                each model type, importance type and folds.
        """
        for model_name, gfi_dict in global_importances_dict.items():
            for fi_type, importance_dfs in gfi_dict.items():
                fold_mean_df = pd.concat(importance_dfs).groupby(level=0).mean()
                fold_std_df = pd.concat(importance_dfs).groupby(level=0).std()
                results_dir = fi_result_dir(
                    helix_experiments_base_dir() / self._exec_opt.experiment_name
                )
                create_directory(results_dir)
                fold_mean_df.to_csv(
                    results_dir / f"global-{fi_type}-{model_name}-all-folds-mean.csv"
                )
                fig = plot_bar_chart(
                    df=fold_mean_df,
                    sort_key=fold_mean_df.columns[
                        0
                    ],  # there's one column which is the FI type
                    plot_opts=self._plot_opt,
                    title=f"{fi_type} - {model_name} - all folds mean",
                    x_label="Feature",
                    y_label="Importance",
                    n_features=self._fi_opt.num_features_to_plot,
                    error_bars=fold_std_df,
                )
                plot_dir = fi_plot_dir(
                    helix_experiments_base_dir() / self._exec_opt.experiment_name
                )
                create_directory(
                    plot_dir
                )  # will create the directory if it doesn't exist
                fig.savefig(plot_dir / f"{fi_type}-{model_name}-all-folds-mean.png")
