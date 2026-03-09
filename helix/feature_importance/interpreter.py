import os
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import pandas as pd
from joblib import Parallel, delayed

from helix.options.data import DataOptions
from helix.options.enums import FeatureImportanceTypes, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.file_paths import (
    fi_plot_dir,
    fi_result_dir,
    helix_experiments_base_dir,
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
from helix.services.plotting import (
    plot_bar_chart,
    plot_global_shap_importance,
    plot_local_shap_importance,
    plot_most_important_feats_violin,
    plot_permutation_importance,
)
from helix.utils.logging_utils import Logger
from helix.utils.plotting import close_figure
from helix.utils.utils import create_directory

_MAX_CPUS = max(cpu_count() - 1, 1)


class FeatureImportanceEstimator:
    """
    Interpreter class to interpret the model results.

    """

    def __init__(
        self,
        fi_opt: FeatureImportanceOptions,
        exec_opt: ExecutionOptions,
        plot_opt: PlottingOptions,
        data_opt: DataOptions,
        data_path: Path,
        logger: Logger | None = None,
        n_cpus: int = _MAX_CPUS,
    ) -> None:
        self._fi_opt = fi_opt
        self._logger = logger
        self._exec_opt = exec_opt
        self._plot_opt = plot_opt
        self._global_importance_methods = self._fi_opt.global_importance_methods
        self._data_opt = data_opt
        self._local_importance_methods = self._fi_opt.local_importance_methods
        self._ensemble_importance_methods = self._fi_opt.feature_importance_ensemble
        self._data_path = data_path
        self._n_cpus = n_cpus

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
        global_feature_importance_results = self._global_feature_importance(
            models, data
        )
        # Create a dict[str, DataFrame] where the keys are the model names and
        # the values are min-max normalised global feature importance values
        # for those models. The data frames combine the data from all folds
        # and all global importance types.
        global_feature_importance_df_dict = self._stack_global_importances(
            global_feature_importance_results
        )
        # Compute average global importance across all folds for each model type
        self._calculate_mean_global_importance_of_folds(
            global_feature_importance_results
        )

        # Load the dataset that was used in the experiment during model training.
        # This is required to calculate local FI.
        # This can either be the raw data or the preprocessed data, if the user
        # preprocessed the data.
        local_feature_importance_results = self._local_feature_importance(models, data)
        # Create a dict[str, DataFrame] where the keys are the model names and
        # the values are normalised local feature importance values
        # for those models. The cells are the FI for each feature (columns) in each
        # sample (rows).
        # The data frames combine the data from all local feature
        # importance types by stacking them vertically.
        # n_rows = n_samples * n_models * n_fi_types * n_folds.
        local_feature_importance_df_dict = self._stack_local_importances(
            local_feature_importance_results
        )
        # Compute average local importance across all folds for each model type
        self._calculate_mean_local_importance_of_folds(local_feature_importance_results)

        # Calculate ensemble FI from stacked global FI
        ensemble_feature_importance_results = self._ensemble_feature_importance(
            global_feature_importance_df_dict
        )
        self._logger.info("-------- End of feature importance logging--------")

        return (
            global_feature_importance_df_dict,
            local_feature_importance_df_dict,
            ensemble_feature_importance_results,
        )

    def _global_feature_importance(self, models: dict, data: TabularData):
        """
        Calculate global feature importance for each model and for each fold.
        This is repeated for each selected global feature importance method.
        Parameters:
            models (dict): Dictionary of models.
            data (TabularData): The data to interpret.
        Returns:
            dict: Dictionary of feature importance results.
        """
        feature_importance_results = {}
        if not any(
            sub_dict["value"] for sub_dict in self._global_importance_methods.values()
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
                ) in self._global_importance_methods.items():
                    # This determines whether or not the feature importance method
                    # has been requested by the user.
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

    def _local_feature_importance(self, models: dict, data: TabularData):
        """
        Calculate local feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            data (TabularData): The data to interpret.
            For local interpretation, the entire data is used.
        Returns:
            dict: Dictionary of feature importance results.
        """

        def _single_local_fi(
            local_importance_method: str,
            model_type: str,
            model,  # The type can vary but it's the ML model
            X: pd.DataFrame,
            y: pd.DataFrame,
            fold: int,
            plot_dir: Path,
            logger: Logger,
            problem_type: ProblemTypes,
            plot_opt: PlottingOptions,
            num_features_to_plot: int,
        ) -> tuple[pd.DataFrame, int, str, str]:
            fold_number = fold + 1

            if local_importance_method == FeatureImportanceTypes.LIME:
                importance_df = calculate_lime_values(
                    model,
                    X,
                    problem_type,
                    logger,
                )
                fig = plot_most_important_feats_violin(
                    df=importance_df,
                    plot_opts=plot_opt,
                    num_features_to_plot=num_features_to_plot,
                    title=f"{local_importance_method} - {model_type} (fold {fold_number})",
                )
                fig.savefig(
                    plot_dir
                    / f"local-{local_importance_method}-{model_type}-violin (fold {fold_number}).png"
                )
                close_figure(fig)

                importance_df = pd.concat([importance_df, y], axis=1)

            if local_importance_method == FeatureImportanceTypes.SHAP:
                importance_df, shap_values = calculate_local_shap_values(
                    model=model,
                    X=X,
                    logger=logger,
                )
                fig = plot_local_shap_importance(
                    shap_values=shap_values,
                    plot_opts=plot_opt,
                    num_features_to_plot=num_features_to_plot,
                    title=f"{local_importance_method} - local - {model_type} (fold {fold_number})",
                )
                fig.savefig(
                    plot_dir
                    / f"local-{local_importance_method}-{model_type}-beeswarm (fold {fold_number}).png"
                )
                close_figure(fig)

                importance_df = pd.concat([importance_df, y], axis=1)

            return importance_df, fold, model_type, local_importance_method

        # Outer dict keys are the model types, inner dict keys are feature importance types.
        # Inner dict values are lists of local feature importance dataframes.
        feature_importance_results: dict[str, dict[str, list[pd.DataFrame]]] = {}
        if not any(
            sub_dict["value"] for sub_dict in self._local_importance_methods.values()
        ):
            self._logger.info("No local feature importance methods selected")
            self._logger.info("Skipping local feature importance methods")
            return feature_importance_results

        # Set up results and plot directories
        results_dir = fi_result_dir(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        create_directory(results_dir)  # will create the directory if it doesn't exist
        plot_dir = fi_plot_dir(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        create_directory(plot_dir)  # will create the directory if it doesn't exist

        # Generate combinations of bootstraps * model type * feature importance types
        bootstraps = list(range(len(data.X_train)))
        model_types = list(models.keys())
        local_fi_types = [
            key
            for key, value in self._local_importance_methods.items()
            if value["value"]
        ]

        combinations = list(product(bootstraps, model_types, local_fi_types))

        # In parallel, iterate through the combinations to produce all the local
        # feature importances for all bootstraps of all models
        interpretation_start = time()
        results = Parallel(n_jobs=self._n_cpus, prefer="processes")(
            delayed(_single_local_fi)(
                local_fi_type,
                model_type,
                models[model_type][bootstrap],
                data.X_train[bootstrap],
                data.y_train[bootstrap],
                bootstrap,
                plot_dir,
                self._logger,
                self._exec_opt.problem_type,
                self._plot_opt,
                self._fi_opt.num_features_to_plot,
            )
            for bootstrap, model_type, local_fi_type in combinations
        )

        interpretation_end = time()
        elapsed = interpretation_end - interpretation_start
        hours = int(elapsed) // 3600
        minutes = (int(elapsed) % 3600) // 60
        seconds = int(elapsed) % 60
        # Create format hh:mm:ss
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self._logger.info(f"Local feature importance completed in {time_str}")

        # Extract the results from the parallel process
        for importance_df, bootstrap, model_type, importance_type in results:
            if model_type not in feature_importance_results:
                feature_importance_results[model_type] = {}
            if importance_type not in feature_importance_results[model_type]:
                feature_importance_results[model_type][importance_type] = []
            feature_importance_results[model_type][importance_type].append(
                importance_df
            )
            # Save the FI results for each fold
            importance_df.to_csv(
                results_dir / f"local-{importance_type} (fold {bootstrap + 1}).csv"
            )

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

        if not any(self._ensemble_importance_methods.values()):
            self._logger.info("No ensemble feature importance method selected")
            self._logger.info("Skipping ensemble feature importance analysis")
        else:
            self._logger.info("Ensemble feature importance methods...")
            for ensemble_type, value in self._ensemble_importance_methods.items():
                # This determines whether or not a feature importance method
                # has been requested by the user.
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

    def _stack_global_importances(
        self, importances: dict[str, dict[str, list[pd.DataFrame]]]
    ) -> dict[str, pd.DataFrame]:
        """Stack and normalise feature importance results from different methods.

        This function processes feature importance results through these steps:
            - For each model:
            - For each importance type (e.g., SHAP, Permutation importance):
                - Concatenate all fold results vertically into a single DataFrame
                - Normalise the importance scores to [0,1] range
                - The values of the cells for the normalised FI for the feature in that sample
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

    def _stack_local_importances(
        self, importances: dict[str, dict[str, list[pd.DataFrame]]]
    ) -> dict[str, pd.DataFrame]:
        """Stack and normalise feature importance results from different methods.

        This function processes feature importance results through these steps:
            - For each model:
            - For each importance type (e.g., SHAP, LIME):
                - Concatenate all fold results vertically into a single DataFrame
                - Normalise the importance scores to [0,1] range
                - The values of the cells for the normalised FI for the feature in that sample
            - Concatenate all normalised importance types vertically

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

            # concat on axis = 0 to preserve unique column names
            # results in n_rows = n_samples * n_models * n_importances
            # and n_columns = n_features + n_targets
            stack_importances[model_name] = pd.concat(importance_type_df_list, axis=0)

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

    def _calculate_mean_local_importance_of_folds(
        self, local_importances_dict: dict[str, dict[str, list[pd.DataFrame]]]
    ):
        """Calculate the mean local importance for all folds through which a model was trained.
        The all-folds mean for each model and importance type is saved along with a plot.

        Args:
            local_importances_dict (dict[str, dict[str, list[pd.DataFrame]]]):
                The local importance results containing the importance calculations for
                each model type, importance type and folds.
        """
        results_dir = fi_result_dir(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        create_directory(results_dir)  # will create the directory if it doesn't exist
        plot_dir = fi_plot_dir(
            helix_experiments_base_dir() / self._exec_opt.experiment_name
        )
        create_directory(plot_dir)  # will create the directory if it doesn't exist
        for model_name, lfi_dict in local_importances_dict.items():
            for fi_type, importance_dfs in lfi_dict.items():
                # Each df in `importance_dfs` is the local feature importance
                # for each fold during model training. These are stacked vertically
                # creating an index with repeating values in 0..n_samples.
                # `groupby(level=0).mean()` combines these such that the final length
                # of `fold_mean_df` is the same as the original data
                # and the values in the cells is the mean local feature importance from
                # all folds.
                fold_mean_df = pd.concat(importance_dfs).groupby(level=0).mean()
                fold_mean_df.to_csv(
                    results_dir / f"local-{fi_type}-{model_name}-all-folds-mean.csv"
                )
                fig = plot_most_important_feats_violin(
                    df=fold_mean_df.iloc[:, :-1],
                    plot_opts=self._plot_opt,
                    num_features_to_plot=self._fi_opt.num_features_to_plot,
                    title=f"{fi_type} - {model_name} - all folds mean",
                )
                fig.savefig(
                    plot_dir / f"local-{fi_type}-{model_name}-all-folds-mean.png"
                )
                close_figure(fig)
