import argparse
import os

from biofefi.feature_importance.call_methods import save_importance_results
from biofefi.feature_importance.ensemble_methods import (
    calculate_ensemble_majorityvote,
    calculate_ensemble_mean,
)
from biofefi.feature_importance.feature_importance_methods import (
    calculate_permutation_importance,
    calculate_shap_values,
    calculate_lime_values,
)


class Interpreter:
    """
    Interpreter class to interpret the model results.

    """

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._opt = opt
        self._logger = logger
        self._feature_importance_methods = self._opt.global_importance_methods
        self._local_importance_methods = self._opt.local_importance_methods
        self._feature_importance_ensemble = self._opt.feature_importance_ensemble

    def interpret(self, models, data):
        """
        Interpret the model results using the selected feature importance methods and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        """
        # Load just the first fold of the data and the first models for interpretation
        X, y = data.X_train[0], data.y_train[0]
        self._logger.info(f"-------- Start of feature importance logging--------")
        global_importance_results = self._individual_feature_importance(models, X, y)
        local_importance_results = self._local_feature_importance(models, X)
        ensemble_results = self._ensemble_feature_importance(global_importance_results)
        self._logger.info(f"-------- End of feature importance logging--------")

        return global_importance_results, local_importance_results, ensemble_results

    def _individual_feature_importance(self, models, X, y):
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

                # Run methods with TRUE values in the dictionary of feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._feature_importance_methods.items():
                    if value["value"]:
                        # Select the first model in the list - model[0]
                        if feature_importance_type == "Permutation Importance":
                            # Run Permutation Importance -
                            permutation_importance_df = (
                                calculate_permutation_importance(
                                    model[0], X, y, self._opt, self._logger
                                )
                            )
                            save_importance_results(
                                permutation_importance_df,
                                model_type,
                                value["type"],
                                feature_importance_type,
                                self._opt,
                                self._logger,
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = permutation_importance_df

                        if feature_importance_type == "SHAP":
                            # Run SHAP
                            shap_df, shap_values = calculate_shap_values(
                                model[0], X, value["type"], self._opt, self._logger
                            )
                            save_importance_results(
                                shap_df,
                                model_type,
                                value["type"],
                                feature_importance_type,
                                self._opt,
                                self._logger,
                                shap_values,
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
                                model[0], X, self._opt, self._logger
                            )
                            save_importance_results(
                                lime_importance_df,
                                model_type,
                                value["type"],
                                feature_importance_type,
                                self._opt,
                                self._logger,
                            )
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = lime_importance_df

                        if feature_importance_type == "SHAP":
                            # Run SHAP
                            shap_df, shap_values = calculate_shap_values(
                                model[0], X, value["type"], self._opt, self._logger
                            )
                            save_importance_results(
                                shap_df,
                                model_type,
                                value["type"],
                                feature_importance_type,
                                self._opt,
                                self._logger,
                                shap_values,
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
                            feature_importance_results, self._opt, self._logger
                        )
                        save_importance_results(
                            mean_results,
                            None,
                            None,
                            ensemble_type,
                            self._opt,
                            self._logger,
                        )
                        ensemble_results[ensemble_type] = mean_results

                    if ensemble_type == "Majority Vote":
                        # Calculate majority vote of feature importance results
                        majority_vote_results = calculate_ensemble_majorityvote(
                            feature_importance_results, self._opt, self._logger
                        )
                        save_importance_results(
                            majority_vote_results,
                            None,
                            None,
                            ensemble_type,
                            self._opt,
                            self._logger,
                        )
                        ensemble_results[ensemble_type] = majority_vote_results

            self._logger.info(
                f"Ensemble feature importance results: {os.linesep}{ensemble_results}"
            )

        return ensemble_results
