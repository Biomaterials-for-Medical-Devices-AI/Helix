from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from biofefi.machine_learning.data import TabularData
from biofefi.options.choices import (
    CLASSIFICATION_METRICS,
    MODEL_PROBLEM_CHOICES,
    REGRESSION_METRICS,
)
from biofefi.options.enums import (
    DataSplitMethods,
    Metrics,
    Normalisations,
    ProblemTypes,
)
from biofefi.services.metrics import get_metrics
from biofefi.services.ml_models import get_models
from biofefi.utils.logging_utils import Logger


class Learner:
    """
    Learner class encapsulates the logic for initialising,
    training, and evaluating machine learning models.

    Args:
        - model_types (dict): Dictionary containing model types
        and their parameters.
        - problem_type (ProblemTypes): Type of problem (
        classification or regression).
        - data_split (dict): Dictionary containing data split type and parameters
        - normalization (Normalisations): Type of normalization to apply to the data
        - n_bootstraps (int): Number of bootstrap samples to generate
        - logger (Logger): Logger object to log messages
    """

    def __init__(
        self,
        model_types: dict,
        problem_type: ProblemTypes,
        data_split: dict,
        normalization: Normalisations,
        n_bootstraps: int,
        logger: Logger | None = None,
    ) -> None:
        self._logger = logger
        self._model_types = model_types
        self._problem_type = problem_type
        self._data_split = data_split
        self._normalization = normalization
        self._n_bootstraps = n_bootstraps
        self._metrics = get_metrics(self._problem_type, logger=self._logger)

    def _process_data_for_bootstrap(self, data, i):
        """
        Extracts and converts the datasets for the given samples
        if it is not a numpy array.

        Args:
            - data: Structured object containing `X_train`,
            `X_test`, `y_train`, `y_test`.
            - i (int): Index of the sample.

        Returns:
            - Tuple: Processed `X_train`, `X_test`, `y_train`,
            `y_test` for the given sample.
        """
        try:
            X_train = (
                data.X_train[i].to_numpy()
                if isinstance(data.X_train[i], pd.DataFrame)
                else data.X_train[i]
            )
            X_test = (
                data.X_test[i].to_numpy()
                if isinstance(data.X_test[i], pd.DataFrame)
                else data.X_test[i]
            )
            y_train = (
                data.y_train[i].to_numpy()
                if isinstance(data.y_train[i], pd.DataFrame)
                else data.y_train[i]
            )
            y_test = (
                data.y_test[i].to_numpy()
                if isinstance(data.y_test[i], pd.DataFrame)
                else data.y_test[i]
            )
        except Exception as e:
            raise ValueError(
                f"Error processing bootstrap data during train test split  {e}"
            )

        return X_train, X_test, y_train, y_test

    def fit(self, data: Tuple):
        """
        Fits machine learning models to the given data
        and applies holdout strategy with bootstrap sampling.

        Args:
            - data (Tuple): Tuple containing training and testing data
            for each bootstrap sample.

        Returns:
            - res (Dict): Dictionary containing model predictions for
            each bootstrap sample.
            - metric_res (Dict): Dictionary containing metric values for
            each bootstrap sample.
            - metric_res_stats (Dict): Dictionary containing average and
            standard deviation of metric values across bootstrap samples.
            - trained_models (Dict): Dictionary containing
            trained models for each model type.
        """
        self._models = get_models(
            self._model_types, self._problem_type, logger=self._logger
        )
        if self._data_split["type"] == DataSplitMethods.Holdout:
            res, metric_res, metric_res_stats, trained_models = self._fit_holdout(data)
            return res, metric_res, metric_res_stats, trained_models

        elif self._data_split["type"] == DataSplitMethods.KFold:
            res, metric_res, metric_res_stats, trained_models = self._fit_kfold(data)
            return res, metric_res, metric_res_stats, trained_models

    def _fit_holdout(self, data: Tuple) -> None:
        """
        Trains models using a holdout strategy with bootstrap sampling.

        Args:
            - data (Tuple): Tuple containing training and testing data
            for each bootstrap sample.

        Returns:
            - res (Dict): Dictionary containing model predictions for
            each bootstrap sample.
            - metric_res (Dict): Dictionary containing metric values for
            each bootstrap sample.
            - metric_res_stats (Dict): Dictionary containing average and
            standard deviation of metric values across bootstrap samples.
            - trained_models (Dict): Dictionary containing
            trained models for each model type.
        """
        self._logger.info("Fitting holdout with bootstrapped datasets...")
        res = {}
        metric_res = {}
        trained_models = {model_name: [] for model_name in self._models.keys()}

        for i in range(self._n_bootstraps):
            self._logger.info(f"Processing bootstrap sample {i+1}...")
            X_train, X_test, y_train, y_test = self._process_data_for_bootstrap(data, i)

            res[i] = {}
            for model_name, model in self._models.items():
                res[i][model_name] = {}
                self._logger.info(f"Fitting {model_name} for bootstrap sample {i+1}...")
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                res[i][model_name]["y_pred_train"] = y_pred_train
                y_pred_test = model.predict(X_test)
                res[i][model_name]["y_pred_test"] = y_pred_test
                if self._problem_type == ProblemTypes.Classification:
                    y_pred_probs_train = model.predict_proba(X_train)
                    y_pred_probs_test = model.predict_proba(X_test)
                else:
                    y_pred_probs_train = None
                    y_pred_probs_test = None
                if model_name not in metric_res:
                    metric_res[model_name] = []
                metric_res[model_name].append(
                    _evaluate(
                        model_name,
                        self._metrics,
                        y_train,
                        y_pred_train,
                        y_pred_probs_train,
                        y_test,
                        y_pred_test,
                        y_pred_probs_test,
                        self._logger,
                    )
                )
                trained_models[model_name].append(model)
        metric_res_stats = _compute_metrics_statistics(metric_res)
        return res, metric_res, metric_res_stats, trained_models

    def _fit_kfold(self, data: Tuple) -> None:
        self._logger.info("Fitting cross validation datasets...")
        res = {}
        metric_res = {}
        trained_models = {model_name: [] for model_name in self._models.keys()}

        for i in range(self._data_split["n_splits"]):
            self._logger.info(f"Processing test fold sample {i+1}...")
            X_train, X_test = data.X_train[i], data.X_test[i]
            y_train, y_test = data.y_train[i], data.y_test[i]

            res[i] = {}
            for model_name, model in self._models.items():
                res[i][model_name] = {}
                self._logger.info(f"Fitting {model_name} for test fold sample {i+1}...")
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                res[i][model_name]["y_pred_train"] = y_pred_train
                y_pred_test = model.predict(X_test)
                res[i][model_name]["y_pred_test"] = y_pred_test
                if self._problem_type == ProblemTypes.Classification:
                    y_pred_probs_train = model.predict_proba(X_train)
                    y_pred_probs_test = model.predict_proba(X_test)
                else:
                    y_pred_probs_train = None
                    y_pred_probs_test = None
                if model_name not in metric_res:
                    metric_res[model_name] = []
                metric_res[model_name].append(
                    _evaluate(
                        model_name,
                        self._metrics,
                        y_train,
                        y_pred_train,
                        y_pred_probs_train,
                        y_test,
                        y_pred_test,
                        y_pred_probs_test,
                        self._logger,
                    )
                )
                trained_models[model_name].append(model)
        metric_res_stats = _compute_metrics_statistics(metric_res)
        return res, metric_res, metric_res_stats, trained_models


class GridSearchLearner:
    def __init__(
        self,
        model_types: dict,
        problem_type: ProblemTypes,
        data_split: dict,
        normalization: Normalisations,
        logger: Logger | None = None,
    ) -> None:
        self._logger = logger
        self._model_types: dict[str, Any] = model_types
        self._problem_type = problem_type
        self._data_split = data_split
        self._normalization = normalization
        self._metrics = get_metrics(self._problem_type, logger=self._logger)
        self._models: dict = {}

    def fit(self, data: TabularData) -> tuple[dict, dict, dict, dict]:
        """Fit models to the data using Grid Search with cross validation. Evaluates them
        and returns metrics and statistics with the models.

        Args:
            data (TabularData): The data to fit the models with.

        Returns:
            res (dict): Dictionary containing model predictions for
            each bootstrap sample.
            metric_res (dict): Dictionary containing metric values for
            each bootstrap sample.
            metric_res_stats (dict): Dictionary containing average and
            standard deviation of metric values across bootstrap samples.
            trained_models (dict): Dictionary containing
            trained models for each model type.
        """
        self._logger.info("Fitting models using Grid Search...")
        self._models = get_models(
            self._model_types,
            self._problem_type,
            logger=self._logger,
            use_params=False,
        )
        # Extract the data
        X_train = data.X_train[0]
        X_test = data.X_test[0]
        y_train = data.y_train[0]
        y_test = data.y_test[0]

        # Make grid search compatible scorers
        metrics = (
            REGRESSION_METRICS
            if self._problem_type == ProblemTypes.Regression
            else CLASSIFICATION_METRICS
        )
        scorers = {key: make_scorer(value) for key, value in metrics.items()}

        # Fit models
        res = {0: {}}
        metric_res = {}
        trained_models = {model_name: [] for model_name in self._models.keys()}
        for model_name, model in self._models.items():
            res[0][model_name] = {}
            # Set up grid search
            model = MODEL_PROBLEM_CHOICES.get(
                (model_name.lower(), self._problem_type.lower())
            )
            refit = (
                "R2" if self._problem_type == ProblemTypes.Regression else "accuracy"
            )
            gs = GridSearchCV(
                estimator=model(),
                param_grid=self._model_types[model_name]["params"],
                scoring=scorers,
                refit=refit,
                cv=self._data_split["n_splits"],
            )

            # Fit the model
            self._logger.info(f"Fitting {model_name}...")
            gs.fit(X_train, y_train)

            # Make predictions for evaluation
            y_pred_train = gs.predict(X_train)
            res[0][model_name]["y_pred_train"] = y_pred_train
            y_pred_test = gs.predict(X_test)
            res[0][model_name]["y_pred_test"] = y_pred_test
            if self._problem_type == ProblemTypes.Classification:
                y_pred_probs_train = gs.predict_proba(X_train)
                y_pred_probs_test = gs.predict_proba(X_test)
            else:
                y_pred_probs_train = None
                y_pred_probs_test = None
            if model_name not in metric_res:
                metric_res[model_name] = []
            metric_res[model_name].append(
                _evaluate(
                    model_name,
                    metrics,
                    y_train,
                    y_pred_train,
                    y_pred_probs_train,
                    y_test,
                    y_pred_test,
                    y_pred_probs_test,
                    self._logger,
                )
            )
            # append the best estimator
            trained_models[model_name].append(gs.best_estimator_)
        metric_res_stats = _compute_metrics_statistics(metric_res)
        return res, metric_res, metric_res_stats, trained_models


def _evaluate(
    model_name: str,
    metrics: dict,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_probs_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_pred_probs_test: np.ndarray,
    logger: object,
) -> dict:
    """
    Evaluates the performance of a model using specified metrics.

    Args:
        - model_name (str): Name of the model being evaluated.
        - metrics (dict): The metrics to use in evaluation.
        - y_train (np.ndarray): True labels for the training set.
        - y_pred_train (np.ndarray): Predicted labels for the training set.
        - y_pred_probs_train (np.ndarray): Predicted probabilities for the training set.
        - y_test (np.ndarray): True labels for the test set.
        - y_pred_test (np.ndarray): Predicted labels for the test set.
        - y_pred_probs_test (np.ndarray): Predicted probabilities for the test set.
        - logger (object): The logger.
    """
    logger.info(f"Evaluating {model_name}...")
    eval_res = {}
    for metric_name, metric in metrics.items():
        eval_res[metric_name] = {}
        logger.info(f"Evaluating {model_name} on {metric_name}...")
        if y_pred_probs_train is None or y_pred_probs_train.shape[1] < 3:
            metric_train = metric(y_train, y_pred_train)
            metric_test = metric(y_test, y_pred_test)
        else:
            if metric_name == Metrics.Accuracy:
                metric_train = metric(y_train, y_pred_train)
                metric_test = metric(y_test, y_pred_test)
            elif metric_name == Metrics.ROC_AUC:
                metric_train = metric(y_train, y_pred_probs_train, multi_class="ovr")
                metric_test = metric(y_test, y_pred_probs_test, multi_class="ovr")
            else:
                metric_train = metric(y_train, y_pred_train, average="micro")
                metric_test = metric(y_test, y_pred_test, average="micro")
        eval_res[metric_name]["train"] = {
            "value": metric_train,
        }
        eval_res[metric_name]["test"] = {
            "value": metric_test,
        }
    return eval_res


def _compute_metrics_statistics(metric_res: dict) -> dict:
    """
    Compute metric statistics for each model.

    Args:
        - metric_res (dict): Dictionary containing metric values
        for each bootstrap sample.

    Returns:
        - dict: Dictionary containing metric statistics for
        each model.
    """
    statistics = {}

    for model_name, metrics_list in metric_res.items():
        # Initialize dictionaries to store metric values for train and test sets
        train_metrics = {}
        test_metrics = {}

        # Aggregate metric values from each bootstrap sample
        for metrics in metrics_list:
            for metric_name, metric_values in metrics.items():
                if metric_name not in train_metrics:
                    train_metrics[metric_name] = []
                    test_metrics[metric_name] = []

                train_metrics[metric_name].append(metric_values["train"]["value"])
                test_metrics[metric_name].append(metric_values["test"]["value"])

        # Compute average and standard deviation for each metric
        statistics[model_name] = {"train": {}, "test": {}}
        for metric_name in train_metrics.keys():
            train_values = np.array(train_metrics[metric_name])
            test_values = np.array(test_metrics[metric_name])

            statistics[model_name]["train"][metric_name] = {
                "mean": np.mean(train_values),
                "std": np.std(train_values),
            }
            statistics[model_name]["test"][metric_name] = {
                "mean": np.mean(test_values),
                "std": np.std(test_values),
            }

    return statistics
