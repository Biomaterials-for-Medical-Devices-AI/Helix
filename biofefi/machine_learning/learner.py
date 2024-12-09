from typing import Dict, Tuple

import numpy as np

from biofefi.machine_learning.get_models import get_models
from biofefi.machine_learning.metrics import get_metrics
from biofefi.options.enums import DataSplitMethods, Normalisations, ProblemTypes
from biofefi.utils.logging_utils import Logger


class Learner:
    """
    Learner class
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

    def fit(self, data: Tuple) -> None:
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
        self._logger.info("Fitting holdout with bootstrapped datasets...")
        res = {}
        metric_res = {}
        trained_models = {model_name: [] for model_name in self._models.keys()}

        for i in range(self._n_bootstraps):
            self._logger.info(f"Processing bootstrap sample {i+1}...")
            X_train, X_test = data.X_train[i], data.X_test[i]
            y_train, y_test = data.y_train[i], data.y_test[i]

            res[i] = {}
            for model_name, model in self._models.items():
                res[i][model_name] = {}
                self._logger.info(f"Fitting {model_name} for bootstrap sample {i+1}...")
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                res[i][model_name]["y_pred_train"] = y_pred_train
                y_pred_test = model.predict(X_test)
                res[i][model_name]["y_pred_test"] = y_pred_test
                if model_name not in metric_res:
                    metric_res[model_name] = []
                metric_res[model_name].append(
                    self._evaluate(
                        model_name, y_train, y_pred_train, y_test, y_pred_test
                    )
                )
                trained_models[model_name].append(model)
        metric_res_stats = self._compute_metrics_statistics(metric_res)
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
                if model_name not in metric_res:
                    metric_res[model_name] = []
                metric_res[model_name].append(
                    self._evaluate(
                        model_name, y_train, y_pred_train, y_test, y_pred_test
                    )
                )
                trained_models[model_name].append(model)
        metric_res_stats = self._compute_metrics_statistics(metric_res)
        return res, metric_res, metric_res_stats, trained_models

    def _evaluate(
        self,
        model_name: str,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict:
        self._logger.info(f"Evaluating {model_name}...")
        eval_res = {}
        for metric_name, metric in self._metrics.items():
            eval_res[metric_name] = {}
            self._logger.info(f"Evaluating {model_name} on {metric_name}...")
            metric_train = metric(y_train, y_pred_train)
            metric_test = metric(y_test, y_pred_test)
            eval_res[metric_name]["train"] = {
                "value": metric_train,
            }
            eval_res[metric_name]["test"] = {
                "value": metric_test,
            }
        return eval_res

    def _compute_metrics_statistics(self, metric_res: Dict) -> Dict:
        """
        Compute average and standard deviation of metric values across bootstrap samples
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
