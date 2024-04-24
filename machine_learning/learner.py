import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from machine_learning.data import normalise_data
from machine_learning.get_models import get_models
from machine_learning.metrics import get_metrics


class Learner:
    """
    Learner class
    """

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._logger = logger
        self._opt = opt
        self._model_types = self._opt.model_types
        self._problem_type = self._opt.problem_type
        self._data_split = self._opt.data_split
        self._normalization = self._opt.normalization
        self._metrics = get_metrics(self._problem_type, logger=self._logger)

    def fit(self, data: Tuple) -> None:
        models = get_models(self._model_types, self._problem_type, logger=self._logger)
        if self._data_split["type"] == "holdout":
            res = self._fit_holdout(models, data)
            metric_res = self._evaluate(res)
            return metric_res

    def _fit_holdout(self, models: List, data: Tuple) -> None:
        self._logger.info("Fitting holdout...")
        X_train, X_test, y_train, y_test = data
        X_train, X_test, scaler = normalise_data(
            X_train, X_test, self._normalization, logger=self._logger
        )
        res = {}
        res["scaler"] = scaler
        for model_name, model in models.items():
            res[model_name] = {}
            self._logger.info(f"Fitting {model_name}...")
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            res[model_name]["y_pred_train"] = y_pred_train
            y_pred_test = model.predict(X_test)
            res[model_name]["y_pred_test"] = y_pred_test
        res["y_test"] = y_test
        res["y_train"] = y_train
        return res

    def _evaluate(self, res: Dict):
        metric_res = {}
        for model_name in self._model_types.keys():
            self._logger.info(f"Evaluating {model_name}...")
            metric_res[model_name] = {}
            y_pred_train = res[model_name]["y_pred_train"]
            y_pred_test = res[model_name]["y_pred_test"]
            y_train = res["y_train"]
            y_test = res["y_test"]
            for metric_name, metric in self._metrics.items():
                metric_res[model_name][metric_name] = {}
                self._logger.info(f"Evaluating {model_name} on {metric_name}...")
                metric_train = metric(y_train, y_pred_train)
                metric_test = metric(y_test, y_pred_test)
                metric_res[model_name][metric_name]["train"] = {
                    "value": metric_train,
                }
                metric_res[model_name][metric_name]["test"] = {
                    "value": metric_test,
                }
        return metric_res
