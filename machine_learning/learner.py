import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

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
        self._models = get_models(
            self._model_types, self._problem_type, logger=self._logger
        )
        if self._data_split["type"] == "holdout":
            res, metric_res, trained_models = self._fit_holdout(data)
            return res, metric_res, trained_models

    def _fit_holdout(self, data: Tuple) -> None:
        self._logger.info("Fitting holdout...")
        X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test
        res = {}
        metric_res = {}
        trained_models = {}
        res["scaler"] = data.scaler
        res["y_test"] = y_test
        res["y_train"] = y_train
        for model_name, model in self._models.items():
            res[model_name] = {}
            self._logger.info(f"Fitting {model_name}...")
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            res[model_name]["y_pred_train"] = y_pred_train
            y_pred_test = model.predict(X_test)
            res[model_name]["y_pred_test"] = y_pred_test
            metric_res[model_name] = self._evaluate(
                model_name, y_train, y_pred_train, y_test, y_pred_test
            )
            trained_models[model_name] = model
        return res, metric_res, trained_models

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
