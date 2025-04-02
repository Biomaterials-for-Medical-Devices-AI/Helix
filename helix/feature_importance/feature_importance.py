from typing import Any, Callable
import pandas as pd
from sklearn.base import BaseEstimator
from helix.feature_importance.interpreter import Interpreter
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.plotting import PlottingOptions
from helix.utils.logging_utils import Logger


def run(
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data,
    models,
    logger,
):

    # Interpret the model results
    interpreter = Interpreter(
        fi_opt=fi_opt, exec_opt=exec_opt, plot_opt=plot_opt, logger=logger
    )
    # TODO: Add indices to the dataframe results-> global + ensemble
    # TODO: Resolve majority vote issue
    gloabl_importance_results, local_importance_results, ensemble_results = (
        interpreter.interpret(models, data)
    )

    return gloabl_importance_results, local_importance_results, ensemble_results


class BaseFeatureImportanceCalculator:
    def __init__(
        self,
        func: Callable,
        logger: Logger | None = None,
    ):
        self._func = func
        self._logger = logger

    def calculate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError


class FeatureImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> pd.DataFrame:
        return self._func(model, X, logger=self._logger, **kwargs)


class ShapFeatureImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> tuple[pd.DataFrame, Any]:
        return self._func(model, X, logger=self._logger, **kwargs)
