from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator
from helix.feature_importance.interpreter import FeatureImportanceEstimator
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.plotting import PlottingOptions
from helix.utils.logging_utils import Logger
from helix.services.feature_importance.global_methods import (
    calculate_global_shap_values,
    calculate_permutation_importance,
)
from helix.services.feature_importance.local_methods import (
    calculate_local_shap_values,
    calculate_lime_values,
)


def run(
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data,
    models,
    logger,
):

    # Interpret the model results
    interpreter = FeatureImportanceEstimator(
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
        logger: Logger | None = None,
    ):
        self._logger = logger

    def calculate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError


class PermutationImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> pd.DataFrame:
        return calculate_permutation_importance(model, X, y, **kwargs)


class GlobalShapFeatureImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> tuple[pd.DataFrame, Any]:
        return calculate_global_shap_values(model, X, **kwargs)

class LocalShapFeatureImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> pd.DataFrame:
        return calculate_local_shap_values(model, X, **kwargs)


class LimeFeatureImportanceCalculator(BaseFeatureImportanceCalculator):
    def calculate(self, model, X, y=None, **kwargs) -> pd.DataFrame:
        return calculate_lime_values(model, X, **kwargs)


def create_feature_importance_calculator(
    feature_importance_type: str,
    logger: Logger | None = None,
) -> BaseFeatureImportanceCalculator:
    """Factory function to create the appropriate feature importance calculator.

    Args:
        feature_importance_type (str): Type of feature importance to calculate.
            Must be one of: 'Permutation Importance', 'SHAP', 'Local SHAP', 'LIME'
        logger (Logger | None, optional): Logger instance. Defaults to None.

    Returns:
        BaseFeatureImportanceCalculator: Instance of the appropriate calculator class.

    Raises:
        ValueError: If feature_importance_type is not recognized.
    """
    calculators = {
        "Permutation Importance": PermutationImportanceCalculator,
        "SHAP": GlobalShapFeatureImportanceCalculator,
        "Local SHAP": LocalShapFeatureImportanceCalculator,
        "LIME": LimeFeatureImportanceCalculator,
    }

    calculator_class = calculators.get(feature_importance_type)
    if calculator_class is None:
        raise ValueError(
            f"Unknown feature importance type: {feature_importance_type}. "
            f"Must be one of: {list(calculators.keys())}"
        )

    return calculator_class(logger=logger)
