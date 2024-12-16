from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from biofefi.machine_learning.nn_models import (
    BayesianRegularisedNNClassifier,
    BayesianRegularisedNNRegressor,
)
from biofefi.options.enums import ModelNames, ProblemTypes
from biofefi.utils.utils import assert_model_param

# Mapping model types and problem types to specific model classes
_MODEL_PROBLEM_DICT = {
    (ModelNames.LinearModel, ProblemTypes.Classification): LogisticRegression,
    (ModelNames.LinearModel, ProblemTypes.Regression): LinearRegression,
    (ModelNames.RandomForest, ProblemTypes.Classification): RandomForestClassifier,
    (ModelNames.RandomForest, ProblemTypes.Regression): RandomForestRegressor,
    (ModelNames.XGBoost, ProblemTypes.Classification): XGBClassifier,
    (ModelNames.XGBoost, ProblemTypes.Regression): XGBRegressor,
    (ModelNames.SVM, ProblemTypes.Classification): SVC,
    (ModelNames.SVM, ProblemTypes.Regression): SVR,
    (
        ModelNames.BRNNClassifier,
        ProblemTypes.Classification,
    ): BayesianRegularisedNNClassifier,
    (ModelNames.BRNNRegressor, ProblemTypes.Regression): BayesianRegularisedNNRegressor,
}


def get_models(
    model_types: Dict[str, Dict], problem_type: str, logger: object = None
) -> List:
    """
    Constructs and initializes machine learning models
    based on the given configuration.

    Args:
        model_types (dict): Dictionary containing model types
        and their parameters.
        problem_type (str): Type of problem (
            classification or regression).
        logger (object): Logger object to log messages.

    Returns:
        - List: A dictionary of initialized models where th
        keys are model names and the values are instances
        of the corresponding models.

    Raises:
        - ValueError: If a model type is not recognized or unsupported
    """
    models = {}
    model_list = [
        (model_type, model["params"])
        for model_type, model in model_types.items()
        if model["use"]
    ]
    for model, model_param in model_list:
        if model_class := _MODEL_PROBLEM_DICT.get(
            (model.lower(), problem_type.lower())
        ):
            if problem_type.lower() == ProblemTypes.Classification:
                model_param = assert_model_param(
                    model_class, model_param, logger=logger
                )
                model_param["class_weight"] = "balanced"
                models[model] = model_class(**model_param)
            else:
                model_param = assert_model_param(
                    model_class, model_param, logger=logger
                )
                models[model] = model_class(**model_param)

        else:
            raise ValueError(f"Model type {model} not recognized")
    return models
