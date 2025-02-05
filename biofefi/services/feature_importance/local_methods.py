import warnings
from typing import Any

import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import is_classifier

from biofefi.options.enums import ProblemTypes
from biofefi.utils.logging_utils import Logger


def calculate_lime_values(
    model, X: pd.DataFrame, problem_type: ProblemTypes, logger: Logger
) -> pd.DataFrame:
    """Calculate LIME values for a given model and dataset.

    Args:
        model: The model.
        X (pd.DataFrame): The dataset.
        problem_type (ProblemTypes): The problem type.
        logger (Logger): The logger.

    Returns:
        pd.DataFrame: The LIME values.
    """
    logger.info(f"Calculating LIME Importance for {model.__class__.__name__} model..")

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    explainer = LimeTabularExplainer(X.to_numpy(), mode=problem_type)

    coefficients = []

    for i in range(X.shape[0]):
        if is_classifier(model):
            explanation = explainer.explain_instance(
                X.iloc[i, :], model.predict_proba, num_features=X.shape[1]
            )
        else:
            explanation = explainer.explain_instance(
                X.iloc[i, :], model.predict, num_features=X.shape[1]
            )

        coefficients.append([item[-1] for item in explanation.local_exp[1]])

    lr_lime_values = pd.DataFrame(coefficients, columns=X.columns, index=X.index)

    # TODO: scale coefficients between 0 and +1 (low to high impact)

    logger.info("LIME Importance Analysis Completed..")

    return lr_lime_values


def calculate_local_shap_values(
    model,
    X: pd.DataFrame,
    shap_reduce_data: int,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate local SHAP values for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): The dataset.
        shap_reduce_data (int): The percentage of data to use for SHAP calculation.
        logger (Logger): The logger.

    Returns:
        tuple[pd.DataFrame, Any]: SHAP dataframe and SHAP values.
    """
    logger.info(f"Calculating SHAP Importance for {model.__class__.__name__} model..")

    if shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(X, int(X.shape[0] * shap_reduce_data / 100))
        explainer = shap.Explainer(model.predict, X_reduced)

    shap_values = explainer(X)

    shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
    # TODO: scale coefficients between 0 and +1 (low to high impact)

    logger.info("SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values
