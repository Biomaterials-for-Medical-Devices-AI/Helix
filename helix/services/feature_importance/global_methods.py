from typing import Any

import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from helix.utils.logging_utils import Logger


def calculate_global_shap_values(
    model,
    X: pd.DataFrame,
    shap_reduce_data: int,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate SHAP values for a given model and dataset.

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

    # Calculate Average Importance + set column names as index
    shap_df = (
        pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame()
    )

    logger.info("SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values


def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    permutation_importance_scoring: str,
    permutation_importance_repeat: int,
    random_state: int,
    logger: Logger,
):
    """Calculate permutation importance for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.
        permutation_importance_scoring (str): Permutation importance scoring method.
        permutation_importance_repeat (int): Number of repeats for importance scoring.
        random_state (int): Seed for the random state.
        logger (Logger): The logger.

    Returns:
        permutation_importance: Permutation importance values
    """

    logger.info(
        f"Calculating Permutation Importance for {model.__class__.__name__} model.."
    )

    # Use permutation importance in sklearn.inspection to calculate feature importance
    permutation_importance_results = permutation_importance(
        model,
        X=X,
        y=y,
        scoring=permutation_importance_scoring,
        n_repeats=permutation_importance_repeat,
        random_state=random_state,
    )
    # Create a DataFrame with the results
    permutation_importance_df = pd.DataFrame(
        permutation_importance_results.importances_mean, index=X.columns
    )

    logger.info("Permutation Importance Analysis Completed..")

    # Return the DataFrame
    return permutation_importance_df
