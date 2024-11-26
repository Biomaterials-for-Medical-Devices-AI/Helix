from typing import Any
import pandas as pd
import shap

from biofefi.utils.logging_utils import Logger


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

    logger.info(f"SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values


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

    logger.info(f"SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values
