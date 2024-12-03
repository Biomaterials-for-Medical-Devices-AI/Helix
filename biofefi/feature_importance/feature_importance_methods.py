import warnings
from typing import Any

import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance

from biofefi.options.enums import ProblemTypes
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.utils.logging_utils import Logger


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


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    shap_type: str,
    fi_opt: FeatureImportanceOptions,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate SHAP values for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): The dataset.
        shap_type (str): The type of SHAP (local or global).
        fi_opt (FeatureImportanceOptions): The options.
        logger (Logger): The logger.

    Raises:
        ValueError: SHAP type is not "local" or "global"

    Returns:
        tuple[pd.DataFrame, Any]: SHAP dataframe and SHAP values.
    """
    logger.info(f"Calculating SHAP Importance for {model.__class__.__name__} model..")

    if fi_opt.shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(
            X, int(X.shape[0] * fi_opt.shap_reduce_data / 100)
        )
        explainer = shap.Explainer(model.predict, X_reduced)

    shap_values = explainer(X)

    # add an option to check if feature importance is local or global
    if shap_type == "local":
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
        # TODO: scale coefficients between 0 and +1 (low to high impact)
    elif shap_type == "global":
        # Calculate Average Importance + set column names as index
        shap_df = (
            pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame()
        )
    else:
        raise ValueError("SHAP type must be either local or global")

    logger.info("SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values


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
