from sklearn.base import is_classifier
import argparse
import pandas as pd


def calculate_permutation_importance(model, X, y, opt: argparse.Namespace, logger):
    """Calculate permutation importance for a given model and dataset
    Args:
        model: Model object
        X: Input features
        y: Target variable
        opt: Options
        logger: Logger
    Returns:
        permutation_importance: Permutation importance values
    """
    from sklearn.inspection import permutation_importance

    logger.info(f"Calculating Permutation Importance..")

    # Use permutation importance in sklearn.inspection to calculate feature importance
    permutation_importance_results = permutation_importance(
        model,
        X=X,
        y=y,
        scoring=opt.permutation_importance_scoring,
        n_repeats=opt.permutation_importance_repeat,
        random_state=opt.random_state,
    )
    # Create a DataFrame with the results
    permutation_importance_df = pd.DataFrame(
        permutation_importance_results.importances_mean, index=X.columns
    )

    # Return the DataFrame
    return permutation_importance_df


def calculate_shap_values(model, X, shap_type, opt: argparse.Namespace, logger):
    """Calculate SHAP values for a given model and dataset
    Args:
        model: Model object
        X: Dataset
        opt: Options
        logger: Logger
    Returns:
        shap_df: Average SHAP values
    """
    import shap

    logger.info(f"Calculating SHAP Importance..")

    if opt.shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(X, int(X.shape[0] * opt.shap_reduce_data / 100))
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

    # Return the DataFrame
    return shap_df, shap_values


def calculate_lime_values(model, X, opt: argparse.Namespace, logger):
    """Calculate LIME values for a given model and dataset
    Args:
        model: Model object
        X: Dataset
        opt: Options
        logger: Logger
    Returns:
        lime_df: Average LIME values
    """
    logger.info(f"Calculating LIME Importance..")
    # Use LIME to explain predictions
    from lime.lime_tabular import LimeTabularExplainer
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    explainer = LimeTabularExplainer(X.to_numpy(), mode=opt.problem_type)

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

    return lr_lime_values
