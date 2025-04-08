import argparse

import pandas as pd

from helix.utils.logging_utils import Logger


def calculate_ensemble_mean(
    feature_importance_dict: dict[str, pd.DataFrame], logger: Logger
) -> pd.DataFrame:
    """Calculate mean of feature importance results across models and methods.

    Args:
        feature_importance_dict: Dictionary where:
            - Keys are model names
            - Values are DataFrames where:
                - Rows are features
                - Columns are importance methods (SHAP, Permutation, etc.)
                - Values are already normalized between 0 and 1
        logger: Logger instance

    Returns:
        pd.DataFrame: DataFrame with unique features as index and a single column
            containing the overall mean importance across all models, methods, and repetitions.
    """
    logger.info("Calculating Mean ensemble importance...")

    # First calculate mean across methods for each model
    model_means = {}
    for model_name, importance_df in feature_importance_dict.items():
        # Calculate mean across all methods for this model
        model_means[model_name] = importance_df.mean(axis=1)

    # Combine all model means into a DataFrame
    all_model_means = pd.DataFrame(model_means)

    # Calculate mean across all models
    combined_mean = all_model_means.mean(axis=1).to_frame(name="Mean Importance")

    # Group by index (feature names) and calculate mean for repeated features
    ensemble_mean = combined_mean.groupby(level=0).mean()

    # Sort by importance in descending order
    ensemble_mean = ensemble_mean.sort_values("Mean Importance", ascending=False)

    logger.info("Mean ensemble importance calculated...")
    return ensemble_mean


def calculate_ensemble_majorityvote(
    feature_importance_dict: dict[str, pd.DataFrame], logger: Logger
) -> pd.DataFrame:
    """Calculate majority vote of feature importance results.
    For majority vote, each vector in the feature importance matrix has their features ranked based on their importance.
    Subsequently, the final feature importance is the average of the most common rank order for each feature.
    For example, feature Xi has a final rank vector of [1, 1, 1, 2], where each rank rk is established by a different feature importance method k.
    The final feature importance value for feature Xi is the average value from the three feature importance methods that ranked it as 1.

    Args:
        feature_importance_dict: Dictionary where:
            - Keys are model names
            - Values are DataFrames where:
                - Rows are features
                - Columns are importance methods (SHAP, Permutation, etc.)
                - Values are already normalized between 0 and 1
        logger: Logger instance

    Returns:
        pd.DataFrame: DataFrame with features as index and a single column
            containing the majority vote importance.
    """
    logger.info("Calculating Majority Vote ensemble importance...")

    # First combine all model DataFrames into a single DataFrame
    ensemble_majorityvote = pd.concat(feature_importance_dict.values())

    # Return early if no feature importances
    if ensemble_majorityvote.empty:
        return pd.DataFrame(columns=["Majority Vote Importance"])

    # Rank the features based on their importance in each model and method
    rank_feature_importance = ensemble_majorityvote.rank(
        axis=0, method="dense", ascending=False
    )

    # Find the mode (most common) rank for each feature
    mode_rank_feature_importance = rank_feature_importance.mode(axis=1)

    # Find the most common rank order for each feature
    majority_votes_1 = mode_rank_feature_importance[0]

    # Create mask for features at their most common rank
    majority_votes_rank_1 = rank_feature_importance.eq(majority_votes_1, axis=0)
    final_majority_votes = majority_votes_rank_1

    # If there's a second most common rank, include it
    if len(mode_rank_feature_importance.columns) > 1:
        majority_votes_2 = mode_rank_feature_importance[1]
        majority_votes_rank_2 = rank_feature_importance.eq(majority_votes_2, axis=0)
        final_majority_votes = final_majority_votes | majority_votes_rank_2

    # Calculate mean importance where features have their most common rank
    ensemble_majorityvote = (
        ensemble_majorityvote.where(final_majority_votes, 0)
        .mean(axis=1)
        .to_frame(name="Majority Vote Importance")
    )

    # Group by index to handle repeated features
    ensemble_majorityvote = ensemble_majorityvote.groupby(level=0).mean()

    # Sort by importance in descending order
    ensemble_majorityvote = ensemble_majorityvote.sort_values(
        "Majority Vote Importance", ascending=False
    )
    print(ensemble_majorityvote)

    logger.info("Majority Vote ensemble importance calculated...")
    return ensemble_majorityvote


def calculate_ensemble_fuzzy(feature_importance_results, opt: argparse.Namespace):
    raise NotImplementedError("Fuzzy ensemble method is not implemented yet")
