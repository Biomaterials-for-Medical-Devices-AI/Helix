"""Feature selection utilities for Helix."""

import numpy as np
from sklearn.feature_selection import VarianceThreshold


def sort_by_feature_name(df: pd.DataFrame) -> pd.DataFrame:
    """Sort features by name length to ensure consistent ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features as columns

    Returns
    -------
    pd.DataFrame
        DataFrame with columns sorted by name length
    """
    df = df.T
    df["len"] = df.T.columns.str.len()
    df_sorted = df.sort_values(["len"])
    df_sorted = df_sorted.drop(["len"], axis=1)
    return df_sorted.T


def remove_correlation(dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove highly correlated features based on Pearson correlation.

    Args:

    dataset : pd.DataFrame
        Input DataFrame with features as columns
    threshold : float
        Correlation threshold. Features with correlation above this will be removed.

    Returns:

    pd.DataFrame
        DataFrame with correlated features removed
    """
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset
    return dataset


def remove_low_variance(X, threshold=0.01):
    """Remove features with low variance.

    Args:
    X : pd.DataFrame
        Input DataFrame with features as columns
    threshold : float, default=0.01
        Features with variance below this threshold will be removed

    Returns:
    pd.DataFrame
        DataFrame with low variance features removed
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return X.iloc[:, selector.get_support()]


def standard_error_prediction(y_true, y_pred):
    """Calculate Standard Error of Prediction.

    Args:
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values

    Returns:
    float
        Standard error of prediction
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
