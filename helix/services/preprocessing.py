from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from helix.options.enums import (
    DataPreprocessingStateKeys,
    Normalisations,
    TransformationsY,
)
from helix.options.file_paths import data_preprocessing_options_path
from helix.options.preprocessing import PreprocessingOptions
from helix.services.configuration import save_options
from helix.preprocessing.variable_scaling import MeanCenterPoissonScaler, ParetoScaler


def find_non_numeric_columns(data: pd.DataFrame | pd.Series) -> List[str]:
    """
    Find non-numeric columns in a DataFrame or check if a Series contains non-numeric values.

    Args:
        data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to check.

    Returns:
        List[str]: If `data` is a DataFrame, returns a list of non-numeric column names.
                   If `data` is a Series, returns ["Series"] if it contains non-numeric values, else an empty list.
    """
    if isinstance(data, pd.Series):  # If input is a Series
        return (
            data.name
            if data.apply(lambda x: pd.to_numeric(x, errors="coerce")).isna().any()
            else []
        )

    elif isinstance(data, pd.DataFrame):  # If input is a DataFrame
        return [
            col
            for col in data.columns
            if data[col].apply(lambda x: pd.to_numeric(x, errors="coerce")).isna().any()
        ]

    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")


def normalise_independent_variables(normalisation_method: str, X):
    """
    Normalise the independent variables based on the selected method.

    Args:
        normalisation_method (str): The normalisation method to use.
        X (pd.DataFrame): The independent variables to normalise.

    Returns:
        pd.DataFrame: The normalised independent variables.
    """

    if normalisation_method == Normalisations.NoNormalisation:
        return X

    elif normalisation_method == Normalisations.Standardisation:
        scaler = StandardScaler()

    elif normalisation_method == Normalisations.MinMax:
        scaler = MinMaxScaler()

    elif normalisation_method == Normalisations.MeanCentering:
        scaler = StandardScaler(with_std=False)

    elif normalisation_method == Normalisations.MeanCenteringPoissonScaling:
        scaler = MeanCenterPoissonScaler()

    elif normalisation_method == Normalisations.ParetoScaling:
        scaler = ParetoScaler()

    column_names = X.columns
    processed_X = scaler.fit_transform(X)

    processed_X = pd.DataFrame(processed_X, columns=column_names)

    return processed_X


def transform_dependent_variable(transformation_y_method: str, y):
    """
    Transform the dependent variable based on the selected method.

    Args:
        transformation_y_method (str): The transformation method to use.
        y (pd.Series): The dependent variable to transform.

    Returns:
        pd.Series: The transformed dependent variable.
    """

    if transformation_y_method == TransformationsY.NoTransformation:
        return y

    column_name = y.name
    y = y.to_numpy().reshape(-1, 1)

    if transformation_y_method == TransformationsY.Log:
        if y.min() <= 0:
            y = y - y.min() + 1
        y = np.log(y)

    elif transformation_y_method == TransformationsY.Sqrt:
        if y.min() < 0:
            y = y - y.min()
        y = np.sqrt(y)

    elif transformation_y_method == TransformationsY.MinMaxNormalisation:
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y)

    elif transformation_y_method == TransformationsY.StandardisationNormalisation:
        scaler = StandardScaler()
        y = scaler.fit_transform(y)

    y = pd.DataFrame(y, columns=[column_name])

    return y


def run_feature_selection(
    preprocessing_opts: PreprocessingOptions, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Run feature selection on the data based on the selected methods.

    Args:
        feature_selection_methods (dict): A dictionary of the feature selection methods to use.
        data (pd.DataFrame): The data to perform feature selection on.

    Returns:
        pd.DataFrame: The processed data.

    """
    feature_selection_methods = preprocessing_opts.feature_selection_methods
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if feature_selection_methods[DataPreprocessingStateKeys.VarianceThreshold]:
        varianceselector = VarianceThreshold(preprocessing_opts.variance_threshold)
        X = varianceselector.fit_transform(X)
        variance_columns = varianceselector.get_feature_names_out()
        X = pd.DataFrame(X, columns=variance_columns)

    if feature_selection_methods[DataPreprocessingStateKeys.CorrelationThreshold]:
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > preprocessing_opts.correlation_threshold)
        ]
        X = X.drop(columns=to_drop)

    if feature_selection_methods[DataPreprocessingStateKeys.LassoFeatureSelection]:
        lasso = Lasso(alpha=preprocessing_opts.lasso_regularisation_term)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        if selected_features.empty:
            raise ValueError(
                "No indepdendent variables remain after applying "
                "LASSO regularisation term "
                f"**{preprocessing_opts.lasso_regularisation_term}**."
            )
        X = X[selected_features]

    processed_data = pd.concat([X, y], axis=1)

    return processed_data


def run_preprocessing(
    data: pd.DataFrame, experiment_path: Path, config: PreprocessingOptions
) -> pd.DataFrame:

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    try:
        columns_to_drop = find_non_numeric_columns(X)
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
    except TypeError as e:
        raise e

    try:
        convert_y = find_non_numeric_columns(y)
        if convert_y:
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = pd.Series(y, name=convert_y)
    except TypeError as e:
        raise e

    save_options(data_preprocessing_options_path(experiment_path), config)
    X = normalise_independent_variables(config.independent_variable_normalisation, X)
    y = transform_dependent_variable(config.dependent_variable_transformation, y)

    normalised_data = pd.concat([X, y], axis=1)

    processed_data = run_feature_selection(config, normalised_data)

    return processed_data


# Function to convert nominal columns to numeric
def convert_nominal_to_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """Convert all nominal (categorical) columns in a DataFrame to numeric values.
    This function identifies all object or category type columns in the input DataFrame
    and converts them to numeric representations using pandas' factorize method.
    Each unique category is assigned a unique integer value.

    Args:
        data (pd.DataFrame): The input DataFrame containing columns to be converted.

    Returns:
        pd.DataFrame: A DataFrame with all categorical columns converted to numeric values.
    """
    for col in data.columns:
        if (
            data[col].dtype == "object" or data[col].dtype.name == "category"
        ):  # Check if column is categorical
            data[col] = pd.factorize(data[col])[0]
    return data
