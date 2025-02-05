from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from biofefi.options.enums import (
    DataPreprocessingStateKeys,
    Normalisations,
    TransformationsY,
)
from biofefi.options.file_paths import data_preprocessing_options_path
from biofefi.options.preprocessing import PreprocessingOptions
from biofefi.services.configuration import save_options


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

    elif normalisation_method == Normalisations.Standardization:
        scaler = StandardScaler()

    elif normalisation_method == Normalisations.MinMax:
        scaler = MinMaxScaler()

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
) -> None:
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
        X = X[selected_features]

    processed_data = pd.concat([X, y], axis=1)

    return processed_data


def run_preprocessing(
    data: pd.DataFrame, experiment_path: Path, config: PreprocessingOptions
) -> None:

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    save_options(data_preprocessing_options_path(experiment_path), config)
    X = normalise_independent_variables(config.independent_variable_normalisation, X)
    y = transform_dependent_variable(config.dependent_variable_transformation, y)

    normalised_data = pd.concat([X, y], axis=1)

    processed_data = run_feature_selection(config, normalised_data)

    return processed_data
