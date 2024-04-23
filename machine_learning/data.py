from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str, data_split: Dict, random_state: int = 1221) -> pd.DataFrame:
    """
    Load data from a csv file

    Parameters
    ----------
    path: str
        The path to the csv file

    data_split_type: str
        The type of data split to use

    random_state: int
        The random state to use for reproducibility

    Returns
    -------
    pd.DataFrame
        The data loaded from the csv file
    """
    df = pd.read_csv(path)
    # Last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if data_split["type"].lower() == "holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_split["test_size"], random_state=random_state)
        return X_train, X_test, y_train, y_test
    else:
        return (df,)
    
def normalise_data(X_train: pd.DataFrame, X_test: pd.DataFrame, normalization: str = 'Standardization', numerical_cols: str = 'all') -> pd.DataFrame:
    '''
    Normalise data using MinMaxScaler

    Parameters
    ----------
    X_train : pd.DataFrame
        Train data to normalise
    X_test : pd.DataFrame
        Test data to normalise
    normalization : str
        Normalization method to use
        Options:
           'standardization' : Standardize features by removing the mean and scaling to unit variance
           'minmax' : Scales features to 0-1
    numerical_cols : str or list
        List of numerical columns to normalise

    Returns
    -------
    X : pd.DataFrame
        Dataframe of normalised data
    '''
    if normalization.lower() == 'none':
        return X_train, X_test, None
   
    print(f'Normalising data using {normalization}...')

    if normalization.lower() =='standardization':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif normalization.lower() =='minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("normalization must be either'Standardization' or'MinMax'.")

    if numerical_cols == 'all':
        numerical_cols = X_train.columns
    elif type(numerical_cols) == list:
        numerical_cols = numerical_cols
    else:
        raise TypeError("numerical_cols must be a list of columns or 'all'.")
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test, scaler