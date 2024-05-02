import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataBuilder:
    """
    Data builder class
    """
    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._path = opt.data_path
        self._data_split = opt.data_split
        self._random_state = opt.random_state
        self._logger = logger
        self._normalization = opt.normalization
        self._numerical_cols = 'all'

    def _load_data(self) -> Dict[str, pd.DataFrame]:
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

        logger: object
            The logger to use for logging

        Returns
        -------
        Dict[str, pd.DataFrame]
            The data loaded from the csv file
        """
        df = pd.read_csv(self._path)
        # Last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        if self._data_split["type"].lower() == "holdout":
            self._logger.info(
                f"Using holdout data split with test size {self._data_split['test_size']}"
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._data_split["test_size"], random_state=self._random_state
            )
            return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        else:
            raise NotImplementedError(
                f"Data split type {self._data_split['type']} is not implemented"
            )

    def normalise_data(self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """
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
            Options:
                'all' : Normalise all columns
                list : Normalise only the columns in the list
                'none' : Do not normalise any columns
        logger : object
            The logger to use for logging

        Returns
        -------
        X : pd.DataFrame
            Dataframe of normalised data
        """
        if self._normalization.lower() == "none":
            return X_train, X_test, None

        self._logger.info(f"Normalising data using {self._normalization}...")

        if self._normalization.lower() == "standardization":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif self._normalization.lower() == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        else:
            raise ValueError("normalization must be either'Standardization' or'MinMax'.")

        if self._numerical_cols == "all":
            self._numerical_cols = X_train.columns
        elif type(self._numerical_cols) == list:
            pass
        else:
            raise TypeError("numerical_cols must be a list of columns or 'all'.")
        X_train[self._numerical_cols] = scaler.fit_transform(X_train[self._numerical_cols])
        X_test[self._numerical_cols] = scaler.transform(X_test[self._numerical_cols])
        return X_train, X_test, scaler
    
    def ingest(self):
        data = self._load_data()
        data['X_train'], data['X_test'], scaler = self.normalise_data(data['X_train'], data['X_test'])
        return TabularData(**data, scaler=scaler)

@dataclass
class TabularData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    scaler: object


