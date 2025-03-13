from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.options.data import DataOptions, DataSplitOptions
from helix.options.enums import DataSplitMethods, Normalisations, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.utils.logging_utils import Logger


class DataBuilder:
    """
    Data builder class
    """

    _normalization_dict = {
        Normalisations.MinMax: MinMaxScaler,
        Normalisations.Standardization: StandardScaler,
    }

    def __init__(
        self,
        data_path: str,
        random_state: int,
        normalisation: str,
        logger: object = None,
        data_split: DataSplitOptions | None = None,
        problem_type: str = None,
    ) -> None:
        self._path = data_path
        self._data_split = data_split
        self._random_state = random_state
        self._logger = logger
        self._normalization = normalisation
        self._numerical_cols = "all"
        self._problem_type = problem_type

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from a csv file

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training data (X) and the targets (y)
        """
        df = pd.read_csv(self._path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    def _generate_data_splits(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Dict[str, List[pd.DataFrame]]:
        """Generate data splits for bootstrapping.

        Args:
            X (pd.DataFrame): The training data.
            y (pd.DataFrame): The prediction targets.

        Raises:
            NotImplementedError: Tried to use an unimplemented data split method.

        Returns:
            Dict[str, List[pd.DataFrame]]: The bootstrapped data.
        """
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        if (
            self._data_split is not None
            and self._data_split.method.lower() == DataSplitMethods.Holdout
        ):
            for i in range(self._data_split.n_bootstraps):
                self._logger.info(
                    "Using holdout data split "
                    f"with test size {self._data_split.test_size} "
                    f"for bootstrap {i+1}"
                )
                if self._problem_type == ProblemTypes.Regression:
                    stratify = None
                elif self._problem_type == ProblemTypes.Classification:
                    stratify = y
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self._data_split.test_size,
                    random_state=self._random_state + i,
                    stratify=stratify,
                    shuffle=True,
                )
                X_train_list.append(X_train)
                X_test_list.append(X_test)
                y_train_list.append(y_train)
                y_test_list.append(y_test)
        elif (
            self._data_split is not None
            and self._data_split.method.lower() == DataSplitMethods.KFold
        ):
            folds = self._data_split.k_folds
            kf = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=self._random_state
            )
            kf.get_n_splits(X)

            if self._problem_type == ProblemTypes.Regression:
                stratify = np.zeros(y.shape[0])
            elif self._problem_type == ProblemTypes.Classification:
                stratify = y

            for i, (train_index, test_index) in enumerate(kf.split(X, stratify)):

                self._logger.info(
                    "Using K-Fold data split "
                    f"with test size {len(test_index)} "
                    f"for fold {i+1}"
                )

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                X_train_list.append(X_train)
                X_test_list.append(X_test)
                y_train_list.append(y_train)
                y_test_list.append(y_test)
        elif (
            self._data_split is not None
            and self._data_split.method.lower() == DataSplitMethods.NoSplit
        ):
            if self._problem_type == ProblemTypes.Regression:
                stratify = None
            else:
                stratify = y
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self._data_split.test_size,
                random_state=self._random_state,
                stratify=stratify,
            )
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
        else:
            raise NotImplementedError(
                f"Data split type {self._data_split.method} is not implemented"
            )

        return {
            "X_train": X_train_list,
            "X_test": X_test_list,
            "y_train": y_train_list,
            "y_test": y_test_list,
        }

    def _normalise_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalise data using MinMaxScaler

        Parameters
        ----------
        data : pd.DataFrame
            The data to normalise

        Returns
        -------
        X : pd.DataFrame
            Dataframe of normalised data
        """
        if self._normalization.lower() == Normalisations.NoNormalisation:
            return data

        self._logger.info(f"Normalising data using {self._normalization}...")

        scaler = self._normalization_dict.get(self._normalization.lower())
        if not scaler:
            raise ValueError(
                f"Normalization {self._normalization} is not available. "
                f"Choices are {self._normalization_dict.keys()}"
            )
        scaler = scaler()  # create the scaler object

        if isinstance(self._numerical_cols, str) and self._numerical_cols == "all":
            self._numerical_cols = data.columns
        elif isinstance(self._numerical_cols, pd.Index):
            pass
        else:
            raise TypeError("numerical_cols must be a list of columns or 'all'.")
        data[self._numerical_cols] = scaler.fit_transform(data[self._numerical_cols])
        return data

    def ingest(self):
        X, y = self._load_data()
        data = self._generate_data_splits(X, y)

        return TabularData(
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=data["y_train"],
            y_test=data["y_test"],
        )


@dataclass
class TabularData:
    # X_train as a list of dataframes
    X_train: list[pd.DataFrame]
    X_test: list[pd.DataFrame]
    y_train: list[pd.DataFrame]
    y_test: list[pd.DataFrame]


@st.cache_data(show_spinner="Loading data...")
def ingest_data(
    exec_opts: ExecutionOptions, data_opts: DataOptions, _logger: Logger
) -> TabularData:
    """
    Load data from disk if the data is not in the streamlit cache,
    else return the stored value. This behaviour is controlled by the
    decorator on the function signature (`@st.cache_data`).

    Args:
        exec_opts (ExecutionOptions): The execution options.
        data_opts (DataOptions): The data options.
        _logger (Logger): The logger.

    Returns:
        TabularData: The ingested data.
    """
    data = DataBuilder(
        data_path=data_opts.data_path,
        random_state=exec_opts.random_state,
        normalisation=data_opts.normalisation,
        logger=_logger,
        data_split=data_opts.data_split,
        problem_type=exec_opts.problem_type,
    ).ingest()
    return data


def read_data(data_path: Path, logger: Logger) -> pd.DataFrame:
    """Read a data file into memory from a '.csv' or '.xlsx' file.

    Args:
        data_path (Path): The path to the file to be read.
        logger (Logger): The logger.

    Raises:
        ValueError: The data file wasn't a '.csv' or '.xlsx' file.

    Returns:
        pd.DataFrame: The data read from the file.
    """
    if data_path.suffix == ".csv":
        try:
            logger.info(f"Reading data from {data_path}")
            return pd.read_csv(data_path, header=0)
        except Exception as e:
            logger.error(f"Failed to read data from {data_path}{os.linesep}{e}")
            raise
    elif data_path.suffix == ".xlsx":
        try:
            logger.info(f"Reading data from {data_path}")
            return pd.read_excel(data_path, header=0)
        except Exception as e:
            logger.error(f"Failed to read data from {data_path}{os.linesep}{e}")
            raise
    else:
        raise ValueError("data_path must be to a '.csv' or '.xlsx' file")


def save_data(data_path: Path, data: pd.DataFrame, logger: Logger):
    """Save data to either a '.csv' or '.xlsx' file.

    Args:
        data_path (Path): The path to save the data to.
        data (pd.DataFrame): The data to save.
        logger (Logger): The logger.

    Raises:
        ValueError: The data file wasn't a '.csv' or '.xlsx' file.
    """
    if data_path.suffix == ".csv":
        try:
            logger.info(f"Saveing data to {data_path}")
            data.to_csv(data_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save data to {data_path}{os.linesep}{e}")
            raise
    elif data_path.suffix == ".xlsx":
        try:
            logger.info(f"Saving data to {data_path}")
            data.to_excel(data_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save data to {data_path}{os.linesep}{e}")
            raise
    else:
        raise ValueError("data_path must be to a '.csv' or '.xlsx' file")
