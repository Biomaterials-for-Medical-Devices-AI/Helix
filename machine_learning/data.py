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
    if data_split["type"].lower() == "holdout":
        df_train, df_test = train_test_split(df, test_size=data_split["test_size"], random_state=random_state)
        return (df_train, df_test)
    else:
        return (df,)