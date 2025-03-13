from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from helix.services.data import read_data
from helix.utils.logging_utils import Logger


def test_read_data_reads_csv_files():
    # Arrange
    file_path = Path(f"{uuid.uuid4()}.csv")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=4,
        random_state=42,
    )
    data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    np.savetxt(file_path, data, delimiter=",")
    logger = Logger().make_logger()

    # Act
    df = read_data(file_path, logger)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (499, 11)  # 499 not 500, first row assumed to be headers

    # Cleanup
    if file_path.exists():
        file_path.unlink()
