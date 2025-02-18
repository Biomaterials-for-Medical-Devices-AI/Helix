from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from biofefi.options.enums import ProblemTypes
from biofefi.services.ml_models import (
    MlModel,
    get_model,
    get_model_type,
    save_model_predictions,
)


def test_save_model_predictions():
    """Test that save_model_predictions correctly writes a DataFrame to a CSV file."""

    # Arrange
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "predictions.csv"

        # Create sample DataFrame
        predictions_df = pd.DataFrame(
            {
                "y_true": [1, 0, 1],
                "y_pred": [1, 0, 0],
                "y_pred_proba": [0.9, 0.3, 0.6],
                "model_name": ["ModelA", "ModelA", "ModelA"],
                "Set": ["train", "test", "val"],
                "Fold": [1, 1, 1],
            }
        )

        # Act
        save_model_predictions(predictions_df, file_path)

        # Assert
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(predictions_df, loaded_df)


def test_get_model_type_returns_type():
    # Arrange
    model_type = "Random Forest"

    # Act
    model = get_model_type(model_type, ProblemTypes.Regression)

    # Assert
    assert isinstance(model, type)


def test_get_model_type_throws_value_error():
    # Arrange
    model_types = "Unknown"

    # Act/Assert
    with pytest.raises(ValueError):
        get_model_type(model_types, ProblemTypes.Regression)


def test_get_model_returns_instance():
    # Arrange
    model_type = RandomForestClassifier
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_depth": None,
    }

    # Act
    model: MlModel = get_model(model_type, params)
    params_in_actual_params = all(
        item in model.get_params().items() for item in params.items()
    )

    # Assert
    assert isinstance(model, RandomForestClassifier)
    assert params_in_actual_params
