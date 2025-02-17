import pytest
from sklearn.ensemble import RandomForestClassifier

from biofefi.options.enums import ProblemTypes
from biofefi.services.ml_models import MlModel, get_model, get_model_type


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
