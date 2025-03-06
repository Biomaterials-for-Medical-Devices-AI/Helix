# 1. Fixture for models to perform feature importance
# 2. Test page loads without error
# 3. Test page finds experiment
# 4. Test what happens when no models to test
# 5. Test plots are produced
# 6. Test for results


from pickle import dump
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from helix.options.data import DataOptions, DataSplitOptions
from helix.options.enums import DataSplitMethods
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import helix_experiments_base_dir, ml_model_dir
from helix.services.ml_models import save_model

from .fixtures import (
    data_opts,
    dummy_data,
    execution_opts,
    new_experiment,
    plotting_opts,
)


@pytest.fixture
def models_to_evaluate(
    new_experiment: str,
    dummy_data: np.ndarray,
    execution_opts: ExecutionOptions,
    data_opts: DataOptions,
):
    save_dir = ml_model_dir(helix_experiments_base_dir() / new_experiment)
    data_opts.data_split = DataSplitOptions(
        n_bootstraps=3, method=DataSplitMethods.Holdout
    )
    X, y = dummy_data[:, :-1], dummy_data[:, -1]
    for i in range(data_opts.data_split.n_bootstraps):
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=data_opts.data_split.test_size,
            random_state=execution_opts.random_state + i,
        )
        model = LogisticRegression()
        model.fit(X_train, y_train)
        save_model(model, save_dir / f"{model.__class__.__name__}-{i}.pkl")


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert at.info  # the info saying there's no models should appear
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_can_find_models(new_experiment: str, models_to_evaluate: None):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert not at.info  # the info box saying there's no models shouldn't appear


def test_warning_appears_with_no_global_methods_selected(
    new_experiment: str, models_to_evaluate: None
):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Select explain all models
    at.toggle[0].set_value(True).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert (
        at.warning[0].value
        == "You must configure at least one global feature importance method to perform ensemble methods."
    )
