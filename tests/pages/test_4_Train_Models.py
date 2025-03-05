# 3. a) Test manual Linear Model
# 3. b) Test manual Random Forest
# 3. c) Test manual SVM
# 3. d) Test manual XGBoost
# 4. a) Test AHPS Linear Model
# 4. b) Test AHPS Random Forest
# 4. c) Test AHPS SVM
# 4. d) Test AHPS XGBoost


import pytest
from streamlit.testing.v1 import AppTest

from .fixtures import (
    data_opts,
    dummy_data,
    execution_opts,
    new_experiment,
    plotting_opts,
)


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/4_Train_Models.py")

    # Act
    at.run(timeout=10.0)

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/4_Train_Models.py")
    at.run(timeout=10.0)

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()
