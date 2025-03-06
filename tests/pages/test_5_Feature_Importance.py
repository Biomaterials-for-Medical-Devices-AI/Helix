# 1. Fixture for models to perform feature importance
# 2. Test page loads without error
# 3. Test page finds experiment
# 4. Test what happens when no models to test
# 5. Test plots are produced
# 6. Test for results


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
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py")

    # Act
    at.run(timeout=10.0)

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py")
    at.run(timeout=10.0)

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()
