# 3. b) Test manual Random Forest
# 3. c) Test manual SVM
# 3. d) Test manual XGBoost
# 4. a) Test AHPS Linear Model
# 4. b) Test AHPS Random Forest
# 4. c) Test AHPS SVM
# 4. d) Test AHPS XGBoost


import pytest
from streamlit.testing.v1 import AppTest

from helix.options.enums import DataSplitMethods
from helix.options.file_paths import (
    helix_experiments_base_dir,
    ml_metrics_path,
    ml_model_dir,
    ml_plot_dir,
    ml_predictions_path,
)

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


@pytest.mark.parametrize(
    "data_split_method,holdout_or_k",
    [
        (DataSplitMethods.Holdout.capitalize(), 3),
        (DataSplitMethods.KFold.capitalize(), 3),
    ],
)
def test_manual_linear_model(
    new_experiment: str, data_split_method: DataSplitMethods, holdout_or_k: int
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=60)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Unselect AHPS, which is on by default
    at.toggle[0].set_value(False).run()
    # Select the data split method
    at.selectbox[1].select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    at.number_input[0].set_value(holdout_or_k).run()
    # Select Linear Model
    at.toggle[1].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()


@pytest.mark.parametrize(
    "data_split_method,holdout_or_k",
    [
        (DataSplitMethods.Holdout.capitalize(), 3),
        (DataSplitMethods.KFold.capitalize(), 3),
    ],
)
def test_manual_random_forest(
    new_experiment: str, data_split_method: DataSplitMethods, holdout_or_k: int
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=60)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Unselect AHPS, which is on by default
    at.toggle[0].set_value(False).run()
    # Select the data split method
    at.selectbox[1].select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    at.number_input[0].set_value(holdout_or_k).run()
    # Select Random Forest
    at.toggle[2].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()


@pytest.mark.parametrize(
    "data_split_method,holdout_or_k",
    [
        (DataSplitMethods.Holdout.capitalize(), 3),
        (DataSplitMethods.KFold.capitalize(), 3),
    ],
)
def test_manual_xgboost(
    new_experiment: str, data_split_method: DataSplitMethods, holdout_or_k: int
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=60)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Unselect AHPS, which is on by default
    at.toggle[0].set_value(False).run()
    # Select the data split method
    at.selectbox[1].select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    at.number_input[0].set_value(holdout_or_k).run()
    # Select XGBoost
    at.toggle[3].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()


@pytest.mark.parametrize(
    "data_split_method,holdout_or_k",
    [
        (DataSplitMethods.Holdout.capitalize(), 3),
        (DataSplitMethods.KFold.capitalize(), 3),
    ],
)
def test_manual_svm(
    new_experiment: str, data_split_method: DataSplitMethods, holdout_or_k: int
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=60)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Unselect AHPS, which is on by default
    at.toggle[0].set_value(False).run()
    # Select the data split method
    at.selectbox[1].select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    at.number_input[0].set_value(holdout_or_k).run()
    # Select SVM
    at.toggle[4].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()
