import pytest
from streamlit.testing.v1 import AppTest

from helix.options.enums import DataSplitMethods, ExecutionStateKeys, ViewExperimentKeys
from helix.options.file_paths import (
    helix_experiments_base_dir,
    log_dir,
    ml_metrics_path,
    ml_model_dir,
    ml_plot_dir,
    ml_predictions_path,
)
from tests.utils import get_element_by_key, get_element_by_label

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
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()

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
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Unselect AHPS, which is on by default
    ahps_toggle = get_element_by_key(
        at, "toggle", ExecutionStateKeys.UseHyperParamSearch
    )
    ahps_toggle.set_value(False).run()
    # Select the data split method
    data_split_selector = get_element_by_label(at, "selectbox", "Data split method")
    data_split_selector.select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    if holdout_input := get_element_by_label(
        at, "number_input", "Number of bootstraps"
    ):
        holdout_input.set_value(holdout_or_k).run()
    if k_input := get_element_by_label(at, "number_input", "k"):
        k_input.set_value(holdout_or_k).run()
    # Select Linear Model
    lm_toggle = get_element_by_label(at, "toggle", "Linear Model")
    lm_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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


def test_auto_linear_model(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Set the number of k-folds
    k_input = get_element_by_label(at, "number_input", "k")
    k_input.set_value(k).run()
    # Select Linear Model
    lm_toggle = get_element_by_label(at, "toggle", "Linear Model")
    lm_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Unselect AHPS, which is on by default
    ahps_toggle = get_element_by_key(
        at, "toggle", ExecutionStateKeys.UseHyperParamSearch
    )
    ahps_toggle.set_value(False).run()
    # Select the data split method
    data_split_selector = get_element_by_label(at, "selectbox", "Data split method")
    data_split_selector.select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    if holdout_input := get_element_by_label(
        at, "number_input", "Number of bootstraps"
    ):
        holdout_input.set_value(holdout_or_k).run()
    if k_input := get_element_by_label(at, "number_input", "k"):
        k_input.set_value(holdout_or_k).run()
    # Select Random Forest
    rm_toggle = get_element_by_label(at, "toggle", "Random Forest")
    rm_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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


def test_auto_random_forest(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Set the number of k-folds
    k_input = get_element_by_label(at, "number_input", "k")
    k_input.set_value(k).run()
    # Select Random Forest
    rf_toggle = get_element_by_label(at, "toggle", "Random Forest")
    rf_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Unselect AHPS, which is on by default
    ahps_toggle = get_element_by_key(
        at, "toggle", ExecutionStateKeys.UseHyperParamSearch
    )
    ahps_toggle.set_value(False).run()
    # Select the data split method
    data_split_selector = get_element_by_label(at, "selectbox", "Data split method")
    data_split_selector.select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    if holdout_input := get_element_by_label(
        at, "number_input", "Number of bootstraps"
    ):
        holdout_input.set_value(holdout_or_k).run()
    if k_input := get_element_by_label(at, "number_input", "k"):
        k_input.set_value(holdout_or_k).run()
    # Select XGBoost
    xgb_toggle = get_element_by_label(at, "toggle", "XGBoost")
    xgb_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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


def test_auto_xgboost(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Set the number of k-folds
    k_input = get_element_by_label(at, "number_input", "k")
    k_input.set_value(k).run()
    # Select XGBoost
    xgb_toggle = get_element_by_label(at, "toggle", "XGBoost")
    xgb_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Unselect AHPS, which is on by default
    ahps_toggle = get_element_by_key(
        at, "toggle", ExecutionStateKeys.UseHyperParamSearch
    )
    ahps_toggle.set_value(False).run()
    # Select the data split method
    data_split_selector = get_element_by_label(at, "selectbox", "Data split method")
    data_split_selector.select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    if holdout_input := get_element_by_label(
        at, "number_input", "Number of bootstraps"
    ):
        holdout_input.set_value(holdout_or_k).run()
    if k_input := get_element_by_label(at, "number_input", "k"):
        k_input.set_value(holdout_or_k).run()
    # Select SVM
    svm_toggle = get_element_by_label(at, "toggle", "Support Vector Machine")
    svm_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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


def test_auto_svm(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Set the number of k-folds
    k_input = get_element_by_label(at, "number_input", "k")
    k_input.set_value(k).run()
    # Select SVM
    svm_toggle = get_element_by_label(at, "toggle", "Support Vector Machine")
    svm_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

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


def test_page_makes_one_log_per_run(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_log_dir = log_dir(exp_dir) / "ml"
    expected_n_log_files = 1
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_experiment).run()
    # Set the number of k-folds
    k_input = get_element_by_label(at, "number_input", "k")
    k_input.set_value(k).run()
    # Select Random Forest
    rf_toggle = get_element_by_label(at, "toggle", "Random Forest")
    rf_toggle.set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    button = get_element_by_label(at, "button", "Run Training")
    button.click().run()

    # log dir contents
    log_dir_contents = list(
        filter(lambda x: x.endswith(".log"), map(str, expected_log_dir.iterdir()))
    )

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_log_dir.exists()
    assert log_dir_contents  # directory is not empty
    assert len(log_dir_contents) == expected_n_log_files
