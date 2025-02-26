# 3. Fixture for PreprocessingOptions to test behaviour when they already exist
# 6. Test the page produces fresh PreprocessingOptions json

from pathlib import Path
import uuid
import numpy as np
import pytest
from sklearn.datasets import make_classification
from streamlit.testing.v1 import AppTest

from helix.options.choices.ui import NORMALISATIONS
from helix.options.data import DataOptions
from helix.options.enums import DataPreprocessingStateKeys, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    biofefi_experiments_base_dir,
    data_options_path,
    data_preprocessing_options_path,
    execution_options_path,
    plot_options_path,
    preprocessed_data_path,
)
from helix.options.plotting import PlottingOptions
from helix.options.preprocessing import PreprocessingOptions
from helix.services.configuration import load_data_preprocessing_options, save_options
from helix.utils.utils import create_directory, delete_directory


@pytest.fixture
def execution_opts():
    experiment_name = str(uuid.uuid4())
    dependent_variable = "test"
    problem_type = ProblemTypes.Classification
    return ExecutionOptions(
        experiment_name=experiment_name,
        dependent_variable=dependent_variable,
        problem_type=problem_type,
    )


@pytest.fixture
def plotting_opts():
    return PlottingOptions(
        plot_axis_font_size=8,
        plot_axis_tick_size=8,
        plot_colour_scheme="Solarize_Light2",
        angle_rotate_xaxis_labels=10,
        angle_rotate_yaxis_labels=60,
        save_plots=True,
        plot_title_font_size=20,
        plot_colour_map="viridis",
        plot_font_family="sans-serif",
    )


@pytest.fixture
def data_opts(execution_opts: ExecutionOptions):
    data_file_name = (
        biofefi_experiments_base_dir()
        / execution_opts.experiment_name
        / "data_file.csv"
    )
    return DataOptions(data_path=str(data_file_name))


@pytest.fixture
def old_preproc_opts():
    return PreprocessingOptions(
        feature_selection_methods={
            DataPreprocessingStateKeys.VarianceThreshold: True,
            DataPreprocessingStateKeys.CorrelationThreshold: True,
            DataPreprocessingStateKeys.LassoFeatureSelection: True,
        },
        variance_threshold=0.10,
        correlation_threshold=0.80,
        lasso_regularisation_term=0.05,
        independent_variable_normalisation=NORMALISATIONS[0].lower(),
        dependent_variable_transformation=NORMALISATIONS[-1].lower(),
        data_is_preprocessed=True,
    )


@pytest.fixture
def dummy_data(execution_opts: ExecutionOptions):
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=4,
        random_state=execution_opts.random_state,
    )
    data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    return data


@pytest.fixture
def new_experiment(
    execution_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    data_opts: DataOptions,
    dummy_data: np.ndarray,
):
    base_dir = biofefi_experiments_base_dir()
    experiment_dir = base_dir / execution_opts.experiment_name
    create_directory(experiment_dir)

    exec_opts_file_path = execution_options_path(experiment_dir)
    save_options(exec_opts_file_path, execution_opts)

    plot_opts_file_path = plot_options_path(experiment_dir)
    save_options(plot_opts_file_path, plotting_opts)

    data_opts_file_path = data_options_path(experiment_dir)
    save_options(data_opts_file_path, data_opts)

    np.savetxt(data_opts.data_path, X=dummy_data, delimiter=",")

    yield execution_opts.experiment_name

    if experiment_dir.exists():
        delete_directory(experiment_dir)


@pytest.fixture
def old_experiment(
    execution_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    data_opts: DataOptions,
    dummy_data: np.ndarray,
    old_preproc_opts: PreprocessingOptions,
):
    base_dir = biofefi_experiments_base_dir()
    experiment_dir = base_dir / execution_opts.experiment_name
    create_directory(experiment_dir)

    exec_opts_file_path = execution_options_path(experiment_dir)
    save_options(exec_opts_file_path, execution_opts)

    plot_opts_file_path = plot_options_path(experiment_dir)
    save_options(plot_opts_file_path, plotting_opts)

    data_opts_file_path = data_options_path(experiment_dir)
    save_options(data_opts_file_path, data_opts)

    preproc_file_path = data_preprocessing_options_path(experiment_dir)
    save_options(preproc_file_path, old_preproc_opts)

    np.savetxt(data_opts.data_path, X=dummy_data, delimiter=",")

    yield execution_opts.experiment_name

    if experiment_dir.exists():
        delete_directory(experiment_dir)


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/2_Data_Preprocessing.py")

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/2_Data_Preprocessing.py")
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_produces_preprocessed_data_file(
    new_experiment: str,
    execution_opts: ExecutionOptions,
    data_opts: DataOptions,
):
    # Arrange
    at = AppTest.from_file("helix/pages/2_Data_Preprocessing.py")
    at.run()

    expected_file = preprocessed_data_path(
        Path(data_opts.data_path).name,
        biofefi_experiments_base_dir() / execution_opts.experiment_name,
    )

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # normalise the features
    at.selectbox[1].select(NORMALISATIONS[0]).run()  # standardisation
    # select variance threshold
    at.checkbox[0].check().run()
    # select correlation threshold
    at.checkbox[1].check().run()
    # select Lasso
    at.checkbox[2].check().run()
    # click the button
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()


def test_page_produces_preprocessing_options_file(
    new_experiment: str, execution_opts: ExecutionOptions
):
    # Arrange
    at = AppTest.from_file("helix/pages/2_Data_Preprocessing.py")
    at.run()

    expected_file = data_preprocessing_options_path(
        biofefi_experiments_base_dir() / execution_opts.experiment_name,
    )
    expected_opts = PreprocessingOptions(
        feature_selection_methods={
            DataPreprocessingStateKeys.VarianceThreshold: True,
            DataPreprocessingStateKeys.CorrelationThreshold: True,
            DataPreprocessingStateKeys.LassoFeatureSelection: True,
        },
        variance_threshold=0.10,
        correlation_threshold=0.80,
        lasso_regularisation_term=0.05,
        independent_variable_normalisation=NORMALISATIONS[0].lower(),
        dependent_variable_transformation=NORMALISATIONS[-1].lower(),
        data_is_preprocessed=True,
    )

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # normalise the features
    at.selectbox[1].select(NORMALISATIONS[0]).run()  # standardisation
    # select variance threshold
    at.checkbox[0].check().run()
    # select correlation threshold
    at.checkbox[1].check().run()
    # select Lasso
    at.checkbox[2].check().run()
    # click the button
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()
    actual_opts = load_data_preprocessing_options(expected_file)
    assert actual_opts == expected_opts


def test_page_detects_old_opts(
    old_experiment: str,
):
    # Arrange
    at = AppTest.from_file("helix/pages/2_Data_Preprocessing.py")
    at.run()

    # Act
    # select the experiment
    at.selectbox[0].select(old_experiment).run()
    # look for the redo elements
    redo_warning = at.warning[0]
    redo_checkbox = at.checkbox[0]

    # Assert
    assert not at.exception
    assert not at.error
    # look
    assert redo_checkbox.label == "Redo preprocessing"
    assert (
        redo_warning.value
        == "Your data are already preprocessed. Would you like to start again?"
    )
