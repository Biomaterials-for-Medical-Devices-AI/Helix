import uuid

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from helix.options.data import DataOptions
from helix.options.enums import ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    plot_options_path,
)
from helix.options.plotting import PlottingOptions
from helix.services.configuration import save_options
from helix.utils.utils import create_directory, delete_directory


@pytest.fixture
def classification_execution_opts():
    experiment_name = str(uuid.uuid4())
    dependent_variable = "test"
    problem_type = ProblemTypes.Classification
    return ExecutionOptions(
        experiment_name=experiment_name,
        dependent_variable=dependent_variable,
        problem_type=problem_type,
    )


@pytest.fixture
def regression_execution_opts():
    experiment_name = str(uuid.uuid4())
    dependent_variable = "test"
    problem_type = ProblemTypes.Regression
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
        dpi=150,
        width=10,
        height=10,
    )


@pytest.fixture
def classification_data_opts(classification_execution_opts: ExecutionOptions):
    data_file_name = (
        helix_experiments_base_dir()
        / classification_execution_opts.experiment_name
        / "data_file.csv"
    )
    return DataOptions(data_path=str(data_file_name))


@pytest.fixture
def regression_data_opts(regression_execution_opts: ExecutionOptions):
    data_file_name = (
        helix_experiments_base_dir()
        / regression_execution_opts.experiment_name
        / "data_file.csv"
    )
    return DataOptions(data_path=str(data_file_name))


@pytest.fixture
def dummy_classification_data(classification_execution_opts: ExecutionOptions):
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=4,
        random_state=classification_execution_opts.random_state,
    )
    data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    return data


@pytest.fixture
def dummy_regression_data(regression_execution_opts: ExecutionOptions):
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=4,
        random_state=regression_execution_opts.random_state,
    )
    data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    return data


@pytest.fixture
def new_classification_experiment(
    classification_execution_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    classification_data_opts: DataOptions,
    dummy_classification_data: np.ndarray,
):
    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / classification_execution_opts.experiment_name
    create_directory(experiment_dir)

    exec_opts_file_path = execution_options_path(experiment_dir)
    save_options(exec_opts_file_path, classification_execution_opts)

    plot_opts_file_path = plot_options_path(experiment_dir)
    save_options(plot_opts_file_path, plotting_opts)

    data_opts_file_path = data_options_path(experiment_dir)
    save_options(data_opts_file_path, classification_data_opts)

    np.savetxt(
        classification_data_opts.data_path, X=dummy_classification_data, delimiter=","
    )

    yield classification_execution_opts.experiment_name

    if experiment_dir.exists():
        delete_directory(experiment_dir)


@pytest.fixture
def new_regression_experiment(
    regression_execution_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    regression_data_opts: DataOptions,
    dummy_regression_data: np.ndarray,
):
    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / regression_execution_opts.experiment_name
    create_directory(experiment_dir)

    exec_opts_file_path = execution_options_path(experiment_dir)
    save_options(exec_opts_file_path, regression_execution_opts)

    plot_opts_file_path = plot_options_path(experiment_dir)
    save_options(plot_opts_file_path, plotting_opts)

    data_opts_file_path = data_options_path(experiment_dir)
    save_options(data_opts_file_path, regression_data_opts)

    np.savetxt(regression_data_opts.data_path, X=dummy_regression_data, delimiter=",")

    yield regression_execution_opts.experiment_name

    if experiment_dir.exists():
        delete_directory(experiment_dir)
