import uuid

import numpy as np
import pytest
from sklearn.datasets import make_classification
from streamlit.testing.v1 import AppTest

from helix.options.data import DataOptions
from helix.options.enums import Normalisations, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_analysis_plots_dir,
    data_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    plot_options_path,
)
from helix.options.plotting import PlottingOptions
from helix.services.configuration import save_options
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
        dpi=150,
    )


@pytest.fixture
def data_opts(execution_opts: ExecutionOptions):
    data_file_name = (
        helix_experiments_base_dir() / execution_opts.experiment_name / "data_file.csv"
    )
    return DataOptions(data_path=str(data_file_name))


@pytest.fixture
def dummy_data(execution_opts: ExecutionOptions):
    X, y = make_classification(
        n_samples=500,
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
    base_dir = helix_experiments_base_dir()
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


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_produces_kde_plot(new_experiment: str, execution_opts: ExecutionOptions):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    expected_file = plot_dir / f"{execution_opts.dependent_variable}_distribution.png"

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # select KDE plot
    at.toggle[0].set_value(True).run()
    # check the box to create the plot
    at.checkbox[0].check().run()
    # save the plot
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()


def test_page_produces_correlation_heatmap(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    expected_file = plot_dir / "correlation_heatmap.png"

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # select all feature
    at.toggle[1].set_value(True).run()
    # check the box to create the plot
    at.checkbox[1].check().run()
    # save the plot
    # since we only choose one visualisation, only one button is visible,
    # hence, at.button[0]
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()


def test_page_produces_pairplot(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    expected_file = plot_dir / "pairplot.png"

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # select all feature
    at.toggle[2].set_value(True).run()
    # check the box to create the plot
    at.checkbox[2].check().run()
    # save the plot
    # since we only choose one visualisation, only one button is visible,
    # hence, at.button[0]
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()


def test_page_produces_tsne_plot(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=10)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    expected_file = plot_dir / "tsne_plot.png"

    # Act
    # select the experiment
    at.selectbox[0].select(new_experiment).run()
    # select tsne normalisation
    at.selectbox[-1].select(Normalisations.Standardization)
    # check the box to create the plot
    at.checkbox[3].check().run()
    # save the plot
    # since we only choose one visualisation, only one button is visible,
    # hence, at.button[0]
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_file.exists()
