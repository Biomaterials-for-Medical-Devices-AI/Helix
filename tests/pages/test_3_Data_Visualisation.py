import uuid
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from streamlit.testing.v1 import AppTest

from helix.options.data import DataOptions
from helix.options.enums import ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_analysis_plots_dir,
    data_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    plot_options_path,
)
from helix.options.file_paths import (
    preprocessed_data_path as get_preprocessed_data_path,
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
        width=10,
        height=10,
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

    # Save raw data
    raw_data_path = Path(data_opts.data_path.replace("_preprocessed", ""))
    np.savetxt(raw_data_path, X=dummy_data, delimiter=",")

    # Save preprocessed data
    preprocessed_path = get_preprocessed_data_path(str(raw_data_path), experiment_dir)
    create_directory(preprocessed_path.parent)
    np.savetxt(preprocessed_path, X=dummy_data, delimiter=",")

    yield execution_opts.experiment_name

    if experiment_dir.exists():
        delete_directory(experiment_dir)


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_raw_tab_displays(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert at.tabs[0].label == "Raw Data"


def test_page_second_tab_displays(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert at.tabs[1].label == "Raw Data Statistics"


def test_page_headings_display(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert any("Graphical Description" in header.value for header in at.markdown if hasattr(header, 'value'))


def test_page_normality_test_displays(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    # Check for the statistical tests section header
    assert any("statistical tests" in text.value.lower() for text in at.markdown if hasattr(text, 'value'))


def test_experiment_directory_exists(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert experiment_dir.exists(), f"Experiment directory {experiment_dir} does not exist"
    assert plot_dir.exists() or plot_dir.parent.exists(), f"Plot directory {plot_dir} or its parent does not exist"


def test_experiment_selector_exists():
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)

    # Act
    at.run()

    # Assert
    assert len(at.selectbox) > 0


def test_experiment_selector_works(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception


def test_statistical_tests_section_exists(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    # Check for the statistical tests section header
    assert any("statistical tests" in text.value.lower() for text in at.markdown if hasattr(text, 'value'))


def test_graphical_description_section_exists(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    # Check for the graphical description section header
    assert any("graphical description" in text.value.lower() for text in at.markdown if hasattr(text, 'value'))


def test_visualization_tabs_exist(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    # Check for visualization tabs
    visualization_tabs = [tab for tab in at.tabs if tab.label in ["Raw Data", "Preprocessed Data"]]
    assert len(visualization_tabs) >= 2
    assert any("Raw Data" in tab.label for tab in visualization_tabs)
    assert any("Preprocessed Data" in tab.label for tab in visualization_tabs)


def test_experiment_directory_exists(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    base_dir = helix_experiments_base_dir()
    experiment_dir = base_dir / new_experiment
    plot_dir = data_analysis_plots_dir(experiment_dir)

    # Act
    # Select experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert experiment_dir.exists()
    assert plot_dir.exists()
    assert (experiment_dir / plot_options_path(experiment_dir).name).exists()


def test_data_statistics_tabs_exist(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    # Check for data statistics tabs
    statistics_tabs = [tab for tab in at.tabs if "Statistics" in tab.label]
    assert len(statistics_tabs) >= 2
    assert any("Raw Data Statistics" in tab.label for tab in statistics_tabs)
    assert any("Preprocessed Data Statistics" in tab.label for tab in statistics_tabs)


def test_visualization_components_exist(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/3_Data_Visualisation.py", default_timeout=60)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    # Check for all visualization components
    assert any("Target Variable Distribution" in text.value for text in at.markdown if hasattr(text, 'value'))
    assert any("Correlation Heatmap" in text.value for text in at.markdown if hasattr(text, 'value'))
    assert any("Pairplot" in text.value for text in at.markdown if hasattr(text, 'value'))
    assert any("t-SNE Plot" in text.value for text in at.markdown if hasattr(text, 'value'))
