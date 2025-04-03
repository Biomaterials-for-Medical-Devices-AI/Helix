from pathlib import Path

import pytest

import helix.options.file_paths as fp


def test_helix_experiments_base_dir():
    # Arrange
    expected_output = Path.home() / "HelixExperiments"

    # Act
    actual_output = fp.helix_experiments_base_dir()

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_uploaded_file_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    file_name = "test_data.csv"
    expected_output = experiment_path / file_name

    # Act
    actual_output = fp.uploaded_file_path(file_name, experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


@pytest.mark.parametrize(
    "file_name,expected_output",
    [
        (
            "test_data.csv",
            fp.helix_experiments_base_dir()
            / "TestExperiment"
            / "test_data_preprocessed.csv",
        ),
        (
            "test_data.xlsx",
            fp.helix_experiments_base_dir()
            / "TestExperiment"
            / "test_data_preprocessed.xlsx",
        ),
    ],
)
def test_raw_data_path(file_name: str, expected_output: Path):
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"

    # Act
    actual_output = fp.preprocessed_data_path(file_name, experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_log_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "logs"

    # Act
    actual_output = fp.log_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_plot_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plots" / "ml"

    # Act
    actual_output = fp.ml_plot_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_model_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "models"

    # Act
    actual_output = fp.ml_model_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_metrics_mean_std_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = (
        experiment_path / "results" / "ml_metrics" / "metrics_mean_std.json"
    )

    # Act
    actual_output = fp.ml_metrics_mean_std_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_predictions_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "results" / "ml_metrics" / "predictions.csv"

    # Act
    actual_output = fp.ml_predictions_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fi_plot_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plots" / "fi"

    # Act
    actual_output = fp.fi_plot_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fi_result_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "results" / "fi"

    # Act
    actual_output = fp.fi_result_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fi_options_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "options" / "fi"

    # Act
    actual_output = fp.fi_options_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fuzzy_plot_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plots" / "fuzzy"

    # Act
    actual_output = fp.fuzzy_plot_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fuzzy_result_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "results" / "fuzzy"

    # Act
    actual_output = fp.fuzzy_result_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_data_analysis_plots_dir():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plots" / "data_analysis"

    # Act
    actual_output = fp.data_analysis_plots_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_plot_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plot_options.json"

    # Act
    actual_output = fp.plot_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_execution_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "execution_options.json"

    # Act
    actual_output = fp.execution_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_data_preprocessing_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "data_preprocessing_options.json"

    # Act
    actual_output = fp.data_preprocessing_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "ml_options.json"

    # Act
    actual_output = fp.ml_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fi_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "fi_options.json"

    # Act
    actual_output = fp.fi_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_fuzzy_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "fuzzy_options.json"

    # Act
    actual_output = fp.fuzzy_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_data_options_path():
    # Arrange
    experiment_path = fp.helix_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "data_options.json"

    # Act
    actual_output = fp.data_options_path(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output
