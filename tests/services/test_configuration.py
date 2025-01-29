import dataclasses
import json
import logging
from pathlib import Path
from typing import Generator
import pytest

from biofefi.options.execution import ExecutionOptions
from biofefi.options.choices import DATA_SPLITS
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.file_paths import (
    execution_options_path,
    fi_options_path,
    fuzzy_options_path,
    ml_options_path,
    plot_options_path,
)
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.configuration import (
    load_execution_options,
    load_fi_options,
    load_plot_options,
    save_options,
)


@pytest.fixture
def execution_opts() -> ExecutionOptions:
    """Produce a test instance of `ExecutionOptions`.

    Returns:
        ExecutionOptions: The test instance.
    """
    # Arrange
    return ExecutionOptions(data_path="test_data.csv", data_split=DATA_SPLITS[0])


@pytest.fixture
def execution_opts_file_path() -> Generator[Path, None, None]:
    """Produce the test `Path` to some execution options.

    Delete the file if it has been created by a test.

    Yields:
        Generator[Path, None, None]: The `Path` to the execution options file.
    """
    # Arrange
    experiment_path = Path.cwd()
    options_file = execution_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def execution_opts_file(
    execution_opts: ExecutionOptions,
    execution_opts_file_path: Generator[Path, None, None],
) -> Path:
    """Saves and `ExecutionOptions` object to a file given by `execution_opts_file_path`
    and returns the `Path` to that file.

    Cleanup is handled by the `execution_opts_file_path` fixture passed in the second
    argument.

    Args:
        execution_opts (ExecutionOptions): Exexution options fixture.
        execution_opts_file_path (Generator[Path, None, None]): File path fixture.

    Returns:
        Path: `execution_opts_file_path`
    """
    # Arrange
    options_json = dataclasses.asdict(execution_opts)
    with open(execution_opts_file_path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)
    return execution_opts_file_path


@pytest.fixture
def plotting_opts() -> PlottingOptions:
    """Produce a test instance of `PlottingOptions`.

    Returns:
        PlottingOptions: The test instance.
    """
    # Arrange
    return PlottingOptions(
        plot_axis_font_size=8,
        plot_axis_tick_size=5,
        plot_colour_scheme="fancy_colours",  # not real but doesn't matter for testing
        angle_rotate_xaxis_labels=0,
        angle_rotate_yaxis_labels=0,
        save_plots=True,
        plot_font_family="sans-serif",
        plot_title_font_size=14,
    )


@pytest.fixture
def plotting_opts_file_path() -> Generator[Path, None, None]:
    """Produce the test `Path` to some plotting options.

    Delete the file if it has been created by a test.

    Yields:
        Generator[Path, None, None]: The `Path` to the plotting options file.
    """
    # Arrange
    experiment_path = Path.cwd()
    options_file = plot_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def plotting_opts_file(
    plotting_opts: PlottingOptions, plotting_opts_file_path: Generator[Path, None, None]
) -> Path:
    """Saves and `PlottingOptions` object to a file given by `plotting_opts_file_path`
    and returns the `Path` to that file.

    Cleanup is handled by the `plotting_opts_file_path` fixture passed in the second
    argument.

    Args:
        plotting_opts (PlottingOptions): Plotting options fixture.
        plotting_opts_file_path (Generator[Path, None, None]): File path fixture.

    Returns:
        Path: `plotting_opts_file_path`
    """
    # Arrange
    options_json = dataclasses.asdict(plotting_opts)
    with open(plotting_opts_file_path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)
    return plotting_opts_file_path


@pytest.fixture
def ml_opts():
    # Arrange
    return MachineLearningOptions(
        model_types={}
    )  # no need to specify any model types, use empty dict


@pytest.fixture
def ml_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = ml_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def fi_opts() -> FeatureImportanceOptions:
    """Produce a test instance of `FeatureImportanceOptions`.

    Returns:
        FeatureImportanceOptions: The test instance.
    """
    # Arrange
    return FeatureImportanceOptions(
        global_importance_methods={},
        feature_importance_ensemble={},
        local_importance_methods={},
    )  # no need to specify any of these, use empty dict


@pytest.fixture
def fi_opts_file_path() -> Generator[Path, None, None]:
    """Produce the test `Path` to some FI options.

    Delete the file if it has been created by a test.

    Yields:
        Generator[Path, None, None]: The `Path` to the FI options file.
    """
    # Arrange
    experiment_path = Path.cwd()
    options_file = fi_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def fi_opts_file(
    fi_opts: FeatureImportanceOptions, fi_opts_file_path: Generator[Path, None, None]
) -> Path:
    """Saves and `FeatureImportanceOptions` object to a file given by `fi_opts_file_path`
    and returns the `Path` to that file.

    Cleanup is handled by the `fi_opts_file_path` fixture passed in the second
    argument.

    Args:
        fi_opts (FeatureImportanceOptions): FI options fixture.
        fi_opts_file_path (Generator[Path, None, None]): File path fixture.

    Returns:
        Path: `fi_opts_file_path`
    """
    # Arrange
    options_json = dataclasses.asdict(fi_opts)
    with open(fi_opts_file_path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)
    return fi_opts_file_path


@pytest.fixture
def fuzzy_opts():
    # Arrange
    return FuzzyOptions(cluster_names=[])  # no need to specify this, use empty list


@pytest.fixture
def fuzzy_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = fuzzy_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


def test_save_execution_opts(execution_opts, execution_opts_file_path):
    # Act
    save_options(execution_opts_file_path, execution_opts)

    # Assert
    assert execution_opts_file_path.exists()


def test_save_plotting_opts(plotting_opts, plotting_opts_file):
    # Act
    save_options(plotting_opts_file, plotting_opts)

    # Assert
    assert plotting_opts_file.exists()


def test_save_ml_opts(ml_opts, ml_opts_file):
    # Act
    save_options(ml_opts_file, ml_opts)

    # Assert
    assert ml_opts_file.exists()


def test_save_fi_opts(fi_opts, fi_opts_file_path):
    # Act
    save_options(fi_opts_file_path, fi_opts)

    # Assert
    assert fi_opts_file_path.exists()


def test_save_fuzzy_opts(fuzzy_opts, fuzzy_opts_file):
    # Act
    save_options(fuzzy_opts_file, fuzzy_opts)

    # Assert
    assert fuzzy_opts_file.exists()


def test_load_execution_options(execution_opts, execution_opts_file):
    # Act
    opts = load_execution_options(execution_opts_file)

    # Assert
    assert isinstance(opts, ExecutionOptions)
    assert opts == execution_opts


def test_load_plot_options(plotting_opts, plotting_opts_file):
    # Act
    opts = load_plot_options(plotting_opts_file)

    # Assert
    assert isinstance(opts, PlottingOptions)
    assert opts == plotting_opts


def test_load_fi_options(fi_opts, fi_opts_file):
    # Arrange
    path_to_non_existent = Path("non_existent.json")  # test the `None` case

    # Act
    opts = load_fi_options(fi_opts_file)
    non_existent_opts = load_fi_options(path_to_non_existent)

    # Assert
    assert isinstance(opts, FeatureImportanceOptions)
    assert opts == fi_opts
    assert non_existent_opts is None
