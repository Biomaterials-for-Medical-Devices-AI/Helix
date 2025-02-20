from pathlib import Path

from biofefi.options.data import DataOptions, DataSplitOptions
from biofefi.options.execution import ExecutionOptions
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.options.preprocessing import PreprocessingOptions
from biofefi.services.configuration import (
    load_data_options,
    load_data_preprocessing_options,
    load_execution_options,
    load_fi_options,
    load_plot_options,
    save_options,
)

# import all the fixtures for services
from .fixtures import *  # noqa: F403, F401


def test_save_execution_opts(
    execution_opts: ExecutionOptions, execution_opts_file_path: Path
):
    # Act
    save_options(execution_opts_file_path, execution_opts)

    # Assert
    assert execution_opts_file_path.exists()


def test_save_plotting_opts(plotting_opts: PlottingOptions, plotting_opts_file: Path):
    # Act
    save_options(plotting_opts_file, plotting_opts)

    # Assert
    assert plotting_opts_file.exists()


def test_save_ml_opts(ml_opts: MachineLearningOptions, ml_opts_file: Path):
    # Act
    save_options(ml_opts_file, ml_opts)

    # Assert
    assert ml_opts_file.exists()


def test_save_fi_opts(fi_opts: FeatureImportanceOptions, fi_opts_file_path: Path):
    # Act
    save_options(fi_opts_file_path, fi_opts)

    # Assert
    assert fi_opts_file_path.exists()


def test_save_fuzzy_opts(fuzzy_opts: FuzzyOptions, fuzzy_opts_file_path: Path):
    # Act
    save_options(fuzzy_opts_file_path, fuzzy_opts)

    # Assert
    assert fuzzy_opts_file_path.exists()


def test_load_execution_options(
    execution_opts: ExecutionOptions, execution_opts_file: Path
):
    # Act
    opts = load_execution_options(execution_opts_file)

    # Assert
    assert isinstance(opts, ExecutionOptions)
    assert opts == execution_opts


def test_load_plot_options(plotting_opts: PlottingOptions, plotting_opts_file: Path):
    # Act
    opts = load_plot_options(plotting_opts_file)

    # Assert
    assert isinstance(opts, PlottingOptions)
    assert opts == plotting_opts


def test_load_fi_options(fi_opts: FeatureImportanceOptions, fi_opts_file: Path):
    # Arrange
    path_to_non_existent = Path("non_existent.json")  # test the `None` case

    # Act
    opts = load_fi_options(fi_opts_file)
    non_existent_opts = load_fi_options(path_to_non_existent)

    # Assert
    assert isinstance(opts, FeatureImportanceOptions)
    assert opts == fi_opts
    assert non_existent_opts is None


def test_save_data_preprocessing_opts(
    data_preprocessing_opts: PreprocessingOptions,
    data_preprocessing_opts_file_path: Path,
):
    # Act
    save_options(data_preprocessing_opts_file_path, data_preprocessing_opts)

    # Assert
    assert data_preprocessing_opts_file_path.exists()


def test_load_data_preprocessing_options(
    data_preprocessing_opts: PreprocessingOptions, data_preprocessing_opts_file: Path
):
    # Act
    opts = load_data_preprocessing_options(data_preprocessing_opts_file)

    # Assert
    assert isinstance(opts, PreprocessingOptions)
    assert opts == data_preprocessing_opts


def test_save_data_opts(
    data_opts: DataOptions,
    data_opts_file_path: Path,
):
    # Act
    save_options(data_opts_file_path, data_opts)

    # Assert
    assert data_opts_file_path.exists()


def test_load_data_options(data_opts: DataOptions, data_opts_file: Path):
    # Act
    opts = load_data_options(data_opts_file)

    # Assert
    assert isinstance(opts, DataOptions)
    assert isinstance(opts.data_split, DataSplitOptions)
    assert opts == data_opts
