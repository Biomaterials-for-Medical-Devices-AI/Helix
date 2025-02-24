import dataclasses
import json
from pathlib import Path
from typing import TypeVar

import streamlit as st

from helix.options.data import DataOptions, DataSplitOptions
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.fuzzy import FuzzyOptions
from helix.options.ml import MachineLearningOptions
from helix.options.plotting import PlottingOptions
from helix.options.preprocessing import PreprocessingOptions

Options = TypeVar(
    "Options",
    DataOptions,
    ExecutionOptions,
    PlottingOptions,
    MachineLearningOptions,
    FeatureImportanceOptions,
    FuzzyOptions,
    PreprocessingOptions,
)


def load_execution_options(path: Path) -> ExecutionOptions:
    """Load experiment execution options from the given path.
    The path will be to a `json` file containing the options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        ExecutionOptions: The execution options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    options = ExecutionOptions(**options_json)
    return options


def load_plot_options(path: Path) -> PlottingOptions:
    """Load plotting options from the given path.
    The path will be to a `json` file containing the plot options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        PlottingOptions: The plotting options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    options = PlottingOptions(**options_json)
    return options


def save_options(path: Path, options: Options):
    """Save options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (T): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)


def load_fi_options(path: Path) -> FeatureImportanceOptions | None:
    """Load feature importance options.

    Args:
        path (Path): The path to the feature importance options file.

    Returns:
        FeatureImportanceOptions | None: The feature importance options.
    """

    try:
        with open(path, "r") as file:
            fi_json_options = json.load(file)
            fi_options = FeatureImportanceOptions(**fi_json_options)
    except FileNotFoundError:
        fi_options = None
    except TypeError:
        fi_options = None

    return fi_options


def load_fuzzy_options(path: Path) -> FuzzyOptions | None:
    """Load fuzzy options.

    Args:
        path (Path): The path to the fuzzy options file.

    Returns:
        FuzzyOptions | None: The fuzzy options.
    """

    try:
        with open(path, "r") as file:
            fuzzy_json_options = json.load(file)
            fuzzy_options = FuzzyOptions(**fuzzy_json_options)
    except FileNotFoundError:
        fuzzy_options = None
    except TypeError:
        fuzzy_options = None

    return fuzzy_options


def load_data_preprocessing_options(path: Path) -> PreprocessingOptions:
    """Load data preprocessing options from the given path.
    The path will be to a `json` file containing the options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        PreprocessingOptions: The data preprocessing options.
    """

    try:
        with open(path, "r") as json_file:
            options_json = json.load(json_file)
        preprocessing_options = PreprocessingOptions(**options_json)
    except FileNotFoundError:
        preprocessing_options = None
    except TypeError:
        preprocessing_options = None
    return preprocessing_options


def load_data_options(path: Path) -> DataOptions:
    """Load the data options from the JSON file given in `path`.

    Args:
        path (Path): The path to the JSON file containing the data options.

    Returns:
        DataOptions: The data options.
    """
    with open(path, "r") as json_file:
        options_json: dict = json.load(json_file)
    if split_opts := options_json.get("data_split"):
        options_json["data_split"] = DataSplitOptions(**split_opts)
    options = DataOptions(**options_json)
    return options


def load_ml_options(path: Path) -> MachineLearningOptions:
    """Load machine learning options from the given path.
    The path will be to a `json` file containing the options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        MachineLearningOptions: The machine learning options.
    """
    try:
        with open(path, "r") as json_file:
            options_json = json.load(json_file)
        options = MachineLearningOptions(**options_json)
    except FileNotFoundError:
        options = None
    except TypeError:
        options = None

    return options
