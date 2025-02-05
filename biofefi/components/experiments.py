from pathlib import Path

import streamlit as st

from biofefi.options.enums import (
    ExecutionStateKeys,
    FeatureImportanceStateKeys,
    ViewExperimentKeys,
)


def experiment_selector(options: list) -> str:
    """Select

    Args:
        options (list): The list of experiment names to choose from.

    Returns:
        str: The name of the experiment on disk.
    """

    return st.selectbox(
        "Select an experiment",
        options=options,
        index=None,
        placeholder="Experiment name",
        key=ViewExperimentKeys.ExperimentName,
    )


def model_selector(options: list) -> Path:
    """Select a model or models to explain. This function creates a multiselect widget
    to allow the user to select multiple models to explain using the FI pipeline.

    Args:
        options (list): The list of model names to choose from.

    Returns:
        Path: The path to the model on disk.
    """

    return st.multiselect(
        "Select a model to explain",
        options=options,
        default=None,
        placeholder="Models to explain",
        key=FeatureImportanceStateKeys.ExplainModels,
    )


def data_selector(options: list) -> Path:
    """Select a model or models to explain. This function creates a multiselect widget
    to allow the user to select multiple models to explain using the FI pipeline.

    Args:
        options (list): The list of model names to choose from.

    Returns:
        Path: The path to the model on disk.
    """

    return st.selectbox(
        "Select a dataset to explain",
        options=options,
        index=None,
        placeholder="Dataset to explain",
        key=ExecutionStateKeys.UploadedFileName,
    )
