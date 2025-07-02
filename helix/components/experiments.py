from pathlib import Path

import streamlit as st

from helix.options.enums import ExecutionStateKeys, ViewExperimentKeys


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


def model_selector(options: list, gui_text: str, placeholder: str, key: str) -> Path:
    """Select a model or models for their intended use. This function creates a multiselect widget
    to allow the user to select multiple models to use for FI pipeline or to Predict on new datasets.

    Args:
        options (list): The list of model names to choose from.
        gui_text (str): The text to display above the widget indicating what the models are meant to be used for.
        placeholder (str): The placeholder to show on the widget before the user does the selection.
        key (str): the key to use to save the information of the widget in the state session.

    Returns:
        Path: The path to the model on disk.
    """

    return st.multiselect(
        gui_text,
        options=options,
        default=None,
        placeholder=placeholder,
        key=key,
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
