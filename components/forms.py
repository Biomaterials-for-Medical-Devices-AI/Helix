import os
from pathlib import Path
import streamlit as st

from options.enums import ConfigStateKeys, ExecutionStateKeys


def data_upload_form():
    st.header("Data Upload")
    save_dir = _save_directory_selector()
    st.text_input("Name of the experiment", key=ConfigStateKeys.ExperimentName)
    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    if not st.session_state.get(ConfigStateKeys.IsMachineLearning, False):
        st.file_uploader(
            "Upload machine leaerning models",
            type="pkl",
            accept_multiple_files=True,
            key=ConfigStateKeys.UploadedModels,
        )
    st.button("Run", key=ExecutionStateKeys.RunPipeline)


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = Path.home()

    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

    col1.text(f"{root}/")
    sub_dir = col2.text_input(label="", placeholder="Directory name")

    return root / sub_dir
