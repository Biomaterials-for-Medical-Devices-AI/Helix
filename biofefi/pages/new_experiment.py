from pathlib import Path
import streamlit as st
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.navigation import navbar
from biofefi.options.enums import ConfigStateKeys
from biofefi.options.file_paths import biofefi_experiments_base_dir
from biofefi.utils.utils import create_directory


def _directory_is_valid(directory: Path) -> bool:
    """Determine if the directory supplied by the user is valid. If it already exists,
    it is invalid.

    Args:
        directory (Path): The path to check.

    Returns:
        bool: `True` if the directory doesn't already exist, else `False`
    """
    return not directory.exists()


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = biofefi_experiments_base_dir()

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    col1.text(f"{root}/", help="Your experiment will be saved here")
    sub_dir = col2.text_input("Name of the experiment", placeholder="e.g. MyExperiment")

    return root / sub_dir


st.set_page_config(
    page_title="New Experiment",
    page_icon=sidebar_logo(),
)
navbar()

st.header("New Experiment")
st.write(
    """
    Here you can start a new experiment. Once you create one, you will be able to select it
    on the Machine Learning & Feature Importance pages.
    """
)
st.write(
    """
    ### Create a new experiment ⚗️

    Give your experiment a name and click **Create**. If an experiment with the same name
    already exists you will not be able to create it again.
    """
)

save_dir = _save_directory_selector()
# If a user has tried to enter a destination to save an experiment, show it
# if it's valid, else show some red text showing the destination and saying
# it's invalid.
is_valid = _directory_is_valid(save_dir)
if not is_valid and st.session_state.get(ConfigStateKeys.ExperimentName):
    st.markdown(f":red[Cannot use {save_dir}; it already exists.]")
else:
    st.session_state[ConfigStateKeys.ExperimentName] = save_dir

st.button(
    "Create",
    type="primary",
    disabled=not is_valid,
    on_click=create_directory,
    args=(save_dir,),
)
