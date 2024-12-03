import os
from pathlib import Path

import streamlit as st

from biofefi.components.configuration import execution_options_box, plot_options_box
from biofefi.components.images.logos import sidebar_logo
from biofefi.options.enums import (
    ConfigStateKeys,
    Normalisations,
    PlotOptionKeys,
    ProblemTypes,
)
from biofefi.options.execution import ExecutionOptions
from biofefi.options.file_paths import biofefi_experiments_base_dir, uploaded_file_path
from biofefi.options.plotting import PlottingOptions
from biofefi.services.experiments import create_experiment
from biofefi.utils.utils import save_upload


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

    col1.text(f"{root}{os.path.sep}", help="Your experiment will be saved here")
    sub_dir = col2.text_input("Name of the experiment", placeholder="e.g. MyExperiment")

    return root / sub_dir


def _file_is_uploaded() -> bool:
    """Determine if the user has uploaded a file to the form.

    Returns:
        bool: `True` if a file was uploaded, else `False`.
    """
    return st.session_state.get(ConfigStateKeys.UploadedFileName) is not None


def _entrypoint(save_dir: Path):
    """Function to serve as the entrypoint for experiment creation, with access
    to the session state. This is so configuration captured in fragements is
    passed correctly to the services in this function.

    Args:
        save_dir (Path): The path to the experiment.
    """
    # Set up options to save
    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName].name,
        biofefi_experiments_base_dir()
        / st.session_state[ConfigStateKeys.ExperimentName],
    )
    exec_opts = ExecutionOptions(
        data_path=str(path_to_data),  # Path objects aren't JSON serialisable
        data_split=st.session_state[ConfigStateKeys.DataSplit],
        problem_type=st.session_state.get(
            ConfigStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        normalization=st.session_state.get(
            ConfigStateKeys.Normalization, Normalisations.NoNormalisation
        ).lower(),
        random_state=st.session_state[ConfigStateKeys.RandomSeed],
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
        n_bootstraps=st.session_state[ConfigStateKeys.NumberOfBootstraps],
    )
    plot_opts = PlottingOptions(
        plot_axis_font_size=st.session_state[PlotOptionKeys.AxisFontSize],
        plot_axis_tick_size=st.session_state[PlotOptionKeys.AxisTickSize],
        plot_colour_scheme=st.session_state[PlotOptionKeys.ColourScheme],
        angle_rotate_xaxis_labels=st.session_state[PlotOptionKeys.RotateXAxisLabels],
        angle_rotate_yaxis_labels=st.session_state[PlotOptionKeys.RotateYAxisLabels],
        save_plots=st.session_state[PlotOptionKeys.SavePlots],
        plot_title_font_size=st.session_state[PlotOptionKeys.TitleFontSize],
        plot_font_family=st.session_state[PlotOptionKeys.FontFamily],
    )

    # Create the experiment directory and save configs
    create_experiment(save_dir, plotting_options=plot_opts, execution_options=exec_opts)

    # Save the data
    uploaded_file = st.session_state[ConfigStateKeys.UploadedFileName]
    save_upload(path_to_data, uploaded_file.read().decode("utf-8-sig"))


st.set_page_config(
    page_title="New Experiment",
    page_icon=sidebar_logo(),
)

st.header("New Experiment")
st.write(
    """
    Here you can start a new experiment. Once you create one, you will be able
    to select it on the Machine Learning & Feature Importance pages.
    """
)
st.write(
    """
    ### Create a new experiment ⚗️

    Give your experiment a name, upload your data, and click **Create**.
    If an experiment with the same name already exists, or you don't provide a file,
    you will not be able to create it.
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
    st.session_state[ConfigStateKeys.ExperimentName] = (
        save_dir.name
    )  # get the experiment name from the file path

st.write(
    """
    Upload your data file as a CSV and then define how the data will be normalised and
    split between training and test data.
    """
)
st.file_uploader("Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName)
st.text_input(
    "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
)

st.subheader("Configure data options")
execution_options_box()

# Set up plotting options for the experiment
st.subheader("Configure experiment plots")
plot_options_box()

st.button(
    "Create",
    type="primary",
    disabled=not is_valid or not _file_is_uploaded(),
    on_click=_entrypoint,
    args=(save_dir,),
)
