import os
import streamlit as st

from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.experiments import experiment_selector
from biofefi.components.plots import plot_box
from biofefi.options.enums import ConfigStateKeys, ViewExperimentKeys
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
    ml_plot_dir,
)
from biofefi.services.logs import get_logs


st.set_page_config(
    page_title="View Experiment",
    page_icon=sidebar_logo(),
)

header = st.session_state.get(ViewExperimentKeys.ExperimentName)

st.header(header if header is not None else "View Experiment")
st.write(
    """
    On this page, you can select one of your experiments to view.

    Use the dropdown below to see the details of your experiment.

    If you have not run any analyses yet, your experiment will be empty.
    Go to the sidebar on the **left** and select an analysis to run.
    """
)

# Get the base directory of all experiments
base_dir = biofefi_experiments_base_dir()
choices = os.listdir(base_dir)
# Filter out hidden files and directories
choices = filter(lambda x: not x.startswith("."), choices)
# Filter out files
choices = filter(lambda x: os.path.isdir(os.path.join(base_dir, x)), choices)

experiment_name = experiment_selector(choices)
if experiment_name:
    experiment_path = base_dir / experiment_name
    ml_plots = ml_plot_dir(experiment_path)
    if ml_plots.exists():
        plot_box(ml_plots, "Machine learning plots")
    fi_plots = fi_plot_dir(experiment_path)
    if fi_plots.exists():
        plot_box(fi_plots, "Feature importance plots")
    fuzzy_plots = fuzzy_plot_dir(experiment_path)
    if fuzzy_plots.exists():
        plot_box(fuzzy_plots, "Fuzzy plots")
    try:
        st.session_state[ConfigStateKeys.MLLogBox] = get_logs(
            log_dir(experiment_path) / "ml"
        )
        st.session_state[ConfigStateKeys.FILogBox] = get_logs(
            log_dir(experiment_path) / "fi"
        )
        st.session_state[ConfigStateKeys.FuzzyLogBox] = get_logs(
            log_dir(experiment_path) / "fuzzy"
        )
        log_box(box_title="Machine Learning Logs", key=ConfigStateKeys.MLLogBox)
        log_box(box_title="Feature Importance Logs", key=ConfigStateKeys.FILogBox)
        log_box(box_title="Fuzzy FI Logs", key=ConfigStateKeys.FuzzyLogBox)
    except NotADirectoryError:
        pass
