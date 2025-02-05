import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.plots import display_metrics_table, plot_box
from biofefi.options.enums import (
    FeatureImportanceStateKeys,
    FuzzyStateKeys,
    MachineLearningStateKeys,
    ViewExperimentKeys,
)
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    data_analysis_plots_dir,
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
    ml_metrics_path,
    ml_plot_dir,
)
from biofefi.services.experiments import get_experiments
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


choices = get_experiments()
experiment_name = experiment_selector(choices)
if experiment_name:
    base_dir = biofefi_experiments_base_dir()
    experiment_path = base_dir / experiment_name
    data_analysis = data_analysis_plots_dir(experiment_path)
    if data_analysis.exists():
        plot_box(data_analysis, "Data Analysis Plots")
    ml_metrics = ml_metrics_path(experiment_path)
    ml_plots = ml_plot_dir(experiment_path)
    if ml_plots.exists() and ml_metrics.exists():
        display_metrics_table(ml_metrics)
        plot_box(ml_plots, "Machine learning plots")
    fi_plots = fi_plot_dir(experiment_path)
    if fi_plots.exists():
        plot_box(fi_plots, "Feature importance plots")
    fuzzy_plots = fuzzy_plot_dir(experiment_path)
    if fuzzy_plots.exists():
        plot_box(fuzzy_plots, "Fuzzy plots")
    try:
        st.session_state[MachineLearningStateKeys.MLLogBox] = get_logs(
            log_dir(experiment_path) / "ml"
        )
        log_box(
            box_title="Machine Learning Logs", key=MachineLearningStateKeys.MLLogBox
        )
    except NotADirectoryError:
        pass
    try:
        st.session_state[FeatureImportanceStateKeys.FILogBox] = get_logs(
            log_dir(experiment_path) / "fi"
        )
        log_box(
            box_title="Feature Importance Logs", key=FeatureImportanceStateKeys.FILogBox
        )
    except NotADirectoryError:
        pass
    try:
        st.session_state[FuzzyStateKeys.FuzzyLogBox] = get_logs(
            log_dir(experiment_path) / "fuzzy"
        )
        log_box(box_title="Fuzzy FI Logs", key=FuzzyStateKeys.FuzzyLogBox)
    except NotADirectoryError:
        pass
