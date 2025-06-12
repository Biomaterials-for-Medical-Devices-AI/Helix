from pathlib import Path

import pandas as pd
import streamlit as st

from helix.components.configuration import (
    display_options,
    load_plot_options,
    load_execution_options,
)
from helix.components.experiments import experiment_selector
from helix.components.images.logos import sidebar_logo
from helix.components.logs import log_box
from helix.components.plot_editor import custom_plot_creator
from helix.components.plots import (
    display_metrics_table,
    display_predictions,
    plot_box,
    plot_box_v2,
)
from helix.options.enums import (
    FeatureImportanceStateKeys,
    MachineLearningStateKeys,
    ViewExperimentKeys,
)
from helix.options.file_paths import (
    data_analysis_plots_dir,
    fi_plot_dir,
    helix_experiments_base_dir,
    log_dir,
    ml_metrics_mean_std_path,
    ml_plot_dir,
    ml_predictions_path,
    plot_options_path,
    execution_options_path,
)
from helix.services.experiments import get_experiments
from helix.services.logs import get_logs

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


def display_experiment_plots(experiment_path: Path) -> None:
    """Display all available plots for the experiment.

    Args:
        experiment_path: Path to the experiment directory
    """
    # Data analysis plots
    data_analysis = data_analysis_plots_dir(experiment_path)
    if data_analysis.exists():
        plot_box(data_analysis, "Data analysis plots")

    # ML metrics and plots
    ml_metrics = ml_metrics_mean_std_path(experiment_path)
    if ml_metrics.exists():
        display_metrics_table(ml_metrics)

    ml_plots = ml_plot_dir(experiment_path)
    if ml_plots.exists():
        plot_box(ml_plots, "Machine learning plots")

    # Predictions
    predictions = ml_predictions_path(experiment_path)
    if predictions.exists():
        preds = pd.read_csv(predictions)
        display_predictions(preds)

    # Feature importance plots
    fi_plots = fi_plot_dir(experiment_path)
    if fi_plots.exists():
        mean_plots = [
            p
            for p in fi_plots.iterdir()
            if p.name.endswith("-all-folds-mean.png")  # mean global FI
            or p.name.startswith("local-")  # local plots
            or p.name.startswith("ensemble-")  # ensemble plots
        ]
        plot_box_v2(mean_plots, "Feature importance plots")

    # # Fuzzy plots
    # fuzzy_plots = fuzzy_plot_dir(experiment_path)
    # if fuzzy_plots.exists():
    #     plot_box(fuzzy_plots, "Fuzzy plots")


def display_experiment_logs(experiment_path: Path) -> None:
    """Display all available logs for the experiment.

    Args:
        experiment_path: Path to the experiment directory
    """
    log_configs = [
        ("ml", MachineLearningStateKeys.MLLogBox, "Machine learning logs"),
        ("fi", FeatureImportanceStateKeys.FILogBox, "Feature importance logs"),
        # ("fuzzy", FuzzyStateKeys.FuzzyLogBox, "Fuzzy FI Logs"),
    ]

    for log_dir_name, state_key, title in log_configs:
        try:
            st.session_state[state_key] = get_logs(
                log_dir(experiment_path) / log_dir_name
            )
            log_box(box_title=title, key=state_key)
        except NotADirectoryError:
            pass


def edit_results_plots(experiment_path: Path) -> None:

    predictions = ml_predictions_path(experiment_path)

    plot_opts_path = plot_options_path(experiment_path)

    execution_opts_path = execution_options_path(experiment_path)

    create_plot = st.checkbox(
        "Create custom results plot",
        key=ViewExperimentKeys.ShowCustomPlotCreator,
        help="Enable to create a custom plot for the results of your experiment.",
        value=False,
    )

    if (
        predictions.exists()
        and plot_opts_path.exists()
        and execution_opts_path.exists()
        and create_plot
    ):

        plot_opts = load_plot_options(plot_opts_path)
        execution_opts = load_execution_options(execution_opts_path)
        preds = pd.read_csv(predictions)

        plot = custom_plot_creator(
            predictions=preds,
            plot_opts=plot_opts,
            exec_opts=execution_opts,
        )

        if plot is not None:
            st.pyplot(plot, use_container_width=True)

            plot_name = st.text_input(
                "Enter a name for the plot (without extension):",
                value="custom_plot",
                key=ViewExperimentKeys.CustomPlotName,
            )

            if not plot_name.endswith(".png"):
                plot_name += ".png"

            if not plot_name.startswith("custom_plot"):
                plot_name = "custom_plot_" + plot_name

            plot_path = ml_plot_dir(experiment_path) / plot_name

            # with col2:

            save_plot = st.button("Save plot", key=ViewExperimentKeys.SavePlotButton)

            if plot_name and save_plot:
                plot.savefig(plot_path, bbox_inches="tight")
                st.success(f"Plot saved to {plot_path}")


if experiment_name:
    base_dir = helix_experiments_base_dir()
    experiment_path = base_dir / experiment_name
    display_options(experiment_path)
    display_experiment_plots(experiment_path)
    edit_results_plots(experiment_path)
    display_experiment_logs(experiment_path)
