import pandas as pd
import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.forms import (
    correlation_heatmap_form,
    pairplot_form,
    target_variable_dist_form,
    tSNE_plot_form,
)
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.plots import plot_box
from biofefi.options.enums import ExecutionStateKeys
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    data_analysis_plots_dir,
    execution_options_path,
    plot_options_path,
    preprocessed_data_path,
)
from biofefi.services.configuration import load_execution_options, load_plot_options
from biofefi.services.experiments import get_experiments
from biofefi.utils.utils import create_directory

st.set_page_config(
    page_title="Data Visualisation",
    page_icon=sidebar_logo(),
)

sidebar_logo()

st.header("Data Visualisation")
st.write(
    """
    Here you can visualise your data. This is useful for understanding the distribution of your data,
    as well as the correlation between different features.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = biofefi_experiments_base_dir()

if experiment_name:
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    data_analysis_plot_dir = data_analysis_plots_dir(biofefi_base_dir / experiment_name)

    create_directory(data_analysis_plot_dir)

    exec_opt = load_execution_options(path_to_exec_opts)
    plot_opt = load_plot_options(path_to_plot_opts)

    data = pd.read_csv(exec_opt.data_path)

    path_to_raw_data = preprocessed_data_path(
        exec_opt.data_path.split("/")[-1],
        biofefi_base_dir / experiment_name,
    )

    st.write("### Data")

    st.write(data)

    st.write("#### Data Description")

    st.write(data.describe())

    st.write("### Data Visualisation")

    if path_to_raw_data.exists():
        data_tsne = pd.read_csv(path_to_raw_data)

        if st.toggle(
            "Visualise raw data",
            help="Turn this on if you'd like to analyse your raw data (before pre-processing).",
        ):
            data = pd.read_csv(path_to_raw_data)

    else:
        data_tsne = pd.read_csv(exec_opt.data_path)

    st.write("#### Target Variable Distribution")

    target_variable_dist_form(
        data,
        exec_opt.dependent_variable,
        data_analysis_plot_dir,
        plot_opt,
    )

    st.write("#### Correlation Heatmap")

    correlation_heatmap_form(data, data_analysis_plot_dir, plot_opt)

    st.write("#### Pairplot")

    pairplot_form(data, data_analysis_plot_dir, plot_opt)

    st.write("#### t-SNE Plot")

    tSNE_plot_form(
        data_tsne,
        exec_opt.random_state,
        data_analysis_plot_dir,
        plot_opt,
        exec_opt.normalization,
    )

    plot_box(data_analysis_plot_dir, "Data Visualisation Plots")
