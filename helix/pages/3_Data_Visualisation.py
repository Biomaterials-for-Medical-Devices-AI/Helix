from pathlib import Path

import streamlit as st

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector
from helix.components.forms import (
    correlation_heatmap_form,
    pairplot_form,
    target_variable_dist_form,
    tSNE_plot_form,
)
from helix.components.images.logos import sidebar_logo
from helix.components.plots import plot_box
from helix.components.statistical_tests import normaility_test_tabs
from helix.options.enums import ExecutionStateKeys
from helix.options.file_paths import (
    data_analysis_plots_dir,
    data_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    plot_options_path,
    preprocessed_data_path,
)
from helix.services.configuration import (
    load_data_options,
    load_execution_options,
    load_plot_options,
)
from helix.services.data import read_data
from helix.services.experiments import get_experiments
from helix.utils.logging_utils import Logger, close_logger
from helix.utils.utils import create_directory

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
biofefi_base_dir = helix_experiments_base_dir()

if experiment_name:
    logger_instance = Logger()
    logger = logger_instance.make_logger()

    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    # Create a container for configuration options
    config_container = st.container()
    with config_container:
        display_options(biofefi_base_dir / experiment_name)

    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    data_analysis_plot_dir = data_analysis_plots_dir(biofefi_base_dir / experiment_name)

    path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)

    create_directory(data_analysis_plot_dir)

    exec_opt = load_execution_options(path_to_exec_opts)
    plot_opt = load_plot_options(path_to_plot_opts)
    data_opts = load_data_options(path_to_data_opts)

    try:

        # `raw_data` refers to the data before it gets any preprocessing,
        # such as Standardisation or Log transformation.
        # `data` can be preprocessed data, if the user has used the preprocessing page.
        # If not, then `data` will be the raw data that they uploaded.
        # In this case `raw_data` will be None.
        path_to_raw_data = Path(data_opts.data_path.replace("_preprocessed", ""))
        path_to_preproc_data = preprocessed_data_path(
            str(path_to_raw_data), biofefi_base_dir / experiment_name
        )
        if path_to_raw_data.exists() and path_to_preproc_data.exists():
            raw_data = read_data(path_to_raw_data, logger)

            st.write(f"### Raw Data [{len(raw_data.columns)} variables]")
            st.info("This is your original data **before** preprocessing.")
            st.dataframe(raw_data)
        else:
            raw_data = None

        if path_to_raw_data.exists():
            data_tsne = read_data(path_to_raw_data, logger)

            if st.toggle(
                "Visualise raw data",
                help="Turn this on if you'd like to analyse your raw data (before pre-processing).",
            ):
                data = read_data(path_to_raw_data, logger)
            else:
                data = read_data(path_to_preproc_data, logger)
        else:
            data_tsne = read_data(Path(data_opts.data_path), logger)
            data = read_data(Path(data_opts.data_path), logger)

        st.write(f"### Data [{len(data.columns)} variables]")
        st.info("This is your dataset **after** preprocessing. If no preprocessing took place, this will be the same as the raw data.")

        st.write(data)

        st.write("#### Aggregated Statistics")

        st.write(data.describe())

        normaility_test_tabs(
            data=data,
            raw_data=raw_data,
        )

        st.write("### Graphical Description")

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
            data_opts.normalisation,
        )
        
        #plot_box(data_analysis_plot_dir, "Data Visualisation Plots")
    # except ValueError:
    #     # When the user uploaded the wrong file type, somehow
    #     st.error("You must upload a .csv or .xlsx file.", icon="🔥")
    # except Exception:
    #     # Catch all error
    #     st.error("Something went wrong.", icon="🔥")
    finally:
        close_logger(logger_instance, logger)
