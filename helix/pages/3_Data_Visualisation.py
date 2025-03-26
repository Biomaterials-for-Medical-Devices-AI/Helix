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
from helix.components.statistical_tests import normality_test_view
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
from helix.services.preprocessing import convert_nominal_to_numeric
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

    def load_dataset(
        path_to_raw_data: Path, path_to_preproc_data: Path, logger
    ) -> tuple:
        """Load raw and preprocessed data if available.

        Args:
            path_to_raw_data: Path to raw data file
            path_to_preproc_data: Path to preprocessed data file
            logger: Logger instance

        Returns:
            tuple: (raw_data, preprocessed_data, data_for_tsne)
        """
        raw_data = None
        preprocessed_data = None
        data_tsne = None

        if path_to_raw_data.exists():
            raw_data = read_data(path_to_raw_data, logger)
            raw_data = convert_nominal_to_numeric(raw_data)
            data_tsne = raw_data

            if path_to_preproc_data.exists():
                preprocessed_data = read_data(path_to_preproc_data, logger)
                data_tsne = preprocessed_data

        return raw_data, preprocessed_data, data_tsne

    def visualisation_view(data, data_tsne, prefix: str | None = None):
        """Display visualisation of data."""

        if data is not None:
            st.write("### Graphical Description")
            st.write("#### Target Variable Distribution")
            target_variable_dist_form(
                data,
                exec_opt.dependent_variable,
                data_analysis_plot_dir,
                plot_opt,
                key_prefix=prefix,
            )

            st.write("#### Correlation Heatmap")
            correlation_heatmap_form(
                data, data_analysis_plot_dir, plot_opt, key_prefix=prefix
            )

            st.write("#### Pairplot")
            pairplot_form(data, data_analysis_plot_dir, plot_opt, key_prefix=prefix)

            st.write("#### t-SNE Plot")
            tSNE_plot_form(
                data_tsne,
                exec_opt.random_state,
                data_analysis_plot_dir,
                plot_opt,
                data_opts.normalisation,
                key_prefix=prefix,
            )

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

        # Load data based on what's available
        raw_data, preprocessed_data, data_tsne = load_dataset(
            path_to_raw_data, path_to_preproc_data, logger
        )

        st.write("### Dataset Overview")

        # Create tabs based on available data
        if preprocessed_data is not None:
            raw_tab, preprocessed_tab = st.tabs(["Raw Data", "Preprocessed Data"])
            # I need to use tabs in the interface to show both data
            with raw_tab:
                st.write(
                    f"#### Raw Data [{len(raw_data.columns)-1} independent variables]"
                )
                st.info("This is your original data **before** preprocessing.")
                st.dataframe(raw_data)

                st.write("#### Data Statistics")
                st.write(raw_data.describe())

                st.write("### Normality Tests")
                normality_test_view(raw_data, "Raw Data")
                # Data visualisation
                visualisation_view(raw_data, data_tsne, prefix="raw")

            # I need to use tabs in the interface to show both data
            with preprocessed_tab:
                st.write(
                    f"#### Preprocessed Data [{len(preprocessed_data.columns)-1} independent variables]"
                )
                st.info("This is your original data **after** preprocessing.")
                st.dataframe(preprocessed_data)

                st.write("#### Data Statistics")
                st.write(preprocessed_data.describe())

                st.write("### Normality Tests")
                normality_test_view(preprocessed_data, "Preprocessed Data")
                # Data visualisation
                visualisation_view(preprocessed_data, data_tsne, prefix="preprocessed")

        else:  # raw data only available, so no need for tabs
            if raw_data is not None:
                st.write(
                    f"#### Raw Data [{len(raw_data.columns)-1} independent variables]"
                )
                st.info("This is your original data **before** preprocessing.")
                st.dataframe(raw_data)

                if preprocessed_data is None:
                    st.write("#### Data Statistics")
                    st.write(raw_data.describe())

                st.write("### Normality Tests")
                normality_test_view(raw_data, "Raw Data")
                visualisation_view(raw_data, data_tsne, prefix="raw")
            else:
                st.info("No raw data available.")

    finally:
        close_logger(logger_instance, logger)
