"""Component for displaying statistical test results."""

import pandas as pd
import streamlit as st

from helix.options.data import DataOptions
from helix.options.file_paths import helix_experiments_base_dir, preprocessed_data_path
from helix.services.data import read_data
from helix.services.statistical_tests import create_normality_test_table
from helix.utils.logging_utils import Logger


def display_normality_test_results(results_df: pd.DataFrame, title: str):
    """Display normality test results in a formatted table.

    Args:
        results_df (pd.DataFrame): DataFrame containing normality test results
        title (str): Title to display above the results table
    """
    if results_df is not None:
        st.write(f"#### {title}")
        st.write(
            """
        These tests evaluate whether the data follows a normal distribution:
        - If p-value < 0.05: Data is likely not normally distributed
        - If p-value ≥ 0.05: Data might be normally distributed
        """
        )
        st.dataframe(
            results_df.style.format(
                {
                    "Shapiro-Wilk Statistic": "{:.3f}",
                    "Shapiro-Wilk p-value": "{:.3f}",
                    "Kolmogorov-Smirnov Statistic": "{:.3f}",
                    "Kolmogorov-Smirnov p-value": "{:.3f}",
                }
            )
        )


@st.experimental_fragment
def normaility_test_tabs(
    data: pd.DataFrame, data_opts: DataOptions, experiment_name: str, logger: Logger
):
    st.write("### Data Normality Tests")

    path_to_raw_data = preprocessed_data_path(
        data_opts.data_path.split("/")[-1],
        helix_experiments_base_dir() / experiment_name,
    )

    # Create tabs for raw and normalised data tests
    raw_tab, norm_tab = st.tabs(["Raw Data", "Normalised Data"])

    with raw_tab:
        # Get normality test results for raw data
        raw_data = (
            read_data(path_to_raw_data, logger) if path_to_raw_data.exists() else data
        )
        raw_results = create_normality_test_table(raw_data)
        display_normality_test_results(raw_results, "Raw Data Normality Tests")

    with norm_tab:
        # Get normality test results for normalized data
        norm_results = create_normality_test_table(data)
        display_normality_test_results(norm_results, "Normalised Data Normality Tests")
