"""Component for displaying statistical test results."""

import pandas as pd
import streamlit as st

from helix.services.statistical_tests import create_normality_test_table


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
    preprocessed_data: pd.DataFrame,
    raw_data: pd.DataFrame | None = None,
):
    st.write("### Data Normality Tests")

    # Create tabs for raw and preprocessed data tests
    raw_tab, norm_tab = st.tabs(["Raw Data", "Preprocessed Data"])

    with raw_tab:
        # Get normality test results for raw data
        if raw_data is not None:
            raw_results = create_normality_test_table(raw_data)
            display_normality_test_results(raw_results, "Raw Data Normality Tests")
        else:
            st.info("No raw data available.")

    with norm_tab:
        # Get normality test results for preprocessed data
        if preprocessed_data is not None:
            norm_results = create_normality_test_table(preprocessed_data)
            display_normality_test_results(
                norm_results, "Preprocessed Data Normality Tests"
            )
        else:
            st.info("No preprocessing has been applied to the data yet.")
