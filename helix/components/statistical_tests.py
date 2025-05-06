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
def normality_test_view(
    data: pd.DataFrame,
    table_title: str | None = None,
):

    # Get normality test results for raw data
    if data is not None:
        results = create_normality_test_table(data)
        display_normality_test_results(results, f"Normality tests for {table_title}")
    else:
        st.info("No data available.")
