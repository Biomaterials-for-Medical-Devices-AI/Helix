import pandas as pd
import streamlit as st
import numpy as np

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
from helix.services.statistical_tests import shapiro_wilk_test, kolmogorov_smirnov_test
from helix.services.experiments import get_experiments
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
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    display_options(biofefi_base_dir / experiment_name)

    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    data_analysis_plot_dir = data_analysis_plots_dir(biofefi_base_dir / experiment_name)

    path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)

    create_directory(data_analysis_plot_dir)

    exec_opt = load_execution_options(path_to_exec_opts)
    plot_opt = load_plot_options(path_to_plot_opts)
    data_opts = load_data_options(path_to_data_opts)

    data = pd.read_csv(data_opts.data_path)

    path_to_raw_data = preprocessed_data_path(
        data_opts.data_path.split("/")[-1],
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
        data_tsne = pd.read_csv(data_opts.data_path)

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

    plot_box(data_analysis_plot_dir, "Data Visualisation Plots")

    def create_normality_test_table(data: pd.DataFrame) -> pd.DataFrame:
        """Create a dataframe with normality test results for numerical columns."""
        test_results = []
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Skip if all values are the same (no variance)
            if len(data[col].unique()) <= 1:
                continue
                
            # Perform Shapiro-Wilk test
            sw_stat, sw_p = shapiro_wilk_test(data[col].dropna())
            
            # Perform Kolmogorov-Smirnov test
            ks_stat, ks_p = kolmogorov_smirnov_test(data[col].dropna())
            
            test_results.append({
                'Variable': col,
                'Shapiro-Wilk Statistic': round(sw_stat, 3),
                'Shapiro-Wilk p-value': round(sw_p, 3),
                'Kolmogorov-Smirnov Statistic': round(ks_stat, 3),
                'Kolmogorov-Smirnov p-value': round(ks_p, 3)
            })
        
        return pd.DataFrame(test_results) if test_results else None

    def display_normality_test_results(results_df: pd.DataFrame, title: str):
        """Display normality test results in a formatted table."""
        if results_df is not None:
            st.write(f"#### {title}")
            st.write("""
            These tests evaluate whether the data follows a normal distribution:
            - If p-value < 0.05: Data is likely not normally distributed
            - If p-value ≥ 0.05: Data might be normally distributed
            """)
            st.dataframe(
                results_df.style.format({
                    'Shapiro-Wilk Statistic': '{:.3f}',
                    'Shapiro-Wilk p-value': '{:.3f}',
                    'Kolmogorov-Smirnov Statistic': '{:.3f}',
                    'Kolmogorov-Smirnov p-value': '{:.3f}'
                })
            )

    st.write("### Data Normality Tests")
    
    # Create tabs for raw and normalized data tests
    raw_tab, norm_tab = st.tabs(["Raw Data", "Normalized Data"])
    
    with raw_tab:
        # Get normality test results for raw data
        raw_data = pd.read_csv(path_to_raw_data) if path_to_raw_data.exists() else data
        raw_results = create_normality_test_table(raw_data)
        display_normality_test_results(raw_results, "Raw Data Normality Tests")
    
    with norm_tab:
        # Get normality test results for normalized data
        norm_results = create_normality_test_table(data)
        display_normality_test_results(norm_results, "Normalised Data Normality Tests")
