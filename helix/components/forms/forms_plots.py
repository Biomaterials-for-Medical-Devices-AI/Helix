import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.components.plot_editor import edit_plot_form
from helix.options.enums import DataAnalysisStateKeys, Normalisations, PlotTypes
from helix.options.plotting import PlottingOptions
from helix.services.plotting import (
    create_pairplot,
    create_tsne_plot,
    plot_correlation_heatmap,
    plot_target_variable_distribution,
)


@st.experimental_fragment
def target_variable_dist_form(
    data,
    dep_var_name,
    data_analysis_plot_dir,
    plot_opts: PlottingOptions,
    key_prefix: str = "",
):
    """
    Form to create the target variable distribution plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    show_kde = st.toggle(
        "Show kernel density estimation in the distribution plot",
        value=True,
        key=f"{key_prefix}_{DataAnalysisStateKeys.ShowKDE}",
    )
    n_bins = st.slider(
        "Number of bins",
        min_value=5,
        max_value=50,
        value=10,
        key=f"{key_prefix}_{DataAnalysisStateKeys.NBins}",
    )

    plot_settings = edit_plot_form(plot_opts, PlotTypes.TargetVariableDistribution)

    show_plot = st.checkbox(
        "Create target variable distribution plot",
        key=f"{key_prefix}_{DataAnalysisStateKeys.TargetVarDistribution}",
    )
    if show_plot:

        displot = plot_target_variable_distribution(
            data,
            show_kde,
            n_bins,
            plot_settings,
            dep_var_name,
        )

        st.pyplot(displot)
        plt.close()

        if st.button(
            "Save Plot",
            key=f"{key_prefix}_{DataAnalysisStateKeys.SaveTargetVarDistribution}",
        ):
            displot.savefig(
                data_analysis_plot_dir / f"{dep_var_name}_distribution_{key_prefix}.png"
            )
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def correlation_heatmap_form(
    data, data_analysis_plot_dir, plot_opts: PlottingOptions, key_prefix: str = ""
):
    """
    Form to create the correlation heatmap plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    if st.toggle(
        "Select all independent variables",
        value=False,
        key=f"{key_prefix}_{DataAnalysisStateKeys.SelectAllDescriptorsCorrelation}",
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = []

    corr_descriptors = st.multiselect(
        "Select independent variables to include in the correlation heatmap",
        data.columns[:-1],
        default=default_corr,
        key=f"{key_prefix}_{DataAnalysisStateKeys.DescriptorCorrelation}",
    )

    corr_data = data[corr_descriptors + [data.columns[-1]]]

    if len(corr_descriptors) < 1:
        st.warning(
            "Please select at least one independent variable to create the correlation heatmap."
        )

    plot_settings = edit_plot_form(plot_opts, PlotTypes.CorrelationHeatmap)

    show_plot = st.checkbox(
        "Create Correlation Heatmap Plot",
        key=f"{key_prefix}_{DataAnalysisStateKeys.CorrelationHeatmap}",
    )
    if show_plot:

        correlation_heatmap = plot_correlation_heatmap(corr_data, plot_settings)

        st.pyplot(correlation_heatmap)

        if st.button(
            "Save Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.SaveHeatmap}"
        ):
            correlation_heatmap.savefig(
                data_analysis_plot_dir / f"correlation_heatmap_{key_prefix}.png"
            )
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def pairplot_form(  # noqa: C901
    data, data_analysis_plot_dir, plot_opts: PlottingOptions, key_prefix: str = ""
):
    """
    Form to create the pairplot plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    if st.toggle(
        "Select all independent variables",
        value=False,
        key=f"{key_prefix}_{DataAnalysisStateKeys.SelectAllDescriptorsPairPlot}",
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = None

    descriptors = st.multiselect(
        "Select independent variables to include in the pairplot",
        data.columns[:-1],
        default=default_corr,
        key=f"{key_prefix}_{DataAnalysisStateKeys.DescriptorPairPlot}",
    )

    pairplot_data = data[descriptors + [data.columns[-1]]]

    if len(descriptors) < 1:
        st.warning(
            "Please select at least one independent variable to create the correlation plot."
        )

    exclude_corner = st.checkbox(
        "Exclude corner plot",
        value=True,
        key=f"{key_prefix}_{DataAnalysisStateKeys.ExcludeCorner}",
    )

    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox(
            "Select how to display the correlation between variables",
            options=["scatter", "kde", "hist", "reg"],
            index=0,
            key=f"{key_prefix}_{DataAnalysisStateKeys.PairPlotCorrelationType}",
            help="Select the type of distribution to plot in the pairplot.",
        )
    with col2:
        diag_kind = st.selectbox(
            "Select the diagonal plot type",
            options=["hist", "kde", None],
            index=0,
            key=f"{key_prefix}_{DataAnalysisStateKeys.PairPlotDiagonalType}",
            help="Select the type of plot to use on the diagonal of the pairplot.",
        )

    plot_settings = edit_plot_form(plot_opts, PlotTypes.PairPlot)

    show_plot = st.checkbox(
        "Create Pairplot", key=f"{key_prefix}_{DataAnalysisStateKeys.PairPlot}"
    )
    if show_plot:

        pairplot = create_pairplot(
            pairplot_data, plot_settings, exclude_corner, kind, diag_kind
        )

        st.pyplot(pairplot)
        plt.close()

        if st.button(
            "Save Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.SavePairPlot}"
        ):
            pairplot.savefig(data_analysis_plot_dir / f"pairplot_{key_prefix}.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def tSNE_plot_form(  # noqa: C901
    data,
    random_state,
    data_analysis_plot_dir,
    plot_opts: PlottingOptions,
    scaler: Normalisations = None,
    key_prefix: str = "",
):

    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]

    if scaler == Normalisations.NoNormalisation:
        scaler = st.selectbox(
            "Select normalisation for comparison (this will not affect the normalisation for ML models)",
            options=[Normalisations.Standardisation, Normalisations.MinMax],
            key=f"{key_prefix}_{DataAnalysisStateKeys.SelectNormTsne}",
        )

    if scaler == Normalisations.MinMax:
        X_scaled = MinMaxScaler().fit_transform(X)
    elif scaler == Normalisations.Standardisation:
        X_scaled = StandardScaler().fit_transform(X)

    perplexity = st.slider(
        "Perplexity",
        min_value=5,
        max_value=50,
        value=30,
        help="The perplexity parameter controls the balance between local and global aspects of the data.",
        key=f"{key_prefix}_{DataAnalysisStateKeys.Perplexity}",
    )

    plot_settings = edit_plot_form(plot_opts, PlotTypes.TSNEPlot)

    show_plot = st.checkbox(
        "Create t-SNE Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.TSNEPlot}"
    )
    if show_plot:

        tsne_plot = create_tsne_plot(
            X,
            X_scaled,
            y,
            plot_settings,
            random_state,
            perplexity,
        )

        st.pyplot(tsne_plot)
        plt.close()

        if st.button(
            "Save Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.SaveTSNEPlot}"
        ):
            tsne_plot.savefig(data_analysis_plot_dir / f"tsne_plot_{key_prefix}.png")
            plt.clf()
            st.success("Plots created and saved successfully.")
