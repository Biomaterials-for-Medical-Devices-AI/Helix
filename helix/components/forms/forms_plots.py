from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
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

    plot_settings = edit_plot_form(
        plot_opts,
        plot_type=PlotTypes.TargetVariableDistribution,
        key_prefix=key_prefix + PlotTypes.TargetVariableDistribution.value,
    )

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


def _render_corr_block(
    *,
    corr: pd.DataFrame,
    plot_opts: PlottingOptions,
    key_prefix: str,
    mode: str,  # "basic" or "advanced"
    data_analysis_plot_dir: Path,
) -> None:
    """Shared UI for plot+data + save. Keeps keys/filenames stable."""
    plot_tab, df_tab = st.tabs(
        [
            f"{mode.capitalize()} Correlation Plot",
            f"{mode.capitalize()} Correlation Data",
        ]
    )

    with plot_tab:
        plot_settings = edit_plot_form(
            plot_opts,
            PlotTypes.CorrelationHeatmap,
            key_prefix + PlotTypes.CorrelationHeatmap.value + f"_{mode}",
        )

        show_plot = st.checkbox(
            f"Create {mode.capitalize()} Correlation Heatmap Plot",
            key=f"{key_prefix}_{DataAnalysisStateKeys.CorrelationHeatmap}_{mode}",
        )

        if show_plot:
            fig = plot_correlation_heatmap(corr, plot_settings)
            st.pyplot(fig)

            if st.button(
                f"Save {mode.capitalize()} Plot",
                key=f"{key_prefix}_{DataAnalysisStateKeys.SaveHeatmap}_{mode}",
            ):
                fig.savefig(
                    data_analysis_plot_dir
                    / f"correlation_heatmap_{key_prefix}_{mode}.png"
                )
                plt.clf()
                st.success(f"{mode.capitalize()} plot created and saved successfully.")

    with df_tab:
        st.dataframe(corr)


def _advanced_selectors(
    data: pd.DataFrame, key_prefix: str
) -> Tuple[List[str], List[str], bool]:
    """Row/column selectors for advanced mode; returns (rows, cols, enabled)."""
    variables_1, variables_2 = st.columns(2)

    with variables_1:
        st.markdown("###### Variables in the rows")

        default_rows = (
            list(data.columns)
            if st.toggle(
                "Select all variables (rows)",
                value=False,
                key=f"{key_prefix}_{DataAnalysisStateKeys.SelectAllDescriptorsCorrelation}_1",
            )
            else []
        )

        corr_rows = st.multiselect(
            "Select variables to include in the correlation rows",
            data.columns,
            default=default_rows,
            key=f"{key_prefix}_{DataAnalysisStateKeys.DescriptorCorrelation}_1",
        )

        if len(corr_rows) < 1:
            st.warning("Please select at least one row variable.")

    with variables_2:
        st.markdown("###### Variables in the columns")

        default_cols = (
            list(data.columns)
            if st.toggle(
                "Select all variables (columns)",
                value=False,
                key=f"{key_prefix}_{DataAnalysisStateKeys.SelectAllDescriptorsCorrelation}_2",
            )
            else []
        )

        corr_cols = st.multiselect(
            "Select variables to include in the correlation columns",
            data.columns,
            default=default_cols,
            key=f"{key_prefix}_{DataAnalysisStateKeys.DescriptorCorrelation}_2",
        )

        if len(corr_cols) < 1:
            st.warning("Please select at least one column variable.")

    enable_adv = (len(corr_rows) > 0) and (len(corr_cols) > 0)
    return corr_rows, corr_cols, enable_adv


@st.experimental_fragment
def correlation_heatmap_form(
    data, data_analysis_plot_dir, plot_opts: PlottingOptions, key_prefix: str = ""
):
    """
    - Always shows the basic (square) correlation first.
    - Advanced options are displayed at the end (below basic).
    - User can generate BOTH basic and advanced plots/data in the same run.
    """

    # ----------------------------
    # BASIC (non-advanced) section
    # ----------------------------
    st.markdown("###### Select the variables to show in the correlation plot")

    default_basic = (
        list(data.columns)
        if st.toggle(
            "Select all independent variables",
            value=False,
            key=f"{key_prefix}_{DataAnalysisStateKeys.SelectAllDescriptorsCorrelation}",
        )
        else []
    )

    corr_basic_vars = st.multiselect(
        "Select independent variables to include in the correlation heatmap",
        data.columns,
        default=default_basic,
        key=f"{key_prefix}_{DataAnalysisStateKeys.DescriptorCorrelation}",
    )

    if len(corr_basic_vars) < 2:
        st.warning(
            "Please select at least two variables to create the correlation heatmap."
        )

    enable_basic = len(corr_basic_vars) > 1

    calc_basic = st.checkbox(
        "Calculate Correlation Matrix (basic)",
        key=f"{key_prefix}_{DataAnalysisStateKeys.CalculateCorrelationMatrix}_basic",
        disabled=not enable_basic,
        value=enable_basic,
    )

    if calc_basic:
        corr_basic = data[corr_basic_vars].corr()
        _render_corr_block(
            corr=corr_basic,
            plot_opts=plot_opts,
            key_prefix=key_prefix,
            mode="basic",
            data_analysis_plot_dir=data_analysis_plot_dir,
        )

    st.divider()

    # ----------------------------
    # ADVANCED section (shown at end)
    # ----------------------------
    show_adv = st.checkbox(
        "Show advanced correlation heatmap options",
        key=f"{key_prefix}_{DataAnalysisStateKeys.AdvancedCorrOptions}",
    )
    if not show_adv:
        return

    st.markdown(
        """
**What advanced mode does**
- Lets you choose **different variables for rows vs columns**.
- Produces a **rectangular correlation matrix** (rows × columns), not necessarily symmetric.
- Useful when you want to correlate one feature set against another subset.
        """
    )

    corr_rows, corr_cols, enable_adv = _advanced_selectors(data, key_prefix=key_prefix)

    calc_adv = st.checkbox(
        "Calculate Correlation Matrix (advanced)",
        key=f"{key_prefix}_{DataAnalysisStateKeys.CalculateCorrelationMatrix}_advanced",
        disabled=not enable_adv,
        value=enable_adv,
    )

    if calc_adv:
        corr_union = list(set(corr_rows) | set(corr_cols))
        corr_adv_full = data[corr_union].corr()
        corr_adv = corr_adv_full.loc[corr_rows, corr_cols]

        _render_corr_block(
            corr=corr_adv,
            plot_opts=plot_opts,
            key_prefix=key_prefix,
            mode="advanced",
            data_analysis_plot_dir=data_analysis_plot_dir,
        )


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

    plot_settings = edit_plot_form(
        plot_opts, PlotTypes.PairPlot, key_prefix + PlotTypes.PairPlot.value
    )

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
        max_value=50 if 50 < len(y) else len(y) - 1,
        value=30 if 50 < len(y) else int(len(y) / 2),
        help="The perplexity parameter controls the balance between local and global aspects of the data.",
        key=f"{key_prefix}_{DataAnalysisStateKeys.Perplexity}",
    )

    plot_settings = edit_plot_form(
        plot_opts, PlotTypes.TSNEPlot, key_prefix + PlotTypes.TSNEPlot.value
    )

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
