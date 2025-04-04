import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.components.plot_editor import edit_plot_modal
from helix.options.enums import (
    DataAnalysisStateKeys,
    Normalisations,
)
from helix.options.plotting import PlottingOptions


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
    st.write("### Target Variable Distribution")

    if not st.session_state.get(f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTargetDistPlot}", False):
        if st.button(
            "Edit Plot",
            key=f"{key_prefix}_edit_target_dist",
            help="Edit the appearance of the plot",
        ):
            st.session_state[
                f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTargetDistPlot}"
            ] = True
            st.rerun()

        fig, ax = plt.subplots(figsize=plot_opts.figsize)
        sns.histplot(data=data, x=dep_var_name, ax=ax)
        ax.set_title(
            f"Distribution of {dep_var_name}",
            fontsize=plot_opts.title_fontsize,
            pad=20,
        )
        ax.set_xlabel(dep_var_name, fontsize=plot_opts.axis_label_fontsize)
        ax.set_ylabel("Count", fontsize=plot_opts.axis_label_fontsize)
        ax.tick_params(labelsize=plot_opts.tick_label_fontsize)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot",
                key=f"{key_prefix}_save_target_dist",
                help="Save the plot to disk",
            ):
                plt.savefig(
                    data_analysis_plot_dir / "target_variable_distribution.png",
                    dpi=plot_opts.dpi,
                    bbox_inches="tight",
                )
                st.success("Plot saved successfully!")
        plt.close()

    else:
        edit_plot_modal(
            "Target Variable Distribution Plot",
            plot_opts,
            key_prefix=key_prefix,
            state_key=DataAnalysisStateKeys.SaveTargetDistPlot,
        )


@st.experimental_fragment
def correlation_heatmap_form(
    data, data_analysis_plot_dir, plot_opts: PlottingOptions, key_prefix: str = ""
):
    """
    Form to create the correlation heatmap plot.

    Uses plot-specific settings that are not saved between sessions.
    """
    st.write("### Correlation Heatmap")

    if not st.session_state.get(f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveHeatmapPlot}", False):
        if st.button(
            "Edit Plot",
            key=f"{key_prefix}_edit_heatmap",
            help="Edit the appearance of the plot",
        ):
            st.session_state[
                f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveHeatmapPlot}"
            ] = True
            st.rerun()

        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=plot_opts.figsize)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap=plot_opts.colormap,
            ax=ax,
            annot_kws={"size": plot_opts.tick_label_fontsize},
        )
        ax.set_title(
            "Correlation Heatmap",
            fontsize=plot_opts.title_fontsize,
            pad=20,
        )
        ax.tick_params(labelsize=plot_opts.tick_label_fontsize)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot",
                key=f"{key_prefix}_save_heatmap",
                help="Save the plot to disk",
            ):
                plt.savefig(
                    data_analysis_plot_dir / "correlation_heatmap.png",
                    dpi=plot_opts.dpi,
                    bbox_inches="tight",
                )
                st.success("Plot saved successfully!")
        plt.close()

    else:
        edit_plot_modal(
            "Correlation Heatmap",
            plot_opts,
            key_prefix=key_prefix,
            state_key=DataAnalysisStateKeys.SaveHeatmapPlot,
        )


@st.experimental_fragment
def pairplot_form(
    data, data_analysis_plot_dir, plot_opts: PlottingOptions, key_prefix: str = ""
):
    """
    Form to create the pairplot plot.

    Uses plot-specific settings that are not saved between sessions.
    """
    st.write("### Pairplot")

    if not st.session_state.get(f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}", False):
        if st.button(
            "Edit Plot",
            key=f"{key_prefix}_edit_pairplot",
            help="Edit the appearance of the plot",
        ):
            st.session_state[
                f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}"
            ] = True
            st.rerun()

        fig = sns.pairplot(data=data)
        plt.suptitle(
            "Pairplot",
            fontsize=plot_opts.title_fontsize,
            y=1.02,
        )
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot",
                key=f"{key_prefix}_save_pairplot",
                help="Save the plot to disk",
            ):
                plt.savefig(
                    data_analysis_plot_dir / "pairplot.png",
                    dpi=plot_opts.dpi,
                    bbox_inches="tight",
                )
                st.success("Plot saved successfully!")
        plt.close()

    else:
        edit_plot_modal(
            "Pairplot",
            plot_opts,
            key_prefix=key_prefix,
            state_key=DataAnalysisStateKeys.SavePairPlot,
        )


@st.experimental_fragment
def tSNE_plot_form(
    data,
    random_state,
    data_analysis_plot_dir,
    plot_opts: PlottingOptions,
    scaler: Normalisations = None,
    key_prefix: str = "",
):
    """Form to create and configure t-SNE plots."""
    st.write("### t-SNE Plot")
    st.write(
        """
        t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for dimensionality reduction
        that is particularly well suited for the visualization of high-dimensional datasets.
        """
    )

    # Get parameters for t-SNE
    col1, col2 = st.columns(2)
    with col1:
        perplexity = st.number_input(
            "Perplexity",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help=(
                "The perplexity is related to the number of nearest neighbors that "
                "is used in other manifold learning algorithms. Larger datasets "
                "usually require a larger perplexity. Consider selecting a value "
                "between 5 and 50."
            ),
            key=f"{key_prefix}_perplexity",
        )
    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
            help=(
                "The learning rate for t-SNE is usually in the range "
                "[10.0, 1000.0]. If the learning rate is too high, the data may "
                "look like a 'ball' with any point approximately equidistant from "
                "its neighbors. If the learning rate is too low, most points may "
                "look compressed in a dense cloud with few outliers."
            ),
            key=f"{key_prefix}_learning_rate",
        )

    # Scale the data if requested
    X = data.iloc[:, :-1].values
    if scaler == Normalisations.Standard:
        X = StandardScaler().fit_transform(X)
    elif scaler == Normalisations.MinMax:
        X = MinMaxScaler().fit_transform(X)

    # Compute t-SNE
    redraw = False
    if st.session_state.get(f"{key_prefix}_redraw_tsne", True):
        X_embedded = TSNE(
            n_components=2,
            learning_rate=learning_rate,
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(X)
        st.session_state[f"{key_prefix}_X_embedded"] = X_embedded
        st.session_state[f"{key_prefix}_redraw_tsne"] = False
        redraw = True

    if not st.session_state.get(f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}", False):
        if st.button(
            "Edit Plot",
            key=f"{key_prefix}_edit_tsne",
            help="Edit the appearance of the plot",
        ):
            st.session_state[
                f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}"
            ] = True
            st.rerun()

        # Plot t-SNE results
        fig, ax = plt.subplots(figsize=plot_opts.figsize)
        scatter = ax.scatter(
            st.session_state[f"{key_prefix}_X_embedded"][:, 0],
            st.session_state[f"{key_prefix}_X_embedded"][:, 1],
            c=data.iloc[:, -1],
            cmap=plot_opts.colormap,
        )
        plt.colorbar(scatter)
        ax.set_title(
            "t-SNE Plot",
            fontsize=plot_opts.title_fontsize,
            pad=20,
        )
        ax.set_xlabel("First t-SNE component", fontsize=plot_opts.axis_label_fontsize)
        ax.set_ylabel("Second t-SNE component", fontsize=plot_opts.axis_label_fontsize)
        ax.tick_params(labelsize=plot_opts.tick_label_fontsize)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "Save Plot",
                key=f"{key_prefix}_save_tsne",
                help="Save the plot to disk",
            ):
                plt.savefig(
                    data_analysis_plot_dir / "tsne_plot.png",
                    dpi=plot_opts.dpi,
                    bbox_inches="tight",
                )
                st.success("Plot saved successfully!")
        with col2:
            if st.button(
                "Recompute t-SNE",
                key=f"{key_prefix}_recompute_tsne",
                help="Generate a new t-SNE plot with current parameters",
            ):
                st.session_state[f"{key_prefix}_redraw_tsne"] = True
                st.rerun()
        plt.close()

    else:
        edit_plot_modal(
            "t-SNE Plot",
            plot_opts,
            key_prefix=key_prefix,
            state_key=DataAnalysisStateKeys.SaveTSNEPlot,
        )