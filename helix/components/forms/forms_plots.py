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
    plot_target_variable_distribution,
    plot_correlation_heatmap,
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
    if show_plot or st.session_state.get("redraw_target_dist", False):
        if st.session_state.get(f"{key_prefix}_redraw_target_dist"):
            st.session_state[f"{key_prefix}_redraw_target_dist"] = False
            plt.close("all")  # Close any existing plots

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
    if show_plot or st.session_state.get(f"{key_prefix}_redraw_heatmap", False):
        if st.session_state.get(f"{key_prefix}_redraw_heatmap"):
            st.session_state[f"{key_prefix}_redraw_heatmap"] = False
            plt.close("all")  # Close any existing plots

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

    show_plot = st.checkbox(
        "Create Pairplot", key=f"{key_prefix}_{DataAnalysisStateKeys.PairPlot}"
    )
    if show_plot or st.session_state.get(f"{key_prefix}_redraw_pairplot", False):
        if st.session_state.get(f"{key_prefix}_redraw_pairplot"):
            st.session_state[f"{key_prefix}_redraw_pairplot"] = False
            plt.close("all")  # Close any existing plots

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            f"{key_prefix}_plot_settings_pairplot",
            plot_opts,
        )

        # Set the style and create the pairplot
        plt.style.use(plot_settings.plot_colour_scheme)

        # Create figure with proper DPI
        with plt.rc_context({"figure.dpi": plot_settings.dpi}):
            # Calculate the figure size based on number of variables
            n_vars = len(pairplot_data.columns)
            aspect_ratio = plot_settings.width / plot_settings.height
            size_per_var = min(plot_settings.width, plot_settings.height) / n_vars

            pairplot = sns.pairplot(
                pairplot_data,
                height=size_per_var,
                aspect=aspect_ratio,
                corner=True,
                plot_kws={"s": 50, "alpha": 0.6},  # marker size  # transparency
                diag_kws={"bins": 20, "alpha": 0.6},
            )

            # Add title to the pairplot
            pairplot.figure.suptitle(
                "Pairplot",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                y=1.02,  # Adjust title position to prevent overlap
            )

            # Apply axis label rotations and styling to all subplots
            for ax in pairplot.axes.flat:
                if ax is not None:  # Some axes might be None in corner=True mode
                    # Set x-axis label rotations
                    ax.set_xticklabels(
                        ax.get_xticklabels(),
                        rotation=plot_settings.angle_rotate_xaxis_labels,
                        family=plot_settings.plot_font_family,
                    )
                    # Set y-axis label rotations
                    ax.set_yticklabels(
                        ax.get_yticklabels(),
                        rotation=plot_settings.angle_rotate_yaxis_labels,
                        family=plot_settings.plot_font_family,
                    )

            # Customize the appearance after creating the plot
            for ax in pairplot.axes.flat:
                if ax is not None:
                    # Set font sizes
                    ax.tick_params(labelsize=plot_opts.plot_axis_tick_size, rotation=45)
                    if ax.get_xlabel():
                        ax.set_xlabel(
                            ax.get_xlabel(),
                            fontsize=plot_opts.plot_axis_font_size,
                            family=plot_opts.plot_font_family,
                        )
                    if ax.get_ylabel():
                        ax.set_ylabel(
                            ax.get_ylabel(),
                            fontsize=plot_opts.plot_axis_font_size,
                            family=plot_opts.plot_font_family,
                        )

            # Adjust layout to prevent label cutoff
            plt.tight_layout()
        st.pyplot(pairplot)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.SavePairPlot}"
            ):
                pairplot.savefig(data_analysis_plot_dir / f"pairplot_{key_prefix}.png")
                plt.clf()
                st.success("Plot created and saved successfully.")
        with col2:
            pass  # Placeholder for commented out edit functionality
            # if st.button(
            #     "Edit Plot", key=f"{key_prefix}_edit_{DataAnalysisStateKeys.SavePairPlot}"
            # ):
            #     st.session_state[
            #         f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}"
            #     ] = True

            # if st.session_state.get(
            #     f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}",
            #     False,
            # ):
            #     # Get plot-specific settings
            #     settings = edit_plot_modal(plot_opts, "pairplot")
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         if st.button(
            #             "Apply Changes", key=f"{key_prefix}_apply_changes_pairplot"
            #         ):
            #             # Store settings in session state
            #             st.session_state[f"{key_prefix}_plot_settings_pairplot"] = (
            #                 settings
            #             )
            #             st.session_state[
            #                 f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}"
            #             ] = False
            #             st.session_state[f"{key_prefix}_redraw_pairplot"] = True
            #             st.rerun()
            #     with col2:
            #         if st.button("Cancel", key=f"{key_prefix}_cancel_pairplot"):
            #             st.session_state[
            #                 f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SavePairPlot}"
            #             ] = False
            #             st.rerun()


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

    show_plot = st.checkbox(
        "Create t-SNE Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.TSNEPlot}"
    )
    if show_plot or st.session_state.get(f"{key_prefix}_redraw_tsne", False):
        if st.session_state.get(f"{key_prefix}_redraw_tsne"):
            st.session_state[f"{key_prefix}_redraw_tsne"] = False
            plt.close("all")  # Close any existing plots

        tsne_normalised = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )
        tsne_original = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )

        X_embedded_normalised = tsne_normalised.fit_transform(X_scaled)
        X_embedded = tsne_original.fit_transform(X)

        df_normalised = pd.DataFrame(X_embedded_normalised, columns=["x", "y"])
        df_normalised["target"] = y

        df = pd.DataFrame(X_embedded, columns=["x", "y"])
        df["target"] = y

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            f"{key_prefix}_plot_settings_tsne",
            plot_opts,
        )

        # Set style and create figure with proper DPI
        plt.style.use(plot_settings.plot_colour_scheme)
        with plt.rc_context({"figure.dpi": plot_settings.dpi}):
            fig, axes = plt.subplots(
                1, 2, figsize=(plot_settings.width, plot_settings.height)
            )

            # Plot 1: Normalised Data
            sns.scatterplot(
                data=df_normalised,
                x="x",
                y="y",
                hue="target",
                palette=plot_settings.plot_colour_map,
                s=100,  # marker size
                alpha=0.6,  # transparency
                ax=axes[0],
            )

            # Customize first plot
            axes[0].set_title(
                "t-SNE Plot (Normalised Features)",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                pad=20,  # Add padding above title
            )
            axes[0].set_xlabel(
                "t-SNE Component 1",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            axes[0].set_ylabel(
                "t-SNE Component 2",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            # Apply axis label rotations and styling for first plot
            axes[0].tick_params(
                axis="both", which="major", labelsize=plot_settings.plot_axis_tick_size
            )
            for label in axes[0].get_xticklabels():
                label.set_rotation(plot_settings.angle_rotate_xaxis_labels)
                label.set_family(plot_settings.plot_font_family)
            for label in axes[0].get_yticklabels():
                label.set_rotation(plot_settings.angle_rotate_yaxis_labels)
                label.set_family(plot_settings.plot_font_family)

            # Plot 2: Original Data
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="target",
                palette=plot_settings.plot_colour_map,
                s=100,  # marker size
                alpha=0.6,  # transparency
                ax=axes[1],
            )

            # Customize second plot
            axes[1].set_title(
                "t-SNE Plot (Original Features)",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                pad=20,  # Add padding above title
            )
            axes[1].set_xlabel(
                "t-SNE Component 1",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            axes[1].set_ylabel(
                "t-SNE Component 2",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            # Apply axis label rotations and styling for second plot
            axes[1].tick_params(
                axis="both", which="major", labelsize=plot_settings.plot_axis_tick_size
            )
            for label in axes[1].get_xticklabels():
                label.set_rotation(plot_settings.angle_rotate_xaxis_labels)
                label.set_family(plot_settings.plot_font_family)
            for label in axes[1].get_yticklabels():
                label.set_rotation(plot_settings.angle_rotate_yaxis_labels)
                label.set_family(plot_settings.plot_font_family)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot", key=f"{key_prefix}_{DataAnalysisStateKeys.SaveTSNEPlot}"
            ):
                fig.savefig(data_analysis_plot_dir / f"tsne_plot_{key_prefix}.png")
                plt.clf()
                st.success("Plots created and saved successfully.")
        with col2:
            pass  # Placeholder for commented out edit functionality
            # if st.button(
            #     "Edit Plot", key=f"{key_prefix}_edit_{DataAnalysisStateKeys.SaveTSNEPlot}"
            # ):
            #     st.session_state[
            #         f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}"
            #     ] = True

            # if st.session_state.get(
            #     f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}",
            #     False,
            # ):
            #     # Get plot-specific settings
            #     settings = edit_plot_modal(plot_opts, "tsne")
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         if st.button(
            #             "Apply Changes", key=f"{key_prefix}_apply_changes_tsne"
            #         ):
            #             # Store settings in session state
            #             st.session_state[f"{key_prefix}_plot_settings_tsne"] = settings
            #             st.session_state[
            #                 f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}"
            #             ] = False
            #             st.session_state[f"{key_prefix}_redraw_tsne"] = True
            #             st.rerun()
            #     with col2:
            #         if st.button("Cancel", key=f"{key_prefix}_cancel_tsne"):
            #             st.session_state[
            #                 f"{key_prefix}_show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}"
            #             ] = False
            #             st.rerun()
