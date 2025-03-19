from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from helix.options.choices.ui import DATA_SPLITS, PLOT_FONT_FAMILIES
from helix.options.data import DataSplitOptions
from helix.options.enums import DataSplitMethods, ExecutionStateKeys, PlotOptionKeys
from helix.options.file_paths import (
    data_options_path,
    data_preprocessing_options_path,
    execution_options_path,
    fi_options_path,
    fuzzy_options_path,
    ml_options_path,
    plot_options_path,
)
from helix.options.plotting import PlottingOptions
from helix.services.configuration import (
    load_data_options,
    load_data_preprocessing_options,
    load_execution_options,
    load_fi_options,
    load_fuzzy_options,
    load_ml_options,
    load_plot_options,
)


def plot_options_box(plot_opts: PlottingOptions | None = None):
    """Expander containing the options for making plots

    Args:
        plot_opts (PlottingOptions, optional): Plot options loaded from configuration.
            If provided, use these values instead of defaults.
    """
    with st.expander("Plot options", expanded=False):
        # Use configuration values if available, otherwise use defaults
        save = st.checkbox(
            "Save all plots",
            key=PlotOptionKeys.SavePlots,
            value=plot_opts.save_plots if plot_opts else True,
        )
        width = st.number_input(
            "Width",
            min_value=10,
            max_value=100,
            key=PlotOptionKeys.Width,
            value=plot_opts.width if plot_opts else 10,
        )
        height = st.number_input(
            "Height",
            min_value=10,
            max_value=100,
            key=PlotOptionKeys.Height,
            value=plot_opts.height if plot_opts else 10,
        )
        dpi = st.slider(
            "DPI",
            min_value=50,
            max_value=300,
            value=plot_opts.dpi if plot_opts else 150,
            key=PlotOptionKeys.DPI,
        )
        rotate_x = st.number_input(
            "Angle to rotate X-axis labels",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_xaxis_labels if plot_opts else 45,
            key=PlotOptionKeys.RotateXAxisLabels,
            disabled=not save,
        )
        rotate_y = st.number_input(
            "Angle to rotate Y-axis labels",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_yaxis_labels if plot_opts else 0,
            key=PlotOptionKeys.RotateYAxisLabels,
            disabled=not save,
        )
        tfs = st.number_input(
            "Title font size",
            value=plot_opts.plot_title_font_size if plot_opts else 16,
            min_value=8,
            key=PlotOptionKeys.TitleFontSize,
            disabled=not save,
        )
        afs = st.number_input(
            "Axis font size",
            value=plot_opts.plot_axis_font_size if plot_opts else 12,
            min_value=8,
            key=PlotOptionKeys.AxisFontSize,
            disabled=not save,
        )
        ats = st.number_input(
            "Axis tick size",
            value=plot_opts.plot_axis_tick_size if plot_opts else 10,
            min_value=8,
            key=PlotOptionKeys.AxisTickSize,
            disabled=not save,
        )
        # Get valid style name, defaulting to seaborn-v0_8-whitegrid
        default_style = (
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else plt.style.available[0]
        )
        style_index = (
            plt.style.available.index(plot_opts.plot_colour_scheme)
            if plot_opts and plot_opts.plot_colour_scheme in plt.style.available
            else plt.style.available.index(default_style)
        )
        cs = st.selectbox(
            "Colour scheme",
            options=plt.style.available,
            key=PlotOptionKeys.ColourScheme,
            disabled=not save,
            index=style_index,
        )
        colormap_index = (
            plt.colormaps().index(plot_opts.plot_colour_map)
            if plot_opts
            else plt.colormaps().index("viridis")
        )
        cm = st.selectbox(
            "Colour map",
            options=plt.colormaps(),
            key=PlotOptionKeys.ColourMap,
            index=colormap_index,
            disabled=not save,
        )
        font_index = (
            PLOT_FONT_FAMILIES.index(plot_opts.plot_font_family)
            if plot_opts
            else PLOT_FONT_FAMILIES.index(PLOT_FONT_FAMILIES[1])  # sans-serif
        )
        font = st.selectbox(
            "Font",
            options=PLOT_FONT_FAMILIES,
            key=PlotOptionKeys.FontFamily,
            disabled=not save,
            index=font_index,
        )
        if save:
            """Here we show a preview of plots with the selected colour style
            colour map, font size and style, etc"""

            plt.rcParams["image.cmap"] = cm  # default colour map
            plt.rcParams["axes.titlesize"] = tfs
            plt.rcParams["axes.labelsize"] = afs
            plt.rcParams["font.family"] = font
            plt.rcParams["xtick.labelsize"] = ats
            plt.rcParams["ytick.labelsize"] = ats

            st.write("### Preview of the selected styles")
            plt.style.use(cs)
            # Generate some random data for demonstration
            arr = np.random.normal(1, 0.5, size=100)
            # Create a violin plot
            data = pd.DataFrame({"A": arr, "B": arr, "C": arr})
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            sns.violinplot(data=data, ax=ax)
            ax.set_title("Title")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.tick_params(labelsize=ats)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=rotate_y)
            st.pyplot(fig, clear_figure=True)
            fig.clear()
            # Create a figure and axis (object-oriented approach)
            fig_cmap = plt.figure(dpi=dpi)
            ax_cmap = fig_cmap.add_subplot(111)

            # Create a scatter plot to show how the colour map is applied
            scatter_plot = ax_cmap.scatter(arr, arr / 2, c=arr)
            fig_cmap.colorbar(scatter_plot, ax=ax_cmap, label="Mapped Values")
            ax_cmap.set_title("Colour Map Preview")
            # Display the figure
            st.pyplot(fig_cmap, clear_figure=True)
            fig.clear()


@st.experimental_fragment
def data_split_options_box(manual: bool = False) -> DataSplitOptions:
    """Component for configuring data split options.

    TODO: in a future PR remove the `manual` param when we can
    perform holdout and kfold with grid search.

    Args:
        manual (bool): Using manual hyperparameter setting?

    Returns:
        DataSplitOptions: The options used to split the data.
    """

    st.subheader("Configure data split method")
    if manual:
        data_split = st.selectbox("Data split method", DATA_SPLITS)
    else:
        data_split = DataSplitMethods.NoSplit
    n_bootsraps = None
    k = None
    if data_split.lower() == DataSplitMethods.Holdout:
        n_bootsraps = st.number_input(
            "Number of bootstraps",
            min_value=1,
            value=10,
            key=ExecutionStateKeys.NumberOfBootstraps,
        )
    else:
        k = st.number_input(
            "k",
            min_value=1,
            value=5,
            help="k is the number of folds in Cross-Validation",
        )
    test_size = st.number_input(
        "Test size",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        help="The fraction (between 0 and 1) to reserve for testing models on unseen data.",
    )

    return DataSplitOptions(
        method=data_split, n_bootstraps=n_bootsraps, k_folds=k, test_size=test_size
    )


def display_options(experiment_path: Path) -> None:
    """Display the options in the sidebar."""

    # Load all options
    options_dict = {
        "Execution Options": load_execution_options(
            execution_options_path(experiment_path)
        ),
        "Data Options": load_data_options(data_options_path(experiment_path)),
        "Plotting Options": load_plot_options(plot_options_path(experiment_path)),
        "Preprocessing Options": load_data_preprocessing_options(
            data_preprocessing_options_path(experiment_path)
        ),
        "Machine Learning Options": load_ml_options(ml_options_path(experiment_path)),
        "Feature Importance Options": load_fi_options(fi_options_path(experiment_path)),
        "Fuzzy Options": load_fuzzy_options(fuzzy_options_path(experiment_path)),
    }

    with st.expander("Show Experiment Options", expanded=False):

        # Display Execution Options (dataclass format)
        for option in options_dict:

            if options_dict[option] is None:
                continue

            st.write(f"### {option}")
            execution_options = options_dict[option]

            data = {
                "Variable": [field.name for field in fields(execution_options)],
                "Value": [
                    getattr(execution_options, field.name)
                    for field in fields(execution_options)
                ],
            }

            st.table(pd.DataFrame(data).set_index("Variable"))
