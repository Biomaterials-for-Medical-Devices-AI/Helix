"""Component for editing plot appearance."""

import matplotlib.pyplot as plt
import streamlit as st

from helix.options.choices.ui import PLOT_FONT_FAMILIES
from helix.options.enums import PlotOptionKeys, PlotTypes
from helix.options.plotting import PlottingOptions


def get_safe_index(value: str, options: list[str], default_value: str) -> int:
    """Safely get the index of a value in a list, returning the index of a default if not found.

    Args:
        value: The value to find in the list
        options: List of options to search in
        default_value: Default value to use if value is not found

    Returns:
        Index of the value in the list, or index of default_value
    """
    try:
        return options.index(value)
    except ValueError:
        return options.index(default_value)


def edit_plot_form(plot_opts: PlottingOptions, plot_type: PlotTypes):
    """
    Form to edit the appearance of plots.
    This form allows users to change the color scheme, font sizes, axis labels,
    and other visual aspects of the plots.

    Args:
        plot_opts (PlottingOptions): Current plotting options to edit.
        plot_type (PlotTypes): Type of the plot being edited.

    Returns:
        PlottingOptions: Updated plotting options based on user input.

    """

    with st.expander("Edit plot", expanded=False):

        st.subheader("Colours and styles")

        colour_scheme = st.selectbox(
            "Color scheme",
            plt.style.available,
            index=get_safe_index(
                plot_opts.plot_colour_scheme, plt.style.available, "default"
            ),
            key=PlotOptionKeys.ColourScheme + plot_type.value,
            help="Select the color scheme for the plot",
        )

        # Color map for heatmaps
        if plot_type in [PlotTypes.CorrelationHeatmap, PlotTypes.TSNEPlot]:
            st.selectbox(
                "Color map",
                plt.colormaps(),
                index=get_safe_index(
                    plot_opts.plot_colour_map,
                    plt.colormaps(),
                    "viridis",
                ),
                key=PlotOptionKeys.ColourMap + plot_type.value,
                help="Select the color map for heatmaps",
            )

        if plot_type in [PlotTypes.TargetVariableDistribution]:
            st.color_picker(
                "Plot colour",
                value="#1f77b4",
                key=PlotOptionKeys.PlotColour + plot_type.value,
                help="Select the colour for the plot",
            )

        st.subheader("Customise plot titles")

        # Custom plot title
        plot_title = st.text_input(
            "Plot title",
            value=None,
            key=PlotOptionKeys.PlotTitle + plot_type.value,
            help="Set a custom title for the plot",
        )

        col1, col2 = st.columns(2)
        with col1:
            yaxis_label = st.text_input(
                "Y-axis label",
                value=None,
                key=PlotOptionKeys.YAxisLabel + plot_type.value,
                help="Set the label for the y-axis",
            )

        with col2:
            xaxis_label = st.text_input(
                "X-axis label",
                value=None,
                key=PlotOptionKeys.XAxisLabel + plot_type.value,
                help="Set the label for the x-axis",
            )

        st.subheader("Font settings")

        col1, col2 = st.columns(2)

        with col1:
            title_font_size = st.number_input(
                "Title font size",
                min_value=8,
                max_value=50,
                value=plot_opts.plot_title_font_size,
                key=PlotOptionKeys.TitleFontSize + plot_type.value,
                help="Set the font size for plot titles",
            )

            axis_font_size = st.number_input(
                "Axis font size",
                min_value=8,
                max_value=35,
                value=plot_opts.plot_axis_font_size,
                key=PlotOptionKeys.AxisFontSize + plot_type.value,
                help="Set the font size for axis labels",
            )

            rotate_x = st.number_input(
                "X-axis label rotation",
                min_value=0,
                max_value=90,
                value=plot_opts.angle_rotate_xaxis_labels,
                key=PlotOptionKeys.RotateXAxisLabels + plot_type.value,
                help="Set the rotation angle for x-axis labels",
            )

        with col2:

            tick_size = st.number_input(
                "Tick label size",
                min_value=6,
                max_value=35,
                value=plot_opts.plot_axis_tick_size,
                key=PlotOptionKeys.AxisTickSize + plot_type.value,
                help="Set the font size for axis tick labels",
            )

            font_family = st.selectbox(
                "Font family",
                PLOT_FONT_FAMILIES,
                index=get_safe_index(
                    plot_opts.plot_font_family,
                    PLOT_FONT_FAMILIES,
                    "sans-serif",
                ),
                key=PlotOptionKeys.FontFamily + plot_type.value,
                help="Select the font family for all text elements",
            )

            rotate_y = st.number_input(
                "Y-axis label rotation",
                min_value=0,
                max_value=90,
                value=plot_opts.angle_rotate_yaxis_labels,
                key=PlotOptionKeys.RotateYAxisLabels + plot_type.value,
                help="Set the rotation angle for y-axis labels",
            )

        # Plot dimensions
        st.subheader("Plot dimensions")
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input(
                "Width (inches)",
                min_value=4,
                max_value=20,
                value=plot_opts.width,
                key=PlotOptionKeys.Width + plot_type.value,
                help="Set the width of the plot in inches",
            )

        with col2:
            height = st.number_input(
                "Height (inches)",
                min_value=3,
                max_value=20,
                value=plot_opts.height,
                key=PlotOptionKeys.Height + plot_type.value,
                help="Set the height of the plot in inches",
            )

        # Plot quality
        dpi = st.slider(
            "Image resolution (DPI)",
            min_value=330,
            max_value=2000,
            value=plot_opts.dpi if 330 < plot_opts.dpi <= 2000 else 330,
            key=PlotOptionKeys.DPI + plot_type.value,
            help="Set the dots per inch (resolution) of the plot",
        )

    return PlottingOptions(
        plot_axis_font_size=axis_font_size,
        plot_axis_tick_size=tick_size,
        plot_colour_scheme=colour_scheme,
        dpi=dpi,
        angle_rotate_xaxis_labels=rotate_x,
        angle_rotate_yaxis_labels=rotate_y,
        save_plots=plot_opts.save_plots,
        plot_title_font_size=title_font_size,
        plot_font_family=font_family,
        height=height,
        width=width,
        plot_colour_map=st.session_state.get(
            PlotOptionKeys.ColourMap + plot_type.value, plot_opts.plot_colour_map
        ),
        plot_title=plot_title,
        yaxis_label=yaxis_label,
        xaxis_label=xaxis_label,
        plot_colour=st.session_state.get(
            PlotOptionKeys.PlotColour + plot_type.value, None
        ),
    )
