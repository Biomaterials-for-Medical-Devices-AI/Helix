"""Component for editing plot appearance."""

import matplotlib.pyplot as plt
import streamlit as st

from helix.options.choices.ui import PLOT_FONT_FAMILIES
from helix.options.enums import PlotOptionKeys


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


def edit_plot_modal(plot_opts, plot_type: str):
    """Display a modal dialog for editing plot appearance.

    Args:
        plot_opts: The current plotting options
        plot_type: Type of plot being edited (e.g., 'target_distribution', 'heatmap', 'pairplot', 'tsne')
            Used to create unique form IDs for each plot type.

    Returns:
        dict: Dictionary containing the updated plot options if submitted, None otherwise
    """

    # Color scheme
    colour_scheme = st.selectbox(
        "Color Scheme",
        plt.style.available,
        index=get_safe_index(
            plot_opts.plot_colour_scheme, plt.style.available, "default"
        ),
        key=PlotOptionKeys.ColourScheme,
        help="Select the color scheme for the plot",
    )

    # Font settings
    col1, col2 = st.columns(2)
    with col1:
        title_font_size = st.number_input(
            "Title Font Size",
            min_value=8,
            max_value=24,
            value=plot_opts.plot_title_font_size,
            key=PlotOptionKeys.TitleFontSize,
            help="Set the font size for plot titles",
        )
        axis_font_size = st.number_input(
            "Axis Font Size",
            min_value=8,
            max_value=20,
            value=plot_opts.plot_axis_font_size,
            key=PlotOptionKeys.AxisFontSize,
            help="Set the font size for axis labels",
        )
        rotate_x = st.number_input(
            "X-Axis Label Rotation",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_xaxis_labels,
            key=PlotOptionKeys.RotateXAxisLabels,
            help="Set the rotation angle for x-axis labels",
        )

    with col2:
        tick_size = st.number_input(
            "Tick Label Size",
            min_value=6,
            max_value=16,
            value=plot_opts.plot_axis_tick_size,
            key=PlotOptionKeys.AxisTickSize,
            help="Set the font size for axis tick labels",
        )
        font_family = st.selectbox(
            "Font Family",
            ["sans-serif", "serif", "monospace"],
            index=get_safe_index(
                plot_opts.plot_font_family,
                PLOT_FONT_FAMILIES,
                "sans-serif",
            ),
            key=PlotOptionKeys.FontFamily,
            help="Select the font family for all text elements",
        )
        rotate_y = st.number_input(
            "Y-Axis Label Rotation",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_yaxis_labels,
            key=PlotOptionKeys.RotateYAxisLabels,
            help="Set the rotation angle for y-axis labels",
        )

    # Plot dimensions
    st.subheader("Plot Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        # Default sizes based on plot type
        default_width = {
            "target_distribution": 10,
            "heatmap": 12,
            "pairplot": 16,
            "tsne": 16,
        }.get(plot_type, 10)

        width = st.number_input(
            "Width (inches)",
            min_value=4,
            max_value=20,
            value=default_width,
            key=PlotOptionKeys.Width,
            help="Set the width of the plot in inches",
        )

    with col2:
        default_height = {
            "target_distribution": 6,
            "heatmap": 10,
            "pairplot": 16,
            "tsne": 8,
        }.get(plot_type, 6)

        height = st.number_input(
            "Height (inches)",
            min_value=3,
            max_value=20,
            value=default_height,
            key=PlotOptionKeys.Height,
            help="Set the height of the plot in inches",
        )

    # Plot quality
    dpi = st.number_input(
        "DPI (Resolution)",
        min_value=72,
        max_value=300,
        value=plot_opts.dpi,
        key=PlotOptionKeys.DPI,
        help="Set the dots per inch (resolution) of the plot",
    )

    # Color map for heatmaps
    colour_map = st.selectbox(
        "Color Map",
        plt.colormaps(),
        index=get_safe_index(
            plot_opts.plot_colour_map,
            plt.colormaps(),
            "viridis",
        ),
        key=PlotOptionKeys.ColourMap,
        help="Select the color map for heatmaps",
    )

    # Return the current settings
    return {
        PlotOptionKeys.ColourScheme: colour_scheme,
        PlotOptionKeys.TitleFontSize: title_font_size,
        PlotOptionKeys.AxisFontSize: axis_font_size,
        PlotOptionKeys.AxisTickSize: tick_size,
        PlotOptionKeys.FontFamily: font_family,
        PlotOptionKeys.DPI: dpi,
        PlotOptionKeys.Width: width,
        PlotOptionKeys.Height: height,
        PlotOptionKeys.ColourMap: colour_map,
        PlotOptionKeys.RotateXAxisLabels: rotate_x,
        PlotOptionKeys.RotateYAxisLabels: rotate_y,
    }
