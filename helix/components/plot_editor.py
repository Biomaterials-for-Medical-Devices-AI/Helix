"""Component for editing plot appearance."""

import matplotlib.pyplot as plt
import streamlit as st


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

    # Create unique keys for this plot type
    def make_key(base_key):
        return f"{plot_type}_{base_key}"

    # Color scheme
    colour_scheme = st.selectbox(
        "Color Scheme",
        plt.style.available,
        index=get_safe_index(
            plot_opts.plot_colour_scheme, plt.style.available, "default"
        ),
        key=make_key("colour_scheme"),
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
            key=make_key("title_font_size"),
            help="Set the font size for plot titles",
        )
        axis_font_size = st.number_input(
            "Axis Font Size",
            min_value=8,
            max_value=20,
            value=plot_opts.plot_axis_font_size,
            key=make_key("axis_font_size"),
            help="Set the font size for axis labels",
        )
        rotate_x = st.number_input(
            "X-Axis Label Rotation",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_xaxis_labels,
            key=make_key("rotate_x"),
            help="Set the rotation angle for x-axis labels",
        )

    with col2:
        tick_size = st.number_input(
            "Tick Label Size",
            min_value=6,
            max_value=16,
            value=plot_opts.plot_axis_tick_size,
            key=make_key("tick_size"),
            help="Set the font size for axis tick labels",
        )
        font_family = st.selectbox(
            "Font Family",
            ["sans-serif", "serif", "monospace"],
            index=get_safe_index(
                plot_opts.plot_font_family,
                ["sans-serif", "serif", "monospace"],
                "sans-serif",
            ),
            key=make_key("font_family"),
            help="Select the font family for all text elements",
        )
        rotate_y = st.number_input(
            "Y-Axis Label Rotation",
            min_value=0,
            max_value=90,
            value=plot_opts.angle_rotate_yaxis_labels,
            key=make_key("rotate_y"),
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
            key=make_key("width"),
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
            key=make_key("height"),
            help="Set the height of the plot in inches",
        )

    # Plot quality
    dpi = st.number_input(
        "DPI (Resolution)",
        min_value=72,
        max_value=300,
        value=plot_opts.dpi,
        key=make_key("dpi"),
        help="Set the dots per inch (resolution) of the plot",
    )

    # Color map for heatmaps
    colour_map = st.selectbox(
        "Color Map",
        ["viridis", "magma", "plasma", "inferno", "coolwarm", "RdBu", "seismic"],
        index=get_safe_index(
            plot_opts.plot_colour_map,
            ["viridis", "magma", "plasma", "inferno", "coolwarm", "RdBu", "seismic"],
            "viridis",
        ),
        key=make_key("colour_map"),
        help="Select the color map for heatmaps",
    )

    # Return the current settings
    return {
        "colour_scheme": colour_scheme,
        "title_font_size": title_font_size,
        "axis_font_size": axis_font_size,
        "axis_tick_size": tick_size,
        "font_family": font_family,
        "dpi": dpi,
        "width": width,
        "height": height,
        "colour_map": colour_map,
        "angle_rotate_xaxis_labels": rotate_x,
        "angle_rotate_yaxis_labels": rotate_y,
    }

    # Return None if the form wasn't submitted
    return None
