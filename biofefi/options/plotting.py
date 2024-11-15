from dataclasses import dataclass


@dataclass
class PlottingOptions:
    plot_axis_font_size: int
    plot_axis_tick_size: int
    plot_colour_scheme: str
    angle_rotate_xaxis_labels: int
    angle_rotate_yaxis_labels: int
    save_plots: bool
    plot_title_font_size: int
    plot_font_family: str
