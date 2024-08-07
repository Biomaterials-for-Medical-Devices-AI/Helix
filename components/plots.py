import streamlit as st
from pathlib import Path


def ml_plots(ml_plot_dir: Path):
    """Display the Machine Learning plots in the UI.

    Args:
        ml_plot_dir (Path): The directory containing the Machine Learning plots.
    """
    plots = list(ml_plot_dir.iterdir())
    with st.expander("Machine learning plots", expanded=len(plots) > 0):
        for p in plots:
            st.image(str(p))
