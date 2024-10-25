import streamlit as st
from pathlib import Path


@st.experimental_fragment
def plot_box(plot_dir: Path, box_title: str):
    """Display the plots in the given directory in the UI.

    Args:
        plot_dir (Path): The directory containing the plots.
        box_title (str): The title of the plot box.
    """
    plots = list(plot_dir.iterdir())
    with st.expander(box_title, expanded=len(plots) > 0):
        for p in plots:
            st.image(str(p))
