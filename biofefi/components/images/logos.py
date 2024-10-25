import streamlit as st
import os

PATH_TO_SMALL_ICON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "static/BioFEFI_Logo_Transparent_160x160.png",
)

PATH_TO_BIG_ICON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "static/BioFEFI_Logo_Transparent_760x296.png",
)


def header_logo():
    """Generate the header logo for the app."""
    st.image(PATH_TO_BIG_ICON, use_column_width=True)


def sidebar_logo():
    """Generate the sidebar logo in the top left."""
    st.logo(PATH_TO_SMALL_ICON)
