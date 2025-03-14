import os

import streamlit as st

PATH_TO_ICON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "static/Helix_Logo_Transparent.png",
)


def header_logo():
    """Generate the header logo for the app."""
    _, col2, _ = st.columns(3)

    with col2:
        st.image(PATH_TO_ICON, use_column_width=True)


def sidebar_logo():
    """Generate the sidebar logo in the top left."""
    st.logo(PATH_TO_ICON)
