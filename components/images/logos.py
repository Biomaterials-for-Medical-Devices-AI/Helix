import streamlit as st


def header_logo():
    """Generate the header logo for the app."""
    st.image("static/BioFEFI_Logo_Transparent_760x296.png", use_column_width=True)


def sidebar_logo():
    """Generate the sidebar logo in the top left."""
    st.logo("static/BioFEFI_Logo_Transparent_160x160.png")
