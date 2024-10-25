import streamlit as st
from biofefi.options.enums import ConfigStateKeys


def log_box():
    """Display a text area which shows that logs of the current pipeline run."""
    with st.expander("Pipeline report", expanded=True):
        st.text_area(
            "Logs",
            value=st.session_state.get(ConfigStateKeys.LogBox, ""),
            key=ConfigStateKeys.LogBox,
            height=200,
        )
