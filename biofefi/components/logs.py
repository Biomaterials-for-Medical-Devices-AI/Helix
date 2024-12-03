from typing import Literal

import streamlit as st

from biofefi.options.enums import ConfigStateKeys


@st.experimental_fragment
def log_box(
    box_title: str,
    key: Literal[
        ConfigStateKeys.MLLogBox, ConfigStateKeys.FILogBox, ConfigStateKeys.FuzzyLogBox
    ],
):
    """Display a text area which shows that logs of the current pipeline run."""
    with st.expander(box_title, expanded=False):
        st.text_area(
            box_title,
            key=key,
            height=200,
            disabled=True,
        )
