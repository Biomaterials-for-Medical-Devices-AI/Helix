from typing import Literal

import streamlit as st

from helix.options.enums import (
    FeatureImportanceStateKeys,
    FuzzyStateKeys,
    MachineLearningStateKeys,
)


@st.experimental_fragment
def log_box(
    box_title: str,
    key: Literal[
        MachineLearningStateKeys.MLLogBox,
        FeatureImportanceStateKeys.FILogBox,
        FuzzyStateKeys.FuzzyLogBox,
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
