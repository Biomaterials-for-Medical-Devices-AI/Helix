import pandas as pd
import streamlit as st


@st.experimental_fragment
def preprocessed_view(data: pd.DataFrame):
    """Display the preprocessed data to the user with column count in titles.

    Args:
        data (pd.DataFrame): The preprocessed data to show.
    """
    st.write("### Processed data")
    st.write(
        f"#### {len(data.columns)-1} independent variables and {len(data)} instances"
    )
    st.write(data)
    st.write("### Processed data description")
    st.write(data.describe())


@st.experimental_fragment
def original_view(data: pd.DataFrame):
    """Display the original data to the user.

    Args:
        data (pd.DataFrame): The original data to show.
    """
    st.write("### Original data")
    st.write(
        f"#### {len(data.columns)-1} independent variables and {len(data)} instances"
    )
    st.write(data)
    st.write("### Original data description")
    st.write(data.describe())
