import streamlit as st


def navbar():
    with st.sidebar:
        st.page_link("main_page.py", label="Home", icon="🏡")
        st.page_link("pages/new_experiment.py", label="New experiment", icon="⚗️")
