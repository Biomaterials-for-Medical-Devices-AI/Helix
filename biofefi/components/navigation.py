import streamlit as st


def navbar():
    with st.sidebar:
        st.page_link("main_page.py", label="Home", icon="🏡")
        st.page_link("pages/new_experiment.py", label="New experiment", icon="⚗️")
        st.page_link("pages/experiment_detail.py", label="View experiment", icon="📈")
        st.page_link("pages/feat_importance.py", label="Feature Importance", icon="📊")
