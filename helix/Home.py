import streamlit as st

from helix.components.images.logos import header_logo, sidebar_logo

st.set_page_config(
    page_title="Helix",
    page_icon=sidebar_logo(),
)
header_logo()
sidebar_logo()

st.write("# Welcome")
st.write(
    "**Helix** is a data science tool that allows you to rapidly visualise your tabular data, develop machine learning models of many kinds, and evaluate their performance down to a **feature-by-feature** level.\n\n"
    "It implements post-training feature importance analysis using SHAP, LIME, and Permutation Importance, their ensembles and extends the interpretability library to support fuzzy logic interpretation rules.\n\n"
    "You can create models to solve either **classification** problems (e.g. is this image a cat 🐱 or a dog 🐶?)\n\n"
    "or **regression** problems (e.g. what will be the price of gold 🏅 tomorrow 📈?).\n\n"
    "Your models can then be evaluated by general measures, such as **accuracy**, and by individual feature metrics, such as **SHAP**.\n\n"
    "### Using Helix\n\n"
    "To create a **new experiment** ⚗️, go to the sidebar on the **left** and click **\"New Experiment\"**.\n\n"
    "To preprocess your data, go to the sidebar on the **left** and click **\"Data Preprocessing\"**.\n\n"
    "To visualise your data as part of EDA, go to the sidebar on the **left** and click **\"Data Visualisation\"**.\n\n"
    "To train new machine learning models 🏋️, go to the sidebar on the **left** and click **\"Train Models\"**.\n\n"
    "To run a feature importance analysis 📊, go to the sidebar on the **left** and click **\"Feature Importance\"**.\n\n"
    "To view your previous experiments 📈, go to the sidebar on the **left** and click **\"View Experiments\"**.\n\n"
)

