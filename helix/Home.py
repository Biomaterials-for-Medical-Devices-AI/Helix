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
    """
    **Helix** is a data science tool that allows you to rapidly visualise your tabular data develop machine learning models of many kinds, and evaluate their performance down to a **feature-by-feature** level.
    It implements post-training feature importance analysis using SHAP, LIME, and Permutation Importance, their ensembles and extends the interpretability library to support fuzzy logic interpretation rules.

    You can create models to solve either **classification** problems (e.g. is this image a cat 🐱 or a dog 🐶?)
    or **regression** problems (e.g. what will be the price of gold 🏅 tomorrow 📈?).

    Your models can then be evaluated by general measures, such as **accuracy**, and by individual feature metrics,
    such as **SHAP**.

    ### Using Helix

    To create a **new experiment** ⚗️, go to the sidebar on the **left** and click **"New Experiment"**.

    To preprocess your data, go to the sidebar on the **left** and click **"Data Preprocessing"**.

    To visualise your data as part of EDA, go to the sidebar on the **left** and click **"Data Visualisation"**.

    To train new machine learning models 🏋️, go to the sidebar on the **left** and click **"Train Models"**.

    To run a feature importance analysis 📊, go to the sidebar on the **left** and click **"Feature Importance"**.

    To view your previous experiments 📈, go to the sidebar on the **left** and click **"View Experiments"**.
    """
)
