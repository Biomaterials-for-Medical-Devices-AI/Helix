import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from biofefi.options.choices.ui import DATA_SPLITS, PLOT_FONT_FAMILIES, PROBLEM_TYPES
from biofefi.options.enums import DataSplitMethods, ExecutionStateKeys, PlotOptionKeys


@st.experimental_fragment
def plot_options_box():
    """Expander containing the options for making plots"""
    with st.expander("Plot options", expanded=False):
        save = st.checkbox(
            "Save all plots",
            key=PlotOptionKeys.SavePlots,
            value=True,
        )
        rotate_x = st.number_input(
            "Angle to rotate X-axis labels",
            min_value=0,
            max_value=90,
            value=10,
            key=PlotOptionKeys.RotateXAxisLabels,
            disabled=not save,
        )
        rotate_y = st.number_input(
            "Angle to rotate Y-axis labels",
            min_value=0,
            max_value=90,
            value=60,
            key=PlotOptionKeys.RotateYAxisLabels,
            disabled=not save,
        )
        tfs = st.number_input(
            "Title font size",
            value=20,
            min_value=8,
            key=PlotOptionKeys.TitleFontSize,
            disabled=not save,
        )
        afs = st.number_input(
            "Axis font size",
            min_value=8,
            key=PlotOptionKeys.AxisFontSize,
            disabled=not save,
        )
        ats = st.number_input(
            "Axis tick size",
            min_value=8,
            key=PlotOptionKeys.AxisTickSize,
            disabled=not save,
        )
        cs = st.selectbox(
            "Colour scheme",
            options=plt.style.available,
            key=PlotOptionKeys.ColourScheme,
            disabled=not save,
        )
        cm = st.selectbox(
            "Colour map",
            options=plt.colormaps(),
            key=PlotOptionKeys.ColourMap,
            index=3,
            disabled=not save,
        )
        font = st.selectbox(
            "Font",
            options=PLOT_FONT_FAMILIES,
            key=PlotOptionKeys.FontFamily,
            disabled=not save,
            index=1,
        )
        if save:
            """Here we show a preview of plots with the selected colour style
            colour map, font size and style, etc"""

            plt.rcParams["image.cmap"] = cm  # default colour map
            plt.rcParams["axes.titlesize"] = tfs
            plt.rcParams["axes.labelsize"] = afs
            plt.rcParams["font.family"] = font
            plt.rcParams["xtick.labelsize"] = ats
            plt.rcParams["ytick.labelsize"] = ats

            st.write("### Preview of the selected styles")
            plt.style.use(cs)
            # Generate some random data for demonstration
            arr = np.random.normal(1, 0.5, size=100)
            # Create a violin plot
            data = pd.DataFrame({"A": arr, "B": arr, "C": arr})
            fig, ax = plt.subplots()
            sns.violinplot(data=data, ax=ax)
            ax.set_title("Title")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.tick_params(labelsize=ats)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=rotate_y)
            st.pyplot(fig, clear_figure=True)
            fig.clear()
            # Create a figure and axis (object-oriented approach)
            fig_cmap = plt.figure()
            ax_cmap = fig_cmap.add_subplot(111)

            # Create a scatter plot to show how the colour map is applied
            scatter_plot = ax_cmap.scatter(arr, arr / 2, c=arr)
            fig_cmap.colorbar(scatter_plot, ax=ax_cmap, label="Mapped Values")
            ax_cmap.set_title("Colour Map Preview")
            # Display the figure
            st.pyplot(fig_cmap, clear_figure=True)
            fig.clear()


@st.experimental_fragment
def execution_options_box_manual():
    """
    The execution options box for when the user wants to manually set the hyper-parameters
    for their models.
    """
    st.write(
        """
        If your dependent variable is categorical (e.g. cat 🐱 or dog 🐶), choose **"Classification"**.

        If your dependent variable is continuous (e.g. stock prices 📈), choose **"Regression"**.
        """
    )
    st.selectbox(
        "Problem type",
        PROBLEM_TYPES,
        key=ExecutionStateKeys.ProblemType,
        index=1,
    )
    data_split = st.selectbox("Data split method", DATA_SPLITS)
    if data_split.lower() == DataSplitMethods.Holdout:
        split_size = st.number_input(
            "Test split",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
        )
        st.session_state[ExecutionStateKeys.DataSplit] = {
            "type": DataSplitMethods.Holdout,
            "test_size": split_size,
        }
    elif data_split.lower() == DataSplitMethods.KFold:
        split_size = st.number_input(
            "n splits",
            min_value=0,
            value=5,
        )
        st.session_state[ExecutionStateKeys.DataSplit] = {
            "type": DataSplitMethods.KFold,
            "n_splits": split_size,
        }
    st.number_input(
        "Number of bootstraps",
        min_value=1,
        value=10,
        key=ExecutionStateKeys.NumberOfBootstraps,
    )
    st.number_input(
        "Random seed", value=1221, min_value=0, key=ExecutionStateKeys.RandomSeed
    )


@st.experimental_fragment
def execution_options_box_auto():
    """
    The execution options box for when the user wants to use automatic
    hyper-parameter search.
    """
    st.write(
        """
        If your dependent variable is categorical (e.g. cat 🐱 or dog 🐶), choose **"Classification"**.

        If your dependent variable is continuous (e.g. stock prices 📈), choose **"Regression"**.
        """
    )
    st.selectbox(
        "Problem type",
        PROBLEM_TYPES,
        key=ExecutionStateKeys.ProblemType,
        index=1,
    )
    test_split = st.number_input(
        "Test split",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
    )
    split_size = st.number_input(
        "n splits",
        min_value=0,
        value=5,
    )
    # Set data split to none for grid search but specify the test size
    st.session_state[ExecutionStateKeys.DataSplit] = {
        "type": DataSplitMethods.NoSplit,
        "n_splits": split_size,
        "test_size": test_split,
    }
    st.number_input(
        "Random seed", value=1221, min_value=0, key=ExecutionStateKeys.RandomSeed
    )
