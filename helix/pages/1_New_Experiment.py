import os
from math import ceil
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from helix.components.configuration import plot_options_box
from helix.components.images.logos import sidebar_logo
from helix.options.choices.ui import PROBLEM_TYPES
from helix.options.data import DataOptions
from helix.options.enums import ExecutionStateKeys, PlotOptionKeys, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import helix_experiments_base_dir, uploaded_file_path
from helix.options.plotting import PlottingOptions
from helix.services.data import read_uploaded_data
from helix.services.experiments import create_experiment
from helix.utils.utils import save_upload


def _directory_is_valid(directory: Path) -> bool:
    """Determine if the directory supplied by the user is valid. If it already exists,
    it is invalid.

    Args:
        directory (Path): The path to check.

    Returns:
        bool: `True` if the directory doesn't already exist, else `False`
    """
    return not directory.exists()


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = helix_experiments_base_dir()

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    col1.text(f"{root}{os.path.sep}", help="Your experiment will be saved here")
    sub_dir = col2.text_input("Name of the experiment", placeholder="e.g. MyExperiment")

    return root / sub_dir


def _file_is_uploaded() -> bool:
    """Determine if the user has uploaded a file to the form.

    Returns:
        bool: `True` if a file was uploaded, else `False`.
    """
    return st.session_state.get(ExecutionStateKeys.UploadedFileName) is not None


def infer_problem_type_from_column(target: pd.Series) -> Optional[ProblemTypes]:
    """
    Infer the problem type given the target column. If the column contains categorical
    values (e.g. strings or few unique integers), return Classification. Otherwise, Regression.

    Returns:
        Optional[ProblemTypes]: The inferred problem type, or None if error occurs.
    """

    # Heuristic: if target is string/object or has few unique integers → classification
    if target.dtype == "object" or target.dtype.name == "category":
        return ProblemTypes.Classification
    elif pd.api.types.is_integer_dtype(target):
        n_unique = target.nunique()
        if (
            n_unique < 20 or n_unique <= len(target) * 0.05
        ):  # we assume that less than 20 unique values or less than 5% of the data are unique values indicates a classification problem
            return ProblemTypes.Classification
        else:
            return ProblemTypes.Regression
    elif pd.api.types.is_float_dtype(target):
        return ProblemTypes.Regression
    else:
        return None


def _entrypoint(save_dir: Path):
    """Function to serve as the entrypoint for experiment creation, with access
    to the session state. This is so configuration captured in fragements is
    passed correctly to the services in this function.

    Args:
        save_dir (Path): The path to the experiment.
    """
    # Set up options to save
    path_to_data = uploaded_file_path(
        st.session_state[ExecutionStateKeys.UploadedFileName].name,
        helix_experiments_base_dir()
        / st.session_state[ExecutionStateKeys.ExperimentName],
    )
    exec_opts = ExecutionOptions(
        problem_type=st.session_state.get(
            ExecutionStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        random_state=st.session_state[ExecutionStateKeys.RandomSeed],
        dependent_variable=st.session_state[ExecutionStateKeys.DependentVariableName],
        experiment_name=st.session_state[ExecutionStateKeys.ExperimentName],
    )
    data_opts = DataOptions(
        data_path=str(path_to_data),  # Path objects aren't JSON serialisable
        target_column=st.session_state[ExecutionStateKeys.DependentVariable],
        feature_columns=st.session_state[ExecutionStateKeys.FeatureColumns],
        # use `.get` as IdColumn is optional and can be None
        id_column=st.session_state.get(ExecutionStateKeys.IdColumn),
    )
    plot_opts = PlottingOptions(
        plot_axis_font_size=st.session_state[PlotOptionKeys.AxisFontSize],
        plot_axis_tick_size=st.session_state[PlotOptionKeys.AxisTickSize],
        plot_colour_scheme=st.session_state[PlotOptionKeys.ColourScheme],
        plot_colour_map=st.session_state[PlotOptionKeys.ColourMap],
        angle_rotate_xaxis_labels=st.session_state[PlotOptionKeys.RotateXAxisLabels],
        angle_rotate_yaxis_labels=st.session_state[PlotOptionKeys.RotateYAxisLabels],
        save_plots=st.session_state[PlotOptionKeys.SavePlots],
        plot_title_font_size=st.session_state[PlotOptionKeys.TitleFontSize],
        plot_font_family=st.session_state[PlotOptionKeys.FontFamily],
        dpi=st.session_state[PlotOptionKeys.DPI],
        height=st.session_state[PlotOptionKeys.Height],
        width=st.session_state[PlotOptionKeys.Width],
    )

    # Create the experiment directory and save configs
    create_experiment(
        save_dir,
        plotting_options=plot_opts,
        execution_options=exec_opts,
        data_options=data_opts,
    )

    # Save the data
    uploaded_file = st.session_state[ExecutionStateKeys.UploadedFileName]
    save_upload(path_to_data, uploaded_file, data_opts)


st.set_page_config(
    page_title="New Experiment",
    page_icon=sidebar_logo(),
)

st.header("New Experiment")
st.write(
    """
    Here you can create a new experiment. Once created, you will be able
    to select it on the Data Preprocessing, Data Visualisation and Train Models pages.
    """
)
st.write(
    """
    ### Create a new experiment ⚗️

    Give your experiment a name, upload your data, and click **Create**, located at the bottom of the page.
    If an experiment with the same name already exists, or you don't provide a file,
    you will not be able to create it.
    """
)

save_dir = _save_directory_selector()
# If a user has tried to enter a destination to save an experiment, show it
# if it's valid, else show some red text showing the destination and saying
# it's invalid.
is_valid = _directory_is_valid(save_dir)
if not is_valid and st.session_state.get(ExecutionStateKeys.ExperimentName):
    st.warning(
        f"Cannot use {save_dir}; it already exists. If you have just created this experiment, please continue."
    )
else:
    st.session_state[ExecutionStateKeys.ExperimentName] = (
        save_dir.name
    )  # get the experiment name from the file path

st.write(
    """
    Upload your data as a CSV or Excel (.xslx) file.
    """
)

st.write(
    """
    **Please note that the last column of the uploaded file should be the dependent variable.**
    """
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx"],
    key=ExecutionStateKeys.UploadedFileName,
    help="Updload a CSV or Excel (.xslx) file containing your data.",
)


def select_target_column(dataframe: pd.DataFrame) -> str:
    """ """
    columns = dataframe.columns.tolist()

    target_col = st.selectbox(
        "Select the target column (dependent variable) to model",
        options=columns,
        index=len(columns) - 1,  # Default to last column
        help="Select the column you want to predict. By default, the last column is selected.",
        key=ExecutionStateKeys.DependentVariable,
    )

    return target_col


suggested_problem_type = None

if uploaded_file is not None:

    uploaded_file.seek(0)
    data = read_uploaded_data(uploaded_file=uploaded_file)
    if st.checkbox("Show uploaded data"):
        st.markdown(" #### Uploaded data")
        st.dataframe(data)

    target_col = select_target_column(data)

    suggested_problem_type = infer_problem_type_from_column(data[target_col])
    st.write(
        f"**Suggested problem type based on the selected column `{target_col}`: {suggested_problem_type}**"
    )

    st.text_input(
        "Name of the dependent variable. **This will be used for the plots. As default, the name of the selected column of the uploaded file will be used**",
        value=target_col,
        key=ExecutionStateKeys.DependentVariableName,
    )

    feature_cols = [col for col in data.columns if col != target_col]

    # TODO: allow user to select ID columns

    n_features = len(feature_cols)
    group_size = 150

    if st.toggle(
        "Select the feature columns manually (optional)",
        help="By default, all columns except the target column will be used as features. You can manually select specific ones below.",
    ):
        # Initialize checkbox states if not set
        if "feature_checkboxes" not in st.session_state:
            st.session_state.feature_checkboxes = {col: True for col in feature_cols}

        st.markdown("#### Select which features to include:")

        # Split features into chunks of group_size
        num_groups = ceil(n_features / group_size)
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, n_features)
            group = feature_cols[start_idx:end_idx]
            first_feat, last_feat = group[0], group[-1]

            with st.expander(
                f"Features {start_idx + 1}–{end_idx} ({len(group)}) | {first_feat} → {last_feat}"
            ):
                # Select-all toggle for this group
                select_all_key = f"group_select_{i}"

                select_all = st.toggle(
                    "Select all features in this group",
                    value=False,
                    key=select_all_key,
                )

                st.divider()
                cols = st.columns(5)  # Adjust layout columns per row

                for j, col_name in enumerate(group):
                    with cols[j % 5]:
                        st.session_state.feature_checkboxes[col_name] = st.checkbox(
                            col_name,
                            value=(select_all),
                            key=f"chk_{col_name}",
                        )

        # Extract selected features
        selected_features = [
            col
            for col, checked in st.session_state.feature_checkboxes.items()
            if checked
        ]

        # Save to session_state
        st.session_state[ExecutionStateKeys.FeatureColumns] = selected_features

        # Display summary
        st.write(f"✅ **Total selected features: `{len(selected_features)}`**")
        if not selected_features:
            st.error(
                "No features selected! Please select some features to create an experiment."
            )
            is_valid = False

    else:
        # Default: all columns except target
        st.session_state[ExecutionStateKeys.FeatureColumns] = [
            col for col in data.columns if col != target_col
        ]
        st.write(f"Using all {len(feature_cols)} feature columns by default.")

    if st.toggle(
        "Select ID column",
        help="If you have a column containing row IDs, check this box to select it.",
    ):
        id_cols = list(
            filter(
                lambda x: not (
                    x in st.session_state[ExecutionStateKeys.FeatureColumns]
                    or x == target_col
                ),
                data.columns,
            )
        )
        st.selectbox(
            "ID column",
            options=id_cols,
            placeholder="Select...",
            index=None,
            key=ExecutionStateKeys.IdColumn,
        )

st.selectbox(
    "Problem type",
    PROBLEM_TYPES,
    key=ExecutionStateKeys.ProblemType,
    index=(
        PROBLEM_TYPES.index(suggested_problem_type.capitalize())
        if suggested_problem_type
        else 1
    ),
)
st.info(
    """
    If your dependent variable is categorical (e.g. cat 🐱 or dog 🐶), choose **"Classification"**.

    If your dependent variable is continuous (e.g. stock prices 📈), choose **"Regression"**.
    """
)

st.number_input(
    "Random seed",
    value=1221,
    min_value=0,
    key=ExecutionStateKeys.RandomSeed,
    help="Setting this allows experiments to be repeatable despite random shuffling of data.",
)

# Set up plotting options for the experiment
st.subheader("Configure experiment plots")
plot_options_box()

if st.button(
    "Create",
    type="primary",
    disabled=not is_valid or not _file_is_uploaded(),
    on_click=_entrypoint,
    args=(save_dir,),
):
    st.success(f"Experiment created at `{save_dir}`")
