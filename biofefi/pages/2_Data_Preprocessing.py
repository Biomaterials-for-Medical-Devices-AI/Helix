import pandas as pd
import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.images.logos import sidebar_logo
from biofefi.options.choices.ui import NORMALISATIONS, TRANSFORMATIONS_Y
from biofefi.options.enums import (
    DataPreprocessingStateKeys,
    ExecutionStateKeys,
    TransformationsY,
)
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    execution_options_path,
    plot_options_path,
    raw_data_path,
)
from biofefi.options.preprocessing import PreprocessingOptions
from biofefi.services.configuration import load_execution_options, load_plot_options
from biofefi.services.experiments import get_experiments
from biofefi.services.preprocessing import run_preprocessing


def build_config() -> PreprocessingOptions:
    """
    Build the configuration object for preprocessing.
    """

    preprocessing_options = PreprocessingOptions(
        feature_selection_methods={
            DataPreprocessingStateKeys.VarianceThreshold: st.session_state[
                DataPreprocessingStateKeys.VarianceThreshold
            ],
            DataPreprocessingStateKeys.CorrelationThreshold: st.session_state[
                DataPreprocessingStateKeys.CorrelationThreshold
            ],
            DataPreprocessingStateKeys.LassoFeatureSelection: st.session_state[
                DataPreprocessingStateKeys.LassoFeatureSelection
            ],
        },
        variance_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdVariance
        ],
        correlation_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdCorrelation
        ],
        lasso_regularisation_term=st.session_state[
            DataPreprocessingStateKeys.RegularisationTerm
        ],
        independent_variable_normalisation=st.session_state[
            DataPreprocessingStateKeys.IndependentNormalisation
        ].lower(),
        dependent_variable_transformation=st.session_state[
            DataPreprocessingStateKeys.DependentNormalisation
        ].lower(),
    )
    return preprocessing_options


st.set_page_config(
    page_title="Data Preprocessing",
    page_icon=sidebar_logo(),
)

sidebar_logo()

st.header("Data Preprocessing")
st.write(
    """
    Here you can make changes to your data before running machine learning models. This includes feature selection and scalling of variables.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = biofefi_experiments_base_dir()

if experiment_name:
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)

    exec_opt = load_execution_options(path_to_exec_opts)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    path_to_raw_data = raw_data_path(
        exec_opt.data_path.split("/")[-1],
        biofefi_base_dir / experiment_name,
    )

    if path_to_raw_data.exists():
        data = pd.read_csv(path_to_raw_data)
    else:
        data = pd.read_csv(exec_opt.data_path)

    plot_opt = load_plot_options(path_to_plot_opts)

    st.write("### Original Data")

    st.write(data)

    st.write("### Data Description")

    st.write(data.describe())

    st.write("## Data Preprocessing Options")

    st.write("#### Data Normalisation")

    st.write(
        """
        If you select **"Standardization"**, your data will be normalised by subtracting the
        mean and dividing by the standard deviation for each feature. The resulting transformation has a
        mean of 0 and values are between -1 and 1.

        If you select **"Minmax"**, your data will be scaled based on the minimum and maximum
        value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"None"**, the data will not be normalised.
        """
    )

    st.write("#### Normalisation Method for Independent Variables")

    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=DataPreprocessingStateKeys.IndependentNormalisation,
        index=len(NORMALISATIONS) - 1,  # default to no normalisation
    )

    st.write("#### Transformation Method for Dependent Variable")

    transformationy = st.selectbox(
        "Transformations",
        TRANSFORMATIONS_Y,
        key=DataPreprocessingStateKeys.DependentNormalisation,
        index=len(TRANSFORMATIONS_Y) - 1,  # default to no transformation
    )

    if (
        transformationy.lower() == TransformationsY.Log
        or transformationy.lower() == TransformationsY.Sqrt
    ):
        if (
            data.iloc[:, -1].min() <= 0
        ):  # deal with user attempting this transformations on negative values
            st.warning(
                "The dependent variable contains negative values. Log and square root transformations require positive values."
            )
            if st.checkbox(
                "Proceed with transformation. This option will add a constant to the dependent variable to make it positive.",
                key=DataPreprocessingStateKeys.ProceedTransformation,
            ):
                pass
            else:
                st.stop()

    st.write("#### Feature Selection")

    st.write("#### Check the Feature Selection Algorithms to Use")

    variance_disabled = True
    if st.checkbox(
        "Variance threshold",
        key=DataPreprocessingStateKeys.VarianceThreshold,
        help="Delete features with variance below a certain threshold",
    ):
        variance_disabled = False
    threshold = st.number_input(
        "Set threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        key=DataPreprocessingStateKeys.ThresholdVariance,
        disabled=variance_disabled,
    )

    correlation_disabled = True
    if st.checkbox(
        "Correlation threshold",
        key=DataPreprocessingStateKeys.CorrelationThreshold,
        help="Delete features with correlation above a certain threshold",
    ):
        correlation_disabled = False
    threshold = st.number_input(
        "Set threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        key=DataPreprocessingStateKeys.ThresholdCorrelation,
        disabled=correlation_disabled,
    )

    lasso_disabled = True
    if st.checkbox(
        "Lasso Feature Selection",
        key=DataPreprocessingStateKeys.LassoFeatureSelection,
        help="Select features using Lasso regression",
    ):
        lasso_disabled = False
    regularisation_term = st.number_input(
        "Set regularisation term",
        min_value=0.0,
        value=0.05,
        key=DataPreprocessingStateKeys.RegularisationTerm,
        disabled=lasso_disabled,
    )

    if st.button("Run Data Preprocessing", type="primary"):

        data.to_csv(path_to_raw_data, index=False)

        config = build_config()

        processed_data = run_preprocessing(
            data,
            biofefi_base_dir / experiment_name,
            config,
        )

        processed_data.to_csv(exec_opt.data_path, index=False)

        st.success("Data Preprocessing Complete")

        st.write("### Processed Data")

        st.write(processed_data)

        st.write("### Processed Data Description")

        st.write(processed_data.describe())
