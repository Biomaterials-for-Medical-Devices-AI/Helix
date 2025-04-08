import pandas as pd
import streamlit as st

from helix.options.choices.ui import NORMALISATIONS, TRANSFORMATIONS_Y
from helix.options.enums import DataPreprocessingStateKeys, TransformationsY


@st.experimental_fragment
def preprocessing_opts_form(data: pd.DataFrame):
    st.write("## Data Preprocessing Options")

    st.write("### Data Normalisation")

    st.write(
        """
        If you select **"Standardisation"**, your data will be normalised by subtracting the
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

    st.write("### Feature Selection")

    st.write("#### Check the Feature Selection Algorithms to Use")

    variance_disabled = True
    if st.checkbox(
        "Variance threshold",
        key=DataPreprocessingStateKeys.VarianceThreshold,
        help="Delete features with variance below a certain threshold",
    ):
        variance_disabled = False
    st.number_input(
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
    st.number_input(
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
    st.number_input(
        "Set regularisation term",
        min_value=0.0,
        value=0.05,
        key=DataPreprocessingStateKeys.RegularisationTerm,
        disabled=lasso_disabled,
    )
