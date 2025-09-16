import pandas as pd
import streamlit as st

from helix.options.choices.ui import NORMALISATIONS, TRANSFORMATIONS_Y
from helix.options.enums import (
    DataPreprocessingStateKeys,
    ProblemTypes,
    TransformationsY,
)


@st.experimental_fragment
def preprocessing_opts_form(data: pd.DataFrame, problem_type: ProblemTypes):
    st.write("## Data preprocessing options")

    st.write("### Data normalisation")

    st.write(
        """
        If you select **"Standardisation"**, your data will be normalised by subtracting the
        mean and dividing by the standard deviation for each feature. The resulting transformation has a
        mean of 0 and values are between -1 and 1.

        If you select **"Minmax"**, your data will be scaled based on the minimum and maximum
        value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"Mean centering"**, your data will be transformed by subtracting the mean
        of each feature, so that the resulting distribution has a mean of 0 but is not scaled by variance.

        If you select **"Mean centering and Poisson scaling"**, your data will first be mean-centered,
        then scaled by dividing each feature by the square root of its mean. This is useful when the
        variance of your data is approximately proportional to the mean, as in Poisson-like distributions.

        If you select **"Pareto scaling"**, your data will be scaled by dividing each feature by
        the square root of its standard deviation, without mean-centering. This reduces the relative
        importance of high-variance features while keeping some of their weight.

        If you select **"None"**, the data will not be normalised.
        """
    )

    st.write("#### Normalisation method for independent variables")

    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=DataPreprocessingStateKeys.IndependentNormalisation,
        index=len(NORMALISATIONS) - 1,  # default to no normalisation
    )

    st.write("#### Transformation method for dependent variable")

    st.write(
        """
        If you select **Log**, the dependent variable will transformed with the natural logarithm (`ln` / `log e`). If the minimum of your dependent variable is less than or equal to 0, it is first transformed by `(y - min(y)) + 1` to make all values positive, followed by the natural logarithm.

        If you select **Square-root**, the dependent variable is transformed by taking the sqare root of each value. If the minimum of your dependent variable is less than 0, it is first transformed by `y - min(y)` to make all values at least 0, followed by the square root.

        If you select **"Minmax"**, your dependent variable will be scaled based on the minimum and maximum value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"Standardisation"**, your dependent variable will be normalised by subtracting the mean and dividing by the standard deviation for each feature. The resulting transformation has a mean of 0 and values are between -1 and 1.

        If you select **"None"**, the data will not be normalised.
        """
    )

    # disable dependent variable transformation for classifications
    no_transformation_y = problem_type.lower() == ProblemTypes.Classification
    if no_transformation_y:
        st.warning(
            "Transformation of dependent variables is disabled for **classification** experiments.",
            icon="⚠️",
        )
    transformation_y = st.selectbox(
        "Transformations",
        TRANSFORMATIONS_Y,
        key=DataPreprocessingStateKeys.DependentNormalisation,
        index=len(TRANSFORMATIONS_Y) - 1,  # default to no transformation,
        disabled=no_transformation_y,
    )

    if (
        transformation_y.lower() == TransformationsY.Log
        or transformation_y.lower() == TransformationsY.Sqrt
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

    st.write("### Feature selection")

    st.write("#### Check the feature selection algorithms to use")

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
