"""Feature importance form components for Helix.

This module contains form components for configuring feature importance analysis:
- Global feature importance methods (Permutation, SHAP)
- Ensemble methods (Mean, Majority Vote)
- Local feature importance methods (LIME, SHAP)
- Fuzzy feature importance options
"""

import streamlit as st

from helix.options.enums import (
    ExecutionStateKeys,
    FeatureImportanceStateKeys,
    FuzzyStateKeys,
    ProblemTypes,
)


@st.experimental_fragment
def fi_options_form():
    """Form for configuring feature importance options."""
    global_methods = {}

    st.write("### Global Feature Importance Methods")
    st.write(
        "Select global methods to assess feature importance across the entire dataset. "
        "These methods help in understanding overall feature impact."
    )

    use_permutation = st.checkbox(
        "Permutation Importance",
        help="Evaluate feature importance by permuting feature values.",
    )

    global_methods["Permutation Importance"] = {
        "type": "global",
        "value": use_permutation,
    }

    use_shap = st.checkbox(
        "SHAP",
        help="Apply SHAP (SHapley Additive exPlanations) for global interpretability.",
    )
    global_methods["SHAP"] = {"type": "global", "value": use_shap}

    st.session_state[FeatureImportanceStateKeys.GlobalFeatureImportanceMethods] = (
        global_methods
    )

    st.write("### Ensemble Feature Importance Methods")
    st.write(
        "Ensemble methods combine results from multiple feature importance techniques, "
        "enhancing robustness. Choose how to aggregate feature importance insights."
    )

    # global methods need to be set to perform ensemble methods
    ensemble_is_disabled = not (use_permutation or use_shap)
    if ensemble_is_disabled:
        st.warning(
            "You must configure at least one global feature importance method to perform ensemble methods.",
            icon="⚠",
        )
    ensemble_methods = {}
    use_mean = st.checkbox(
        "Mean",
        help="Calculate the mean importance score across methods.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Mean"] = use_mean
    use_majority = st.checkbox(
        "Majority Vote",
        help="Use majority voting to identify important features.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Majority Vote"] = use_majority

    st.session_state[FeatureImportanceStateKeys.EnsembleMethods] = ensemble_methods

    st.write("### Local Feature Importance Methods")
    st.write(
        "Select local methods to interpret individual predictions. "
        "These methods focus on explaining predictions at a finer granularity."
    )

    local_importance_methods = {}
    use_lime = st.checkbox(
        "LIME",
        help="Use LIME (Local Interpretable Model-Agnostic Explanations) for local interpretability.",
    )
    local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
    use_local_shap = st.checkbox(
        "Local SHAP",
        help="Use SHAP for local feature importance at the instance level.",
    )
    local_importance_methods["SHAP"] = {
        "type": "local",
        "value": use_local_shap,
    }

    st.session_state[FeatureImportanceStateKeys.LocalImportanceFeatures] = (
        local_importance_methods
    )

    st.write("### Additional Configuration Options")

    # Number of important features
    st.number_input(
        "Number of most important features to plot",
        min_value=1,
        value=10,
        help="Select how many top features to visualise based on their importance score.",
        key=FeatureImportanceStateKeys.NumberOfImportantFeatures,
    )

    # Scoring function for permutation importance
    if (
        st.session_state.get(ExecutionStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Regression
    ):
        scoring_options = [
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
    elif (
        st.session_state.get(ExecutionStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Classification
    ):
        scoring_options = ["accuracy", "f1"]
    else:
        scoring_options = []

    st.selectbox(
        "Scoring function for permutation importance",
        scoring_options,
        help="Choose a scoring function to evaluate the model during permutation importance.",
        key=FeatureImportanceStateKeys.ScoringFunction,
    )

    # Number of repetitions for permutation importance
    st.number_input(
        "Number of repetitions for permutation importance",
        min_value=1,
        value=5,
        help="Specify the number of times to shuffle each feature for importance estimation.",
        key=FeatureImportanceStateKeys.NumberOfRepetitions,
    )

    # Fuzzy Options
    st.write("### Fuzzy Feature Importance Options")
    st.write(
        "Activate fuzzy methods to capture interactions between features in a fuzzy rule-based system. "
        "Define the number of features, clusters, and granular options for enhanced interpretability."
    )

    # both ensemble_methods and local_importance_methods
    fuzzy_is_disabled = (not (use_lime or use_local_shap)) or (
        not (use_mean or use_majority)
    )
    if fuzzy_is_disabled:
        st.warning(
            "You must configure both ensemble and local importance methods to use fuzzy feature selection.",
            icon="⚠",
        )
    fuzzy_feature_importance = st.checkbox(
        "Enable Fuzzy Feature Importance",
        help="Toggle fuzzy feature importance to analyze feature interactions.",
        key=FuzzyStateKeys.FuzzyFeatureSelection,
        disabled=fuzzy_is_disabled,
    )

    if fuzzy_feature_importance:
        st.number_input(
            "Number of features for fuzzy interpretation",
            min_value=1,
            value=5,
            help="Set the number of features for fuzzy analysis.",
            key=FuzzyStateKeys.NumberOfFuzzyFeatures,
        )

        st.checkbox(
            "Granular features",
            help="Divide features into granular categories for in-depth analysis.",
            key=FuzzyStateKeys.GranularFeatures,
        )

        st.number_input(
            "Number of clusters for target variable",
            min_value=2,
            value=5,
            help="Set the number of clusters to categorise the target variable for fuzzy interpretation.",
            key=FuzzyStateKeys.NumberOfClusters,
        )

        st.text_input(
            "Names of clusters (comma-separated)",
            help="Specify names for each cluster (e.g., Low, Medium, High).",
            key=FuzzyStateKeys.ClusterNames,
            value=", ".join(["very low", "low", "medium", "high", "very high"]),
        )

        st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
            help="Set the number of most frequent fuzzy rules for synergy analysis.",
            key=FuzzyStateKeys.NumberOfTopRules,
        )

    st.subheader("Select outputs to save")

    # Save options
    st.toggle(
        "Save feature importance options",
        help="Save the selected configuration of feature importance methods.",
        key=FeatureImportanceStateKeys.SaveFeatureImportanceOptions,
        value=True,
    )

    st.toggle(
        "Save feature importance results",
        help="Store the results from feature importance computations.",
        key=FeatureImportanceStateKeys.SaveFeatureImportanceResults,
        value=True,
    )
