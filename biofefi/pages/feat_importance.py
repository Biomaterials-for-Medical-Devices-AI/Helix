import streamlit as st
from biofefi.options.enums import ConfigStateKeys
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.navigation import navbar

st.set_page_config(
    page_title="Feature Importance",
    page_icon=sidebar_logo(),
)
navbar()

st.header("Feature Importance")
st.write(
    """
    This page provides options for exploring and customising feature importance and interpretability methods in the trained machine learning models. 
    You can configure global and local feature importance techniques, select ensemble approaches, and apply fuzzy feature selection. Options include tuning scoring functions, 
    setting data percentages for SHAP analysis, and configuring rules for fuzzy synergy analysis to gain deeper insights into model behavior.
    """
)

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

st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = global_methods

st.write("### Feature Importance Ensemble Methods")
st.write(
    "Ensemble methods combine results from multiple feature importance techniques, "
    "enhancing robustness. Choose how to aggregate feature importance insights."
)

ensemble_methods = {}
use_mean = st.checkbox(
    "Mean", help="Calculate the mean importance score across methods."
)
ensemble_methods["Mean"] = use_mean
use_majority = st.checkbox(
    "Majority Vote", help="Use majority voting to identify important features."
)
ensemble_methods["Majority Vote"] = use_majority

st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

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
    "Local SHAP", help="Use SHAP for local feature importance at the instance level."
)
local_importance_methods["SHAP"] = {
    "type": "local",
    "value": use_local_shap,
}

st.session_state[ConfigStateKeys.LocalImportanceFeatures] = local_importance_methods

st.write("### Additional Configuration Options")

# Number of important features
st.number_input(
    "Number of most important features to plot",
    min_value=1,
    value=10,
    help="Select how many top features to visualise based on their importance score.",
    key=ConfigStateKeys.NumberOfImportantFeatures,
)

# Scoring function for permutation importance
st.selectbox(
    "Scoring function for permutation importance",
    [
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "accuracy",
        "f1",
    ],
    help="Choose a scoring function to evaluate the model during permutation importance.",
    key=ConfigStateKeys.ScoringFunction,
)

# Number of repetitions for permutation importance
st.number_input(
    "Number of repetitions for permutation importance",
    min_value=1,
    value=5,
    help="Specify the number of times to shuffle each feature for importance estimation.",
    key=ConfigStateKeys.NumberOfRepetitions,
)

# Percentage of data to consider for SHAP
st.slider(
    "Percentage of data to consider for SHAP",
    0,
    100,
    100,
    help="Set the percentage of data used to calculate SHAP values.",
    key=ConfigStateKeys.ShapDataPercentage,
)

# Save options
st.checkbox(
    "Save feature importance options",
    help="Save the selected configuration of feature importance methods.",
    key=ConfigStateKeys.SaveFeatureImportanceOptions,
)

st.checkbox(
    "Save feature importance results",
    help="Store the results from feature importance computations.",
    key=ConfigStateKeys.SaveFeatureImportanceResults,
)

# Fuzzy Options
st.write("### Fuzzy Feature Selection Options")
st.write(
    "Activate fuzzy methods to capture interactions between features in a fuzzy rule-based system. "
    "Define the number of features, clusters, and granular options for enhanced interpretability."
)

fuzzy_feature_selection = st.checkbox(
    "Enable Fuzzy Feature Selection",
    help="Toggle fuzzy feature selection to analyze feature interactions.",
    key=ConfigStateKeys.FuzzyFeatureSelection,
)

if fuzzy_feature_selection:

    st.number_input(
        "Number of features for fuzzy interpretation",
        min_value=1,
        value=5,
        help="Set the number of features for fuzzy analysis.",
        key=ConfigStateKeys.NumberOfFuzzyFeatures,
    )

    st.checkbox(
        "Granular features",
        help="Divide features into granular categories for in-depth analysis.",
        key=ConfigStateKeys.GranularFeatures,
    )

    st.number_input(
        "Number of clusters for target variable",
        min_value=2,
        value=3,
        help="Set the number of clusters to categorise the target variable for fuzzy interpretation.",
        key=ConfigStateKeys.NumberOfClusters,
    )

    st.text_input(
        "Names of clusters (comma-separated)",
        help="Specify names for each cluster (e.g., Low, Medium, High).",
        key=ConfigStateKeys.ClusterNames,
    )

    st.number_input(
        "Number of top occurring rules for fuzzy synergy analysis",
        min_value=1,
        value=10,
        help="Set the number of most frequent fuzzy rules for synergy analysis.",
        key=ConfigStateKeys.NumberOfTopRules,
    )
