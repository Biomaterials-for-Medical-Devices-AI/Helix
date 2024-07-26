from numba.cuda import initialize
from feature_importance import feature_importance, fuzzy_interpretation
from feature_importance.feature_importance_options import FeatureImportanceOptions
from feature_importance.fuzzy_options import FuzzyOptions
from machine_learning import train
from machine_learning.call_methods import save_actual_pred_plots
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from utils.logging_utils import Logger, close_logger
from utils.utils import set_seed
import streamlit as st


import pandas as pd


st.image("ui/bioFEFI header.png")
# Sidebar
with st.sidebar:
    st.header("Options")
    st.checkbox("Feature Engineering")

    # Machine Learning Options
    with st.expander("Machine Learning Options"):
        ml_on = st.checkbox("Machine Learning")
        st.subheader("Machine Learning Options")
        data_split = st.selectbox("Data split method", ["Holdout", "K-Fold"])
        num_bootstraps = st.number_input("Number of bootstraps", min_value=1, value=10)
        save_plots = st.checkbox("Save actual or predicted plots")

        st.write("Model types to use:")
        use_linear = st.checkbox("Linear Model")
        use_rf = st.checkbox("Random Forest")
        use_xgb = st.checkbox("XGBoost")

        normalization = st.checkbox("Normalization")

    # Feature Importance Options
    with st.expander("Feature importance options"):
        fi_on = st.checkbox("Feature Importance")
        st.write("Global feature importance methods:")
        use_permutation = st.checkbox("Permutation Importance")
        use_shap = st.checkbox("SHAP")

        st.write("Feature importance ensemble methods:")
        use_mean = st.checkbox("Mean")
        use_majority = st.checkbox("Majority vote")

        st.write("Local feature importance methods:")
        use_lime = st.checkbox("LIME")
        use_local_shap = st.checkbox("Local SHAP")

        num_important_features = st.number_input(
            "Number of most important features to plot", min_value=1, value=10
        )
        scoring_function = st.selectbox(
            "Scoring function for permutation importance", ["accuracy", "f1", "roc_auc"]
        )
        num_repetitions = st.number_input(
            "Number of repetitions for permutation importance", min_value=1, value=5
        )
        shap_data_percentage = st.slider(
            "Percentage of data to consider for SHAP", 0, 100, 100
        )

        # Fuzzy Options
        st.subheader("Fuzzy Options")
        fuzzy_feature_selection = st.checkbox("Fuzzy feature selection")
        num_fuzzy_features = st.number_input(
            "Number of features for fuzzy interpretation", min_value=1, value=5
        )
        granular_features = st.checkbox("Granular features")
        num_clusters = st.number_input(
            "Number of clusters for target variable", min_value=2, value=3
        )
        cluster_names = st.text_input("Names of clusters (comma-separated)")
        num_top_rules = st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
        )
    seed = st.number_input("Random seed", value=1221, min_value=0)
# Main body
st.header("Data Upload")
st.text_input("Name of the experiment")
dependent_variable = st.text_input("Name of the dependent variable")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Columns:", df.columns.tolist())
    st.write("Target variable:", df.columns[-1])

    # Model training status
    st.header("Model Training Status")
    if use_linear:
        st.checkbox("Linear Model", value=False, disabled=True)
    if use_rf:
        st.checkbox("Random Forest", value=False, disabled=True)
    if use_xgb:
        st.checkbox("XGBoost", value=False, disabled=True)

    # Plot selection
    st.header("Plots")
    plot_options = [
        "Metric values across bootstrap samples",
        "Feature importance plots",
    ]
    selected_plots = st.multiselect("Select plots to display", plot_options)

    for plot in selected_plots:
        st.subheader(plot)
        st.write("Placeholder for", plot)

    # Feature importance description
    st.header("Feature Importance Description")
    if st.button("Generate Feature Importance Description"):
        st.write("Placeholder for feature importance description")

# Set seed for reproducibility
set_seed(seed)

# Pass the UI options into the Namespaces
## Instantiate a FuzzyOptions
fuzzy_opt = FuzzyOptions()
### initialize it to load the args
fuzzy_opt.initialize()
### use set_defaults to override the relevant options
### In Llettuce, I used a pydantic model to hold the options, then wrote a method to update a BaseOptions with the model's values. It's not necessary, just a matter of taste maybe?
### It might be better to get the options first, then only set the options if the relevant checkbox is clicked ¯\_(ツ)_/¯
fuzzy_opt.parser.set_defaults(
    fuzzy_feature_selection=fuzzy_feature_selection,
    num_fuzzy_features=num_fuzzy_features,
    granular_features=granular_features,
    num_clusters=num_clusters,
    cluster_names=cluster_names,
    num_top_rules=1,
    dependent_variable=dependent_variable,
)
### Then parse loads the options
fuzzy_opt = fuzzy_opt.parse()
## repeat for other ..Options
fi_opt = FeatureImportanceOptions()
fi_opt.initialize()
fi_opt.parser.set_defaults(
    # I'm not sure how to set the global_importance_methods, feature_importance_ensemble, and local_importance_methods. Do you have to call ast.literal_eval("Permutation Importance")?
    num_features_to_plot=num_important_features,
    permutation_importance_scoring=scoring_function,
    permutation_importance_repeat=num_repetitions,
    shap_reduce_data=shap_data_percentage,
    dependent_variable=dependent_variable,
)
fi_opt = fi_opt.parse()

ml_opt = MLOptions()
ml_opt.initialize()
ml_opt.parser.set_defaults(
    n_bootstraps=num_bootstraps,
    save_actual_pred_plots=save_actual_pred_plots,
    # not sure how to do model_types either
    normalization=normalization,
    dependent_variable=dependent_variable,
)
ml_opt = ml_opt.parse()

seed = ml_opt.random_state
ml_logger_instance = Logger(ml_opt.ml_log_dir, ml_opt.experiment_name)
ml_logger = ml_logger_instance.make_logger()
