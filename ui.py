from argparse import Namespace
import argparse
from multiprocessing import Process
from numba.cuda import initialize
from feature_importance import feature_importance, fuzzy_interpretation
from feature_importance.feature_importance_options import FeatureImportanceOptions
from feature_importance.fuzzy_options import FuzzyOptions
from machine_learning import train
from machine_learning.call_methods import save_actual_pred_plots
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from options.enums import ConfigStateKeys
from utils.logging_utils import Logger, close_logger
from utils.utils import set_seed
from pathlib import Path
import streamlit as st
import os


import pandas as pd


def build_configuration(
    fuzzy_feature_selection,
    num_fuzzy_features,
    granular_features,
    num_clusters,
    cluster_names,
    dependent_variable,
    num_features_to_plot,
    permutation_importance_scoring,
    permutation_importance_repeat,
    shap_reduce_data,
    n_bootstraps,
    save_actual_pred_plots,
    normalization,
    data_path,
    experiment_name,
    problem_type,
    num_top_rules = 1
) -> tuple[argparse.Namespace]:
    """Build the configuration objects for the pipeline.

    Args:
        fuzzy_feature_selection (_type_): _description_
        num_fuzzy_features (_type_): _description_
        granular_features (_type_): _description_
        num_clusters (_type_): _description_
        cluster_names (_type_): _description_
        dependent_variable (_type_): _description_
        num_features_to_plot (_type_): _description_
        permutation_importance_scoring (_type_): _description_
        permutation_importance_repeat (_type_): _description_
        shap_reduce_data (_type_): _description_
        n_bootstraps (_type_): _description_
        save_actual_pred_plots (_type_): _description_
        normalization (_type_): _description_
        data_path (_type_): _description_
        experiment_name (_type_): _description_
        problem_type (_type_): _description_
        num_top_rules (int, optional): _description_. Defaults to 1.

    Returns:
        tuple[argparse.Namespace]: The configuration for fuzzy, FI and ML pipelines.
    """

    fuzzy_opt = FuzzyOptions()
    fuzzy_opt.initialize()
    fuzzy_opt.parser.set_defaults(
        fuzzy_feature_selection=fuzzy_feature_selection,
        num_fuzzy_features=num_fuzzy_features,
        granular_features=granular_features,
        num_clusters=num_clusters,
        cluster_names=cluster_names,
        num_top_rules=num_top_rules,
        dependent_variable=dependent_variable,
        experiment_name=experiment_name,
        problem_type=problem_type,
    )
    fuzzy_opt = fuzzy_opt.parse()

    fi_opt = FeatureImportanceOptions()
    fi_opt.initialize()
    fi_opt.parser.set_defaults(
        num_features_to_plot=num_features_to_plot,
        permutation_importance_scoring=permutation_importance_scoring,
        permutation_importance_repeat=permutation_importance_repeat,
        shap_reduce_data=shap_reduce_data,
        dependent_variable=dependent_variable,
        experiment_name=experiment_name,
        problem_type=problem_type,
    )
    fi_opt = fi_opt.parse()

    ml_opt = MLOptions()
    ml_opt.initialize()
    ml_opt.parser.set_defaults(
        n_bootstraps=n_bootstraps,
        save_actual_pred_plots=save_actual_pred_plots,
        normalization=normalization,
        dependent_variable=dependent_variable,
        experiment_name=experiment_name,
        data_path=data_path,
        problem_type=problem_type,
    )
    ml_opt = ml_opt.parse()

    return fuzzy_opt, fi_opt, ml_opt


@st.cache_data
def uploaded_file_path(file_name: str) -> str:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The full upload path for the file.
    """
    return Path.home() / "BioFEFIUploads" / file_name


def save_upload(file_to_upload: str, content: str):
    """Save a file given to the UI to disk.

    Args:
        file_to_upload (str): The name of the file to save.
        content (str): The contents to save to the file.
    """
    base_dir = os.path.dirname(file_to_upload)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(file_to_upload, "w") as f:
        f.write(content)


def _pipeline(fuzzy_opts: Namespace, fi_opts: Namespace, ml_opts: Namespace):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (Namespace): Options for fuzzy feature importance.
        fi_opts (Namespace): Options for feature importance.
        ml_opts (Namespace): Options for machine learning.
    """
    seed = ml_opts.random_state
    set_seed(seed)
    ml_logger_instance = Logger(ml_opts.ml_log_dir, ml_opts.experiment_name)
    ml_logger = ml_logger_instance.make_logger()

    data = DataBuilder(ml_opts, ml_logger).ingest()

    # Machine learning
    trained_models = train.run(ml_opts, data, ml_logger)
    close_logger(ml_logger_instance, ml_logger)

    # Feature importance
    fi_logger_instance = Logger(fi_opts.fi_log_dir, fi_opts.experiment_name)
    fi_logger = fi_logger_instance.make_logger()
    gloabl_importance_results, local_importance_results, ensemble_results = (
        feature_importance.run(fi_opts, data, trained_models, fi_logger)
    )
    close_logger(fi_logger_instance, fi_logger)

    # Fuzzy interpretation
    fuzzy_logger_instance = Logger(fuzzy_opts.fuzzy_log_dir, fuzzy_opts.experiment_name)
    fuzzy_logger = fuzzy_logger_instance.make_logger()
    fuzzy_rules = fuzzy_interpretation.run(
        fuzzy_opts, data, trained_models, ensemble_results, fuzzy_logger
    )
    close_logger(fuzzy_logger_instance, fuzzy_logger)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        print("Cancelling pipeline run")
        p.terminate()


st.image("ui/bioFEFI header.png")
# Sidebar
with st.sidebar:
    st.header("Options")
    st.checkbox("Feature Engineering", key=ConfigStateKeys.IsFeatureEngineering)

    # Machine Learning Options
    with st.expander("Machine Learning Options"):
        ml_on = st.checkbox("Machine Learning", key=ConfigStateKeys.IsMachineLearning)
        st.subheader("Machine Learning Options")
        problem_type = st.selectbox("Problem type", ["Classification", "Regression"], key=ConfigStateKeys.ProblemType).lower()
        data_split = st.selectbox("Data split method", ["Holdout", "K-Fold"], key=ConfigStateKeys.DataSplit)
        num_bootstraps = st.number_input("Number of bootstraps", min_value=1, value=10, key=ConfigStateKeys.NumberOfBootstraps)
        save_plots = st.checkbox("Save actual or predicted plots", key=ConfigStateKeys.SavePlots)

        st.write("Model types to use:")
        use_linear = st.checkbox("Linear Model", key=ConfigStateKeys.UseLinear)
        use_rf = st.checkbox("Random Forest", key=ConfigStateKeys.UseRandomForest)
        use_xgb = st.checkbox("XGBoost", key=ConfigStateKeys.UseXGBoost)

        normalization = st.selectbox("Normalization", ["Standardization", "MinMax", "None"])

    # Feature Importance Options
    with st.expander("Feature importance options"):
        fi_on = st.checkbox("Feature Importance", key=ConfigStateKeys.IsFeatureImportance)
        st.write("Global feature importance methods:")
        use_permutation = st.checkbox("Permutation Importance", key=ConfigStateKeys.UsePermutation)
        use_shap = st.checkbox("SHAP", key=ConfigStateKeys.UseShap)

        st.write("Feature importance ensemble methods:")
        use_mean = st.checkbox("Mean", key=ConfigStateKeys.UseMean)
        use_majority = st.checkbox("Majority vote", key=ConfigStateKeys.UseMajorityVote)

        st.write("Local feature importance methods:")
        use_lime = st.checkbox("LIME", key=ConfigStateKeys.UseLime)
        use_local_shap = st.checkbox("Local SHAP", key=ConfigStateKeys.UseLocalShap)

        num_important_features = st.number_input(
            "Number of most important features to plot", min_value=1, value=10, key=ConfigStateKeys.NumberOfImportantFeatures
        )
        scoring_function = st.selectbox(
            "Scoring function for permutation importance", ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'accuracy', 'f1'], key=ConfigStateKeys.ScoringFunction
        )
        num_repetitions = st.number_input(
            "Number of repetitions for permutation importance", min_value=1, value=5, key=ConfigStateKeys.NumberOfRepetitions
        )
        shap_data_percentage = st.slider(
            "Percentage of data to consider for SHAP", 0, 100, 100, key=ConfigStateKeys.ShapDataPercentage
        )

        # Fuzzy Options
        st.subheader("Fuzzy Options")
        fuzzy_feature_selection = st.checkbox("Fuzzy feature selection", key=ConfigStateKeys.FuzzyFeatureSelection)
        num_fuzzy_features = st.number_input(
            "Number of features for fuzzy interpretation", min_value=1, value=5, key=ConfigStateKeys.NumberOfFuzzyFeatures
        )
        granular_features = st.checkbox("Granular features", key=ConfigStateKeys.GranularFeatures)
        num_clusters = st.number_input(
            "Number of clusters for target variable", min_value=2, value=3, key=ConfigStateKeys.NumberOfClusters
        )
        cluster_names = st.text_input("Names of clusters (comma-separated)", key=ConfigStateKeys.ClusterNames)
        num_top_rules = st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
            key=ConfigStateKeys.NumberOfTopRules
        )
    seed = st.number_input("Random seed", value=1221, min_value=0)
# Main body
st.header("Data Upload")
experiment_name = st.text_input("Name of the experiment", key=ConfigStateKeys.ExperimentName)
dependent_variable = st.text_input("Name of the dependent variable", key=ConfigStateKeys.DependentVariableName)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName)
run_button = st.button("Run")


if uploaded_file is not None and run_button:
    upload_path = uploaded_file_path(uploaded_file.name)
    save_upload(upload_path, uploaded_file.read().decode("utf-8"))
    config = build_configuration(
        fuzzy_feature_selection=fuzzy_feature_selection,
        num_fuzzy_features=num_fuzzy_features,
        granular_features=granular_features,
        num_clusters=num_clusters,
        cluster_names=cluster_names,
        dependent_variable=dependent_variable,
        num_features_to_plot=num_important_features,
        permutation_importance_scoring=scoring_function,
        permutation_importance_repeat=num_repetitions,
        shap_reduce_data=shap_data_percentage,
        n_bootstraps=num_bootstraps,
        save_actual_pred_plots=save_actual_pred_plots,
        normalization=normalization,
        data_path=upload_path,
        experiment_name=experiment_name,
        problem_type=problem_type,
    )
    process = Process(target=_pipeline, args=config, daemon=True)
    process.start()
    cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
    df = pd.read_csv(upload_path)
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
