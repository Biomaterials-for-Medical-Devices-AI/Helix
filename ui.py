from argparse import Namespace
from multiprocessing import Process
from numba.cuda import initialize
from components.images.logos import header_logo, sidebar_logo
from feature_importance import feature_importance, fuzzy_interpretation
from feature_importance.feature_importance_options import FeatureImportanceOptions
from feature_importance.fuzzy_options import FuzzyOptions
from machine_learning import train
from machine_learning.call_methods import save_actual_pred_plots
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from options.enums import ConfigStateKeys
from options.file_paths import uploaded_file_path, log_dir
from utils.logging_utils import Logger, close_logger
from utils.utils import set_seed
import streamlit as st
import os


import pandas as pd


def build_configuration() -> tuple[Namespace, Namespace, Namespace, str]:
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[Namespace, Namespace, Namespace, str]: The configuration for fuzzy, FI and ML pipelines,
        and the experiment name.
    """

    fuzzy_opt = FuzzyOptions()
    fuzzy_opt.initialize()
    fuzzy_opt.parser.set_defaults(
        fuzzy_feature_selection=st.session_state[ConfigStateKeys.FuzzyFeatureSelection],
        num_fuzzy_features=st.session_state[ConfigStateKeys.NumberOfFuzzyFeatures],
        granular_features=st.session_state[ConfigStateKeys.GranularFeatures],
        num_clusters=st.session_state[ConfigStateKeys.NumberOfClusters],
        cluster_names=st.session_state[ConfigStateKeys.ClusterNames],
        num_rules=st.session_state[ConfigStateKeys.NumberOfTopRules],
        save_fuzzy_set_plots=st.session_state[ConfigStateKeys.SaveFuzzySetPlots],
        # fuzzy_log_dir=
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
        problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
        is_granularity=st.session_state[ConfigStateKeys.GranularFeatures],
    )
    fuzzy_opt = fuzzy_opt.parse()

    fi_opt = FeatureImportanceOptions()
    fi_opt.initialize()
    fi_opt.parser.set_defaults(
        num_features_to_plot=st.session_state[
            ConfigStateKeys.NumberOfImportantFeatures
        ],
        permutation_importance_scoring=st.session_state[
            ConfigStateKeys.ScoringFunction
        ],
        permutation_importance_repeat=st.session_state[
            ConfigStateKeys.NumberOfRepetitions
        ],
        shap_reduce_data=st.session_state[ConfigStateKeys.ShapDataPercentage],
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
        problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
        is_feature_importance=st.session_state[ConfigStateKeys.IsFeatureImportance],
        # fi_log_dir=
        angle_rotate_xaxis_labels=st.session_state[ConfigStateKeys.RotateXAxisLabels],
        angle_rotate_yaxis_labels=st.session_state[ConfigStateKeys.RotateYAxisLabels],
        save_feature_importance_plots=st.session_state[
            ConfigStateKeys.SaveFeatureImportancePlots
        ],
        save_feature_importance_options=st.session_state[
            ConfigStateKeys.SaveFeatureImportanceOptions
        ],
        save_feature_importance_results=st.session_state[
            ConfigStateKeys.SaveFeatureImportanceResults
        ],
        local_importance_methods=st.session_state[
            ConfigStateKeys.LocalImportanceFeatures
        ],
        feature_importance_ensemble=st.session_state[ConfigStateKeys.EnsembleMethods],
        global_importance_methods=st.session_state[
            ConfigStateKeys.GlobalFeatureImportanceMethods
        ],
    )
    fi_opt = fi_opt.parse()

    ml_opt = MLOptions()
    ml_opt.initialize()
    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName].name,
        st.session_state[ConfigStateKeys.ExperimentName],
    )
    ml_opt.parser.set_defaults(
        n_bootstraps=st.session_state[ConfigStateKeys.NumberOfBootstraps],
        save_actual_pred_plots=save_actual_pred_plots,
        normalization=st.session_state[ConfigStateKeys.Normalization],
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
        data_path=path_to_data,
        data_split=st.session_state[ConfigStateKeys.DataSplit],
        model_types=st.session_state[ConfigStateKeys.ModelTypes],
        # ml_log_dir=
        problem_type=st.session_state[ConfigStateKeys.ProblemType].lower(),
        random_state=st.session_state[ConfigStateKeys.RandomSeed],
        is_machine_learning=st.session_state[ConfigStateKeys.IsMachineLearning],
    )
    ml_opt = ml_opt.parse()

    return fuzzy_opt, fi_opt, ml_opt, st.session_state[ConfigStateKeys.ExperimentName]


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


def pipeline(
    fuzzy_opts: Namespace, fi_opts: Namespace, ml_opts: Namespace, experiment_name: str
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (Namespace): Options for fuzzy feature importance.
        fi_opts (Namespace): Options for feature importance.
        ml_opts (Namespace): Options for machine learning.
        experiment_name (str): The name of the experiment.
    """
    seed = ml_opts.random_state
    set_seed(seed)
    logger_instance = Logger(log_dir(experiment_name))
    logger = logger_instance.make_logger()

    data = DataBuilder(ml_opts, logger).ingest()

    # Machine learning
    if ml_opts.is_machine_learning:
        trained_models = train.run(ml_opts, data, logger)

    # Feature importance
    if fi_opts.is_feature_importance:
        gloabl_importance_results, local_importance_results, ensemble_results = (
            feature_importance.run(fi_opts, data, trained_models, logger)
        )

    # Fuzzy interpretation
    if fuzzy_opts.fuzzy_feature_selection:
        fuzzy_rules = fuzzy_interpretation.run(
            fuzzy_opts, ml_opts, data, trained_models, ensemble_results, logger
        )

    # Close the logger
    close_logger(logger_instance, logger)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        p.terminate()


header_logo()
# Sidebar
sidebar_logo()
with st.sidebar:
    st.header("Options")
    # st.checkbox("Feature Engineering", key=ConfigStateKeys.IsFeatureEngineering)

    # Machine Learning Options
    ml_on = st.checkbox("Machine Learning", key=ConfigStateKeys.IsMachineLearning)
    if ml_on:
        with st.expander("Machine Learning Options"):
            st.subheader("Machine Learning Options")
            problem_type = st.selectbox(
                "Problem type",
                ["Classification", "Regression"],
                key=ConfigStateKeys.ProblemType,
            ).lower()
            data_split = st.selectbox("Data split method", ["Holdout", "K-Fold"])
            if data_split == "Holdout":
                split_size = st.number_input(
                    "Test split",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                )
                st.session_state[ConfigStateKeys.DataSplit] = {
                    "type": "holdout",
                    "test_size": split_size,
                }
            elif data_split == "K-Fold":
                split_size = st.number_input(
                    "n splits",
                    min_value=0,
                    value=5,
                )
                st.session_state[ConfigStateKeys.DataSplit] = {
                    "type": "kfold",
                    "n_splits": split_size,
                }
            else:
                split_size = None
            num_bootstraps = st.number_input(
                "Number of bootstraps",
                min_value=1,
                value=10,
                key=ConfigStateKeys.NumberOfBootstraps,
            )
            save_plots = st.checkbox(
                "Save actual or predicted plots", key=ConfigStateKeys.SavePlots
            )

            st.write("Model types to use:")
            model_types = {}
            use_linear = st.checkbox("Linear Model", value=True)
            if use_linear:
                st.write("Options:")
                fit_intercept = st.checkbox("Fit intercept")
                model_types["Linear Model"] = {
                    "use": use_linear,
                    "params": {
                        "fit_intercept": fit_intercept,
                    },
                }
                st.divider()

            use_rf = st.checkbox("Random Forest", value=True)
            if use_rf:
                st.write("Options:")
                n_estimators_rf = st.number_input(
                    "Number of estimators", value=300, key="n_estimators_rf"
                )
                min_samples_split = st.number_input("Minimum samples split", value=2)
                min_samples_leaf = st.number_input("Minimum samples leaf", value=1)
                max_depth_rf = st.number_input(
                    "Maximum depth", value=6, key="max_depth_rf"
                )
                model_types["Random Forest"] = {
                    "use": use_rf,
                    "params": {
                        "n_estimators": n_estimators_rf,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "max_depth": max_depth_rf,
                    },
                }
                st.divider()

            use_xgb = st.checkbox("XGBoost", value=True)
            if use_xgb:
                st.write("Options:")
                n_estimators_xgb = st.number_input(
                    "Number of estimators", value=300, key="n_estimators_xgb"
                )
                max_depth_xbg = st.number_input(
                    "Maximum depth", value=6, key="max_depth_xgb"
                )
                learning_rate = st.number_input("Learning rate", value=0.01)
                subsample = st.number_input("Subsample size", value=0.5)
                model_types["XGBoost"] = {
                    "use": use_xgb,
                    "params": {
                        "kwargs": {
                            "n_estimators": n_estimators_xgb,
                            "max_depth": max_depth_xbg,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                        }
                    },
                }
                st.divider()
            st.session_state[ConfigStateKeys.ModelTypes] = model_types

            normalization = st.selectbox(
                "Normalization",
                ["Standardization", "MinMax", "None"],
                key=ConfigStateKeys.Normalization,
            )

    # Feature Importance Options
    fi_on = st.checkbox("Feature Importance", key=ConfigStateKeys.IsFeatureImportance)
    if fi_on:
        with st.expander("Feature importance options"):
            st.write("Global feature importance methods:")
            global_methods = {}
            use_permutation = st.checkbox("Permutation Importance")
            global_methods["Permutation Importance"] = {
                "type": "global",
                "value": use_permutation,
            }
            use_shap = st.checkbox("SHAP")
            global_methods["SHAP"] = {"type": "global", "value": use_shap}
            st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = (
                global_methods
            )

            st.write("Feature importance ensemble methods:")
            ensemble_methods = {}
            use_mean = st.checkbox("Mean")
            ensemble_methods["Mean"] = use_mean
            use_majority = st.checkbox("Majority vote")
            ensemble_methods["Majority Vote"] = use_majority
            st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

            st.write("Local feature importance methods:")
            local_importance_methods = {}
            use_lime = st.checkbox("LIME")
            local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
            use_local_shap = st.checkbox("Local SHAP")
            local_importance_methods["SHAP"] = {
                "type": "local",
                "value": use_local_shap,
            }
            st.session_state[ConfigStateKeys.LocalImportanceFeatures] = (
                local_importance_methods
            )

            num_important_features = st.number_input(
                "Number of most important features to plot",
                min_value=1,
                value=10,
                key=ConfigStateKeys.NumberOfImportantFeatures,
            )
            scoring_function = st.selectbox(
                "Scoring function for permutation importance",
                [
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "accuracy",
                    "f1",
                ],
                key=ConfigStateKeys.ScoringFunction,
            )
            num_repetitions = st.number_input(
                "Number of repetitions for permutation importance",
                min_value=1,
                value=5,
                key=ConfigStateKeys.NumberOfRepetitions,
            )
            shap_data_percentage = st.slider(
                "Percentage of data to consider for SHAP",
                0,
                100,
                100,
                key=ConfigStateKeys.ShapDataPercentage,
            )
            angle_rotate_xaxis_labels = st.number_input(
                "Angle to rotate X-axis labels",
                min_value=0,
                max_value=90,
                value=10,
                key=ConfigStateKeys.RotateXAxisLabels,
            )
            angle_rotate_yaxis_labels = st.number_input(
                "Angle to rotate Y-axis labels",
                min_value=0,
                max_value=90,
                value=60,
                key=ConfigStateKeys.RotateYAxisLabels,
            )
            save_feature_importance_plots = st.checkbox(
                "Save feature importance plots",
                key=ConfigStateKeys.SaveFeatureImportancePlots,
            )
            save_feature_importance_options = st.checkbox(
                "Save feature importance options",
                key=ConfigStateKeys.SaveFeatureImportanceOptions,
            )
            save_feature_importance_results = st.checkbox(
                "Save feature importance results",
                key=ConfigStateKeys.SaveFeatureImportanceResults,
            )

            # Fuzzy Options
            st.subheader("Fuzzy Options")
            fuzzy_feature_selection = st.checkbox(
                "Fuzzy feature selection", key=ConfigStateKeys.FuzzyFeatureSelection
            )
            if fuzzy_feature_selection:
                num_fuzzy_features = st.number_input(
                    "Number of features for fuzzy interpretation",
                    min_value=1,
                    value=5,
                    key=ConfigStateKeys.NumberOfFuzzyFeatures,
                )
                granular_features = st.checkbox(
                    "Granular features", key=ConfigStateKeys.GranularFeatures
                )
                num_clusters = st.number_input(
                    "Number of clusters for target variable",
                    min_value=2,
                    value=3,
                    key=ConfigStateKeys.NumberOfClusters,
                )
                cluster_names = st.text_input(
                    "Names of clusters (comma-separated)",
                    key=ConfigStateKeys.ClusterNames,
                )
                num_top_rules = st.number_input(
                    "Number of top occurring rules for fuzzy synergy analysis",
                    min_value=1,
                    value=10,
                    key=ConfigStateKeys.NumberOfTopRules,
                )
                save_fuzzy_set_plots = st.checkbox(
                    "Save fuzzy set plots", key=ConfigStateKeys.SaveFuzzySetPlots
                )

    seed = st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )
# Main body
st.header("Data Upload")
experiment_name = st.text_input(
    "Name of the experiment", key=ConfigStateKeys.ExperimentName
)
dependent_variable = st.text_input(
    "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
)
uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
)
run_button = st.button("Run")


if uploaded_file is not None and run_button:
    upload_path = uploaded_file_path(uploaded_file.name, experiment_name)
    save_upload(upload_path, uploaded_file.read().decode("utf-8"))
    config = build_configuration()
    process = Process(target=pipeline, args=config, daemon=True)
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
