from argparse import Namespace
from multiprocessing import Process
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.plots import plot_box
from biofefi.components.forms import fi_options_form
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import load_models
from biofefi.feature_importance import feature_importance, fuzzy_interpretation
from biofefi.feature_importance.feature_importance_options import (
    FeatureImportanceOptions,
)
from biofefi.feature_importance.fuzzy_options import FuzzyOptions
from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import (
    ConfigStateKeys,
    ProblemTypes,
    PlotOptionKeys,
)

from biofefi.options.enums import ConfigStateKeys, ViewExperimentKeys

from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
)

from biofefi.options.file_paths import (
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
    ml_model_dir,
)
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import set_seed, cancel_pipeline
from biofefi.components.experiments import experiment_selector, model_selector
import streamlit as st
import os


def build_configuration() -> tuple[Namespace, Namespace, Namespace, str]:
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[Namespace, Namespace, Namespace, str]: The configuration for fuzzy, FI and ML pipelines,
        and the experiment name.
    """

    fuzzy_opt = FuzzyOptions()
    fuzzy_opt.initialize()
    if st.session_state.get(ConfigStateKeys.FuzzyFeatureSelection, False):
        fuzzy_opt.parser.set_defaults(
            fuzzy_feature_selection=st.session_state[
                ConfigStateKeys.FuzzyFeatureSelection
            ],
            num_fuzzy_features=st.session_state[ConfigStateKeys.NumberOfFuzzyFeatures],
            granular_features=st.session_state[ConfigStateKeys.GranularFeatures],
            num_clusters=st.session_state[ConfigStateKeys.NumberOfClusters],
            cluster_names=st.session_state[ConfigStateKeys.ClusterNames],
            num_rules=st.session_state[ConfigStateKeys.NumberOfTopRules],
            save_fuzzy_set_plots=st.session_state[PlotOptionKeys.SavePlots],
            # fuzzy_log_dir=
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            problem_type=st.session_state.get(
                ConfigStateKeys.ProblemType, ProblemTypes.Auto
            ).lower(),
            is_granularity=st.session_state[ConfigStateKeys.GranularFeatures],
        )
    fuzzy_opt = fuzzy_opt.parse()

    fi_opt = FeatureImportanceOptions()
    fi_opt.initialize()
    if st.session_state.get(ConfigStateKeys.IsFeatureImportance, False):
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
            problem_type=st.session_state.get(
                ConfigStateKeys.ProblemType, ProblemTypes.Auto
            ).lower(),
            is_feature_importance=st.session_state[ConfigStateKeys.IsFeatureImportance],
            # fi_log_dir=
            angle_rotate_xaxis_labels=st.session_state[
                PlotOptionKeys.RotateXAxisLabels
            ],
            angle_rotate_yaxis_labels=st.session_state[
                PlotOptionKeys.RotateYAxisLabels
            ],
            save_feature_importance_plots=st.session_state[PlotOptionKeys.SavePlots],
            save_feature_importance_options=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceOptions
            ],
            save_feature_importance_results=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceResults
            ],
            local_importance_methods=st.session_state[
                ConfigStateKeys.LocalImportanceFeatures
            ],
            feature_importance_ensemble=st.session_state[
                ConfigStateKeys.EnsembleMethods
            ],
            global_importance_methods=st.session_state[
                ConfigStateKeys.GlobalFeatureImportanceMethods
            ],
        )
    fi_opt = fi_opt.parse()

    return (
        fuzzy_opt,
        fi_opt,
        st.session_state[ConfigStateKeys.ExperimentName],
        st.session_state[ConfigStateKeys.ExplainModels],
    )


def pipeline(
    fuzzy_opts: Namespace,
    fi_opts: Namespace,
    experiment_name: str,
    explain_models: list,
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (Namespace): Options for fuzzy feature importance.
        fi_opts (Namespace): Options for feature importance.
        ml_opts (Namespace): Options for machine learning.
        experiment_name (str): The name of the experiment.
    """
    seed = fuzzy_opts.random_state
    set_seed(seed)
    logger_instance = Logger(log_dir(experiment_name))
    logger = logger_instance.make_logger()

    data = DataBuilder(fuzzy_opts, logger).ingest()

    ## Models will already be trained before feature importance
    trained_models = load_models(ml_model_dir(experiment_name))

    trained_models = [model for model in trained_models if model in explain_models]

    # Feature importance
    if fi_opts.is_feature_importance:
        gloabl_importance_results, local_importance_results, ensemble_results = (
            feature_importance.run(fi_opts, data, trained_models, logger)
        )

        # Fuzzy interpretation
        if fuzzy_opts.fuzzy_feature_selection:
            fuzzy_rules = fuzzy_interpretation.run(
                fuzzy_opts, fuzzy_opts, data, trained_models, ensemble_results, logger
            )

    # Close the logger
    close_logger(logger_instance, logger)


# Set page contents
st.set_page_config(
    page_title="Feature Importance",
    page_icon=sidebar_logo(),
)


st.header("Feature Importance")
st.write(
    """
    This page provides options for exploring and customising feature importance and interpretability methods in the trained machine learning models. 
    You can configure global and local feature importance techniques, select ensemble approaches, and apply fuzzy feature selection. Options include tuning scoring functions, 
    setting data percentages for SHAP analysis, and configuring rules for fuzzy synergy analysis to gain deeper insights into model behavior.
    """
)

# Get the base directory of all experiments
base_dir = biofefi_experiments_base_dir()
choices = os.listdir(base_dir)
# Filter out hidden files and directories
choices = filter(lambda x: not x.startswith("."), choices)
# Filter out files
choices = filter(lambda x: os.path.isdir(os.path.join(base_dir, x)), choices)

experiment_selector(choices)

if experiment_name := st.session_state.get(ViewExperimentKeys.ExperimentName):

    st.session_state[ConfigStateKeys.ExperimentName] = base_dir / experiment_name
    experiment_name = st.session_state[ConfigStateKeys.ExperimentName]

    model_choices = os.listdir(ml_model_dir(experiment_name))
    model_choices = filter(lambda x: x.endswith(".pkl"), model_choices)

    model_selector(model_choices)

    if model_choices := st.session_state.get(ConfigStateKeys.ExplainModels):

        fi_options_form()

        if st.button("Run Feature Importance"):
            config = build_configuration()
            process = Process(target=pipeline, args=config, daemon=True)
            process.start()
            cancel_button = st.button(
                "Cancel", on_click=cancel_pipeline, args=(process,)
            )
            with st.spinner(
                "Feature Importance pipeline is running in the background. Check the logs for progress."
            ):
                # wait for the process to finish or be cancelled
                process.join()
            st.session_state[ConfigStateKeys.LogBox] = get_logs(
                log_dir(experiment_name)
            )
            log_box()
            fi_plots = fi_plot_dir(experiment_name)
            if fi_plots.exists():
                plot_box(fi_plots, "Feature importance plots")
            fuzzy_plots = fuzzy_plot_dir(experiment_name)
            if fuzzy_plots.exists():
                plot_box(fuzzy_plots, "Fuzzy plots")
