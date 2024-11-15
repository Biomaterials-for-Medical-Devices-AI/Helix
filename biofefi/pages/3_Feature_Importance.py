from argparse import Namespace
from multiprocessing import Process
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.plots import plot_box
from biofefi.components.forms import fi_options_form
from biofefi.options.choices import PROBLEM_TYPES
from biofefi.services.experiments import get_experiments
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import load_models_to_explain
from biofefi.feature_importance import feature_importance, fuzzy_interpretation
from biofefi.feature_importance.feature_importance_options import (
    FeatureImportanceOptions,
)
from biofefi.feature_importance.fuzzy_options import FuzzyOptions
from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import (
    ConfigStateKeys,
    ProblemTypes,
)

from biofefi.options.enums import ConfigStateKeys, ViewExperimentKeys

from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
    plot_options_path,
    uploaded_file_path,
)

from biofefi.options.file_paths import (
    fi_plot_dir,
    fuzzy_plot_dir,
    log_dir,
    ml_model_dir,
)
from biofefi.services.plotting import load_plot_options
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import set_seed, cancel_pipeline
from biofefi.components.experiments import (
    experiment_selector,
    model_selector,
    data_selector,
)
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
    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName],
        biofefi_experiments_base_dir()
        / st.session_state[ViewExperimentKeys.ExperimentName],
    )
    path_to_plot_opts = plot_options_path(
        biofefi_experiments_base_dir()
        / st.session_state[ViewExperimentKeys.ExperimentName]
    )
    plotting_options = load_plot_options(path_to_plot_opts)
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
            angle_rotate_xaxis_labels=plotting_options.angle_rotate_xaxis_labels,
            angle_rotate_yaxis_labels=plotting_options.angle_rotate_yaxis_labels,
            plot_axis_font_size=plotting_options.plot_axis_font_size,
            plot_axis_tick_size=plotting_options.plot_axis_tick_size,
            plot_title_font_size=plotting_options.plot_title_font_size,
            plot_font_family=plotting_options.plot_font_family,
            plot_colour_scheme=plotting_options.plot_colour_scheme,
            save_fuzzy_set_plots=plotting_options.save_plots,
            fuzzy_log_dir=log_dir(
                biofefi_experiments_base_dir()
                / st.session_state[ViewExperimentKeys.ExperimentName]
            )
            / "fuzzy",
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            data_path=path_to_data,
            problem_type=st.session_state.get(
                ConfigStateKeys.ProblemType, ProblemTypes.Auto
            ).lower(),
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
        data_path=path_to_data,
        problem_type=st.session_state.get(
            ConfigStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        is_feature_importance=True,
        angle_rotate_xaxis_labels=plotting_options.angle_rotate_xaxis_labels,
        angle_rotate_yaxis_labels=plotting_options.angle_rotate_yaxis_labels,
        plot_axis_font_size=plotting_options.plot_axis_font_size,
        plot_axis_tick_size=plotting_options.plot_axis_tick_size,
        plot_title_font_size=plotting_options.plot_title_font_size,
        plot_colour_scheme=plotting_options.plot_colour_scheme,
        plot_font_family=plotting_options.plot_font_family,
        save_feature_importance_plots=plotting_options.save_plots,
        fi_log_dir=log_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ViewExperimentKeys.ExperimentName]
        )
        / "fi",
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
    seed = fi_opts.random_state
    set_seed(seed)
    fi_logger_instance = Logger(
        log_dir(biofefi_experiments_base_dir() / experiment_name) / "fi"
    )
    fi_logger = fi_logger_instance.make_logger()

    data = DataBuilder(fi_opts, fi_logger).ingest()

    ## Models will already be trained before feature importance
    trained_models = load_models_to_explain(
        ml_model_dir(biofefi_experiments_base_dir() / experiment_name), explain_models
    )

    # Feature importance
    if fi_opts.is_feature_importance:
        (
            gloabl_importance_results,
            local_importance_results,
            ensemble_results,
        ) = feature_importance.run(fi_opts, data, trained_models, fi_logger)

        # Fuzzy interpretation
        if fuzzy_opts.fuzzy_feature_selection:
            fuzzy_logger_instance = Logger(
                log_dir(biofefi_experiments_base_dir() / experiment_name) / "fuzzy"
            )
            fuzzy_logger = fuzzy_logger_instance.make_logger()
            fuzzy_rules = fuzzy_interpretation.run(
                fuzzy_opts,
                fuzzy_opts,
                data,
                trained_models,
                ensemble_results,
                fuzzy_logger,
            )
            close_logger(fuzzy_logger_instance, fuzzy_logger)

    # Close the fi logger
    close_logger(fi_logger_instance, fi_logger)


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


choices = get_experiments()
experiment_name = experiment_selector(choices)

if experiment_name:
    st.session_state[ConfigStateKeys.ExperimentName] = experiment_name

    data_choices = os.listdir(biofefi_experiments_base_dir() / experiment_name)
    data_choices = filter(lambda x: x.endswith(".csv"), data_choices)

    data_selector(data_choices)

    # Fuzzy options require this
    # TODO: get this from a saved configuration from ML
    st.selectbox(
        "Problem type",
        PROBLEM_TYPES,
        key=ConfigStateKeys.ProblemType,
    )
    model_choices = os.listdir(
        ml_model_dir(biofefi_experiments_base_dir() / experiment_name)
    )
    model_choices = [x for x in model_choices if x.endswith(".pkl")]

    explain_all_models = st.toggle(
        "Explain all models", key=ConfigStateKeys.ExplainAllModels
    )

    if explain_all_models:
        st.session_state[ConfigStateKeys.ExplainModels] = model_choices
    else:
        model_selector(model_choices)

    if model_choices := st.session_state.get(
        ConfigStateKeys.ExplainModels
    ) and st.session_state.get(ConfigStateKeys.UploadedFileName):
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
            try:
                st.session_state[ConfigStateKeys.FILogBox] = get_logs(
                    log_dir(biofefi_experiments_base_dir() / experiment_name) / "fi"
                )
                st.session_state[ConfigStateKeys.FuzzyLogBox] = get_logs(
                    log_dir(biofefi_experiments_base_dir() / experiment_name) / "fuzzy"
                )
                log_box(
                    box_title="Feature Importance Logs", key=ConfigStateKeys.FILogBox
                )
                log_box(box_title="Fuzzy FI Logs", key=ConfigStateKeys.FuzzyLogBox)
            except NotADirectoryError:
                pass
            fi_plots = fi_plot_dir(biofefi_experiments_base_dir() / experiment_name)
            if fi_plots.exists():
                plot_box(fi_plots, "Feature importance plots")
            fuzzy_plots = fuzzy_plot_dir(
                biofefi_experiments_base_dir() / experiment_name
            )
            if fuzzy_plots.exists():
                plot_box(fuzzy_plots, "Fuzzy plots")
