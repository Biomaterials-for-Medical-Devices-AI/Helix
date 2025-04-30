"""Feature Importance page for Helix.

This page provides options for analyzing feature importance using various methods:
- Global methods (Permutation, SHAP)
- Local methods (LIME, SHAP)
- Ensemble methods
- Fuzzy feature importance
"""

from multiprocessing import Process
from pathlib import Path

import streamlit as st

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector, model_selector
from helix.components.forms.forms_fi import fi_options_form
from helix.components.images.logos import sidebar_logo
from helix.components.logs import log_box
from helix.components.plots import plot_box_v2
from helix.feature_importance import feature_importance
from helix.options.data import DataOptions
from helix.options.enums import (
    ExecutionStateKeys,
    FeatureImportanceStateKeys,
    FuzzyStateKeys,
    ViewExperimentKeys,
)
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.file_paths import (
    data_options_path,
    execution_options_path,
    fi_options_path,
    fi_plot_dir,
    helix_experiments_base_dir,
    log_dir,
    ml_model_dir,
    plot_options_path,
)
from helix.options.fuzzy import FuzzyOptions
from helix.options.plotting import PlottingOptions
from helix.services.configuration import (
    load_data_options,
    load_execution_options,
    load_plot_options,
    save_options,
)
from helix.services.data import TabularData, ingest_data
from helix.services.experiments import (
    delete_previous_fi_results,
    find_previous_fi_results,
    get_experiments,
)
from helix.services.logs import get_logs
from helix.services.ml_models import load_models_to_explain
from helix.utils.logging_utils import Logger, close_logger
from helix.utils.utils import cancel_pipeline, set_seed


def build_configuration() -> tuple[
    FuzzyOptions | None,
    FeatureImportanceOptions,
    ExecutionOptions,
    PlottingOptions,
    DataOptions,
    str,
    list,
]:
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[
        FuzzyOptions | None,
        FeatureImportanceOptions,
        ExecutionOptions,
        PlottingOptions,
        DataOptions,
        str,
        list]:
        - The options for fuzzy,
        - The options for feature importance
        - The options for pipeline execution
        - The plotting options
        - The data options
        - The experiment name
        - The list of models to explain.
    """
    biofefi_base_dir = helix_experiments_base_dir()
    experiment_name = st.session_state[ExecutionStateKeys.ExperimentName]

    # Load plotting options
    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)
    plotting_options = load_plot_options(path_to_plot_opts)

    # Load executuon options
    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)
    exec_opt = load_execution_options(path_to_exec_opts)

    # Load data options
    path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)
    data_options = load_data_options(path_to_data_opts)

    # Set up fuzzy options
    fuzzy_opt = None
    # if st.session_state.get(FuzzyStateKeys.FuzzyFeatureSelection, False):
    #     fuzzy_opt = FuzzyOptions(
    #         fuzzy_feature_selection=st.session_state[
    #             FuzzyStateKeys.FuzzyFeatureSelection
    #         ],
    #         number_fuzzy_features=st.session_state[
    #             FuzzyStateKeys.NumberOfFuzzyFeatures
    #         ],
    #         granular_features=st.session_state[FuzzyStateKeys.GranularFeatures],
    #         number_clusters=st.session_state[FuzzyStateKeys.NumberOfClusters],
    #         cluster_names=st.session_state.get(FuzzyStateKeys.ClusterNames, "").split(
    #             ", "
    #         ),
    #         number_rules=st.session_state[FuzzyStateKeys.NumberOfTopRules],
    #         save_fuzzy_set_plots=plotting_options.save_plots,
    #         fuzzy_log_dir=str(
    #             log_dir(
    #                 biofefi_base_dir
    #                 / st.session_state[ViewExperimentKeys.ExperimentName]
    #             )
    #             / "fuzzy"
    #         ),
    #     )

    # Set up feature importance options
    fi_opt = FeatureImportanceOptions(
        num_features_to_plot=st.session_state[
            FeatureImportanceStateKeys.NumberOfImportantFeatures
        ],
        permutation_importance_scoring=st.session_state[
            FeatureImportanceStateKeys.ScoringFunction
        ],
        permutation_importance_repeat=st.session_state[
            FeatureImportanceStateKeys.NumberOfRepetitions
        ],
        save_feature_importance_plots=plotting_options.save_plots,
        fi_log_dir=str(
            log_dir(
                biofefi_base_dir / st.session_state[ViewExperimentKeys.ExperimentName]
            )
            / "fi"
        ),
        save_feature_importance_options=st.session_state[
            FeatureImportanceStateKeys.SaveFeatureImportanceOptions
        ],
        save_feature_importance_results=st.session_state[
            FeatureImportanceStateKeys.SaveFeatureImportanceResults
        ],
        local_importance_methods=st.session_state[
            FeatureImportanceStateKeys.LocalImportanceFeatures
        ],
        feature_importance_ensemble=st.session_state[
            FeatureImportanceStateKeys.EnsembleMethods
        ],
        global_importance_methods=st.session_state[
            FeatureImportanceStateKeys.GlobalFeatureImportanceMethods
        ],
    )

    return (
        fuzzy_opt,
        fi_opt,
        exec_opt,
        plotting_options,
        data_options,
        experiment_name,
        st.session_state[FeatureImportanceStateKeys.ExplainModels],
    )


def pipeline(
    fuzzy_opts: FuzzyOptions,
    fi_opts: FeatureImportanceOptions,
    exec_opts: ExecutionOptions,
    plot_opts: PlottingOptions,
    experiment_name: str,
    explain_models: list,
    data: TabularData,
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (FuzzyOptions): Options for fuzzy feature importance.
        fi_opts (FeatureImportanceOptions): Options for feature importance.
        exec_opts (ExecutionOptions): Options for pipeline execution.
        plot_opts (PlottingOptions): Options for plotting.
        experiment_name (str): The experiment name.
        explain_models (list): The models to analyse.
        data (TabularData): The data that will be used in the pipeline.
    """
    biofefi_base_dir = helix_experiments_base_dir()
    seed = exec_opts.random_state
    set_seed(seed)

    # Models will already be trained before feature importance
    trained_models = load_models_to_explain(
        ml_model_dir(biofefi_base_dir / experiment_name), explain_models
    )

    fi_logger_instance = Logger(Path(fi_opts.fi_log_dir))
    fi_logger = fi_logger_instance.make_logger()

    # Load data options to get path to data
    path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)
    data_opts = load_data_options(path_to_data_opts)
    # Feature importance
    (
        gloabl_importance_results,
        local_importance_results,
        ensemble_results,
    ) = feature_importance.run(
        fi_opt=fi_opts,
        exec_opt=exec_opts,
        plot_opt=plot_opts,
        data=data,
        models=trained_models,
        data_path=Path(data_opts.data_path),
        logger=fi_logger,
    )
    # Close the fi logger
    close_logger(fi_logger_instance, fi_logger)

    # Fuzzy interpretation
    # if fuzzy_opts is not None and fuzzy_opts.fuzzy_feature_selection:
    #     fuzzy_logger_instance = Logger(Path(fuzzy_opts.fuzzy_log_dir))
    #     fuzzy_logger = fuzzy_logger_instance.make_logger()
    #     fuzzy_interpretation.run(
    #         fuzzy_opt=fuzzy_opts,
    #         fi_opt=fi_opts,
    #         exec_opt=exec_opts,
    #         plot_opt=plot_opts,
    #         data=data,
    #         models=trained_models,
    #         ensemble_results=ensemble_results,
    #         logger=fuzzy_logger,
    #     )
    #     close_logger(fuzzy_logger_instance, fuzzy_logger)


# Set page contents
st.set_page_config(
    page_title="Feature Importance",
    page_icon=sidebar_logo(),
)


st.header("Feature Importance")
# st.write(
#     """
#     This page provides options for exploring and customising feature importance and
#     interpretability methods in the trained machine learning models.
#     You can configure global and local feature importance techniques,
#     select ensemble approaches, and apply fuzzy feature importance.
#     Options include tuning scoring functions, setting data percentages
#     for SHAP analysis, and configuring rules for fuzzy synergy analysis
#     to gain deeper insights into model behaviour.
#     """
# )
st.write(
    """
    This page provides options for exploring and customising feature importance and
    interpretability methods in the trained machine learning models.
    You can configure global, local feature importance techniques and
    select ensemble approaches.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)
base_dir = helix_experiments_base_dir()


if experiment_name:

    previous_results_exist = find_previous_fi_results(
        helix_experiments_base_dir() / experiment_name
    )
    display_options(base_dir / experiment_name)

    path_to_exec_opts = execution_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    exec_opt = load_execution_options(path_to_exec_opts)

    if previous_results_exist:
        st.warning("You have run feature importance in this experiment previously.")
        st.checkbox(
            "Would you like to rerun feature importance? This will overwrite the existing results.",
            value=True,
            key=FuzzyStateKeys.RerunFI,
        )
    else:
        st.session_state[FuzzyStateKeys.RerunFI] = True

    if st.session_state[FuzzyStateKeys.RerunFI]:

        st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name
        st.session_state[ExecutionStateKeys.ProblemType] = exec_opt.problem_type

        model_dir = ml_model_dir(base_dir / experiment_name)
        if model_dir.exists():
            model_choices = list(
                filter(
                    lambda x: x.endswith(".pkl"), [x.name for x in model_dir.iterdir()]
                )
            )
        else:
            model_choices = []

        if len(model_choices) == 0:
            st.info(
                "You don't have any trained models in this experiment. "
                "Go to **Train Models** to create some models to evaulate."
            )
        elif st.toggle(
            "Explain all models",
            key=FeatureImportanceStateKeys.ExplainAllModels,
        ):
            st.session_state[FeatureImportanceStateKeys.ExplainModels] = model_choices
        else:
            model_selector(model_choices)

        if model_choices := st.session_state.get(
            FeatureImportanceStateKeys.ExplainModels
        ):
            fi_options_form()

            if st.button("Run Feature Importance", type="primary"):
                delete_previous_fi_results(base_dir / experiment_name)
                (
                    fuzzy_opts,
                    fi_opts,
                    exec_opts,
                    plot_opts,
                    data_opts,
                    exp_name,
                    models_to_explaion,
                ) = build_configuration()
                # Create the logger
                logger_instance = Logger()
                logger = logger_instance.make_logger()
                # Ingest the data
                data = ingest_data(exec_opts, data_opts, logger)
                # save FI options
                fi_options_file = fi_options_path(base_dir / experiment_name)
                save_options(fi_options_file, fi_opts)
                # save Fuzzy options if configured
                # if fuzzy_opts is not None:
                #     fuzzy_options_file = fuzzy_options_path(base_dir / experiment_name)
                #     save_options(fuzzy_options_file, fuzzy_opts)

                process = Process(
                    target=pipeline,
                    args=(
                        fuzzy_opts,
                        fi_opts,
                        exec_opts,
                        plot_opts,
                        exp_name,
                        models_to_explaion,
                        data,
                    ),
                    daemon=True,
                )
                process.start()
                cancel_button = st.button(
                    "Cancel", on_click=cancel_pipeline, args=(process,)
                )
                with st.spinner(
                    "Feature Importance pipeline is running in the background. "
                    "Check the logs for progress."
                ):
                    # wait for the process to finish or be cancelled
                    process.join()
                try:
                    st.session_state[FeatureImportanceStateKeys.FILogBox] = get_logs(
                        log_dir(base_dir / experiment_name) / "fi"
                    )
                    st.session_state[FuzzyStateKeys.FuzzyLogBox] = get_logs(
                        log_dir(base_dir / experiment_name) / "fuzzy"
                    )
                    log_box(
                        box_title="Feature Importance Logs",
                        key=FeatureImportanceStateKeys.FILogBox,
                    )
                    # log_box(box_title="Fuzzy FI Logs", key=FuzzyStateKeys.FuzzyLogBox)
                except NotADirectoryError:
                    pass
                fi_plots = fi_plot_dir(base_dir / experiment_name)
                mean_plots = [
                    p
                    for p in fi_plots.iterdir()
                    if p.name.endswith("-all-folds-mean.png")  # mean global FI
                    or p.name.startswith("local-")  # local plots
                    or p.name.startswith("ensemble-")  # ensemble plots
                ]
                plot_box_v2(mean_plots, "Feature importance plots")
                # fuzzy_plots = fuzzy_plot_dir(base_dir / experiment_name)
                # plot_box(fuzzy_plots, "Fuzzy plots")

    else:
        st.success(
            "You have chosen not to rerun the feature importance experiments. "
            "You can proceed to see the experiment results."
        )
