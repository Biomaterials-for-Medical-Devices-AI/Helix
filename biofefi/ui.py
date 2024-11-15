from argparse import Namespace
from multiprocessing import Process
from biofefi.components.images.logos import header_logo, sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.forms import data_upload_form
from biofefi.components.plots import plot_box
from biofefi.components.configuration import (
    ml_options_box,
    plot_options_box,
    fi_options_box,
)
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import save_model, load_models
from biofefi.feature_importance import feature_importance, fuzzy_interpretation
from biofefi.feature_importance.feature_importance_options import (
    FeatureImportanceOptions,
)
from biofefi.feature_importance.fuzzy_options import FuzzyOptions
from biofefi.machine_learning import train
from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.ml_options import MLOptions
from biofefi.options.enums import (
    ConfigStateKeys,
    ExecutionStateKeys,
    ProblemTypes,
    PlotOptionKeys,
    ViewExperimentKeys,
)
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_plot_dir,
    fuzzy_plot_dir,
    uploaded_file_path,
    log_dir,
    ml_plot_dir,
    ml_model_dir,
)
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import save_upload, set_seed
import streamlit as st


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
            plot_axis_font_size=st.session_state[PlotOptionKeys.AxisFontSize],
            plot_axis_tick_size=st.session_state[PlotOptionKeys.AxisTickSize],
            plot_title_font_size=st.session_state[PlotOptionKeys.TitleFontSize],
            plot_colour_scheme=st.session_state[PlotOptionKeys.ColourScheme],
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

    ml_opt = MLOptions()
    ml_opt.initialize()
    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName].name,
        biofefi_experiments_base_dir()
        / st.session_state[ConfigStateKeys.ExperimentName],
    )
    ml_opt.parser.set_defaults(
        n_bootstraps=st.session_state[ConfigStateKeys.NumberOfBootstraps],
        save_actual_pred_plots=st.session_state[PlotOptionKeys.SavePlots],
        normalization=st.session_state[ConfigStateKeys.Normalization],
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
        data_path=path_to_data,
        data_split=st.session_state[ConfigStateKeys.DataSplit],
        model_types=st.session_state[ConfigStateKeys.ModelTypes],
        ml_plot_dir=ml_plot_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ConfigStateKeys.ExperimentName]
        ),
        ml_log_dir=log_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ConfigStateKeys.ExperimentName]
        )
        / "ml",
        problem_type=st.session_state.get(
            ConfigStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        random_state=st.session_state[ConfigStateKeys.RandomSeed],
        is_machine_learning=st.session_state[ConfigStateKeys.IsMachineLearning],
        save_models=st.session_state[ConfigStateKeys.SaveModels],
    )
    ml_opt = ml_opt.parse()

    return fuzzy_opt, fi_opt, ml_opt, st.session_state[ConfigStateKeys.ExperimentName]


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
    ml_logger_instance = Logger(
        log_dir(biofefi_experiments_base_dir() / experiment_name) / "ml"
    )
    ml_logger = ml_logger_instance.make_logger()

    data = DataBuilder(ml_opts, ml_logger).ingest()

    # Machine learning
    if ml_opts.is_machine_learning:
        trained_models = train.run(ml_opts, data, ml_logger)
        if ml_opts.save_models:
            for model_name in trained_models:
                for i, model in enumerate(trained_models[model_name]):
                    save_path = (
                        ml_model_dir(biofefi_experiments_base_dir() / experiment_name)
                        / f"{model_name}-{i}.pkl"
                    )
                    save_model(model, save_path)
    else:
        trained_models = load_models(
            ml_model_dir(biofefi_experiments_base_dir() / experiment_name)
        )

    close_logger(ml_logger_instance, ml_logger)

    # Feature importance
    if fi_opts.is_feature_importance:
        fi_logger_instance = Logger(
            log_dir(biofefi_experiments_base_dir() / experiment_name) / "fi"
        )
        fi_logger = fi_logger_instance.make_logger()
        gloabl_importance_results, local_importance_results, ensemble_results = (
            feature_importance.run(fi_opts, data, trained_models, fi_logger)
        )

        close_logger(fi_logger_instance, fi_logger)

        # Fuzzy interpretation
        if fuzzy_opts.fuzzy_feature_selection:
            fuzzy_logger_instance = Logger(
                log_dir(biofefi_experiments_base_dir() / experiment_name) / "fuzzy"
            )
            fuzzy_logger = fuzzy_logger_instance.make_logger()
            fuzzy_rules = fuzzy_interpretation.run(
                fuzzy_opts,
                ml_opts,
                data,
                trained_models,
                ensemble_results,
                fuzzy_logger,
            )

            close_logger(fuzzy_logger_instance, fuzzy_logger)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        p.terminate()


## Page contents
st.set_page_config(
    page_title="BioFEFI",
    page_icon=sidebar_logo(),
)
header_logo()
sidebar_logo()
with st.sidebar:
    st.header("Options")
    # st.checkbox("Feature Engineering", key=ConfigStateKeys.IsFeatureEngineering)

    # Machine Learning Options
    ml_options_box()

    # Feature Importance (+ Fuzzy) Options
    fi_options_box()

    # Global plot options
    plot_options_box()
    seed = st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )
data_upload_form()


# If the user has uploaded a file and pressed the run button, run the pipeline
if (
    uploaded_file := st.session_state.get(ConfigStateKeys.UploadedFileName)
) and st.session_state.get(ExecutionStateKeys.RunPipeline, False):
    experiment_name = st.session_state.get(ConfigStateKeys.ExperimentName)
    upload_path = uploaded_file_path(
        uploaded_file.name, biofefi_experiments_base_dir() / experiment_name
    )
    save_upload(upload_path, uploaded_file.read().decode("utf-8-sig"))
    if uploaded_models := st.session_state.get(ConfigStateKeys.UploadedModels):
        for m in uploaded_models:
            upload_path = (
                ml_model_dir(biofefi_experiments_base_dir() / experiment_name) / m.name
            )
            save_upload(upload_path, m.read(), "wb")
    config = build_configuration()
    process = Process(target=pipeline, args=config, daemon=True)
    process.start()
    cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
    with st.spinner("Running pipeline..."):
        # wait for the process to finish or be cancelled
        process.join()
    try:
        st.session_state[ConfigStateKeys.MLLogBox] = get_logs(
            log_dir(
                biofefi_experiments_base_dir()
                / st.session_state[ConfigStateKeys.ExperimentName]
            )
            / "ml"
        )
        st.session_state[ConfigStateKeys.FILogBox] = get_logs(
            log_dir(
                biofefi_experiments_base_dir()
                / st.session_state[ConfigStateKeys.ExperimentName]
            )
            / "fi"
        )
        st.session_state[ConfigStateKeys.FuzzyLogBox] = get_logs(
            log_dir(
                biofefi_experiments_base_dir()
                / st.session_state[ConfigStateKeys.ExperimentName]
            )
            / "fuzzy"
        )
        log_box(box_title="Machine Learning Logs", key=ConfigStateKeys.MLLogBox)
        log_box(box_title="Feature Importance Logs", key=ConfigStateKeys.FILogBox)
        log_box(box_title="Fuzzy FI Logs", key=ConfigStateKeys.FuzzyLogBox)
    except NotADirectoryError:
        pass
    ml_plots = ml_plot_dir(biofefi_experiments_base_dir() / experiment_name)
    if ml_plots.exists():
        plot_box(ml_plots, "Machine learning plots")
    fi_plots = fi_plot_dir(biofefi_experiments_base_dir() / experiment_name)
    if fi_plots.exists():
        plot_box(fi_plots, "Feature importance plots")
    fuzzy_plots = fuzzy_plot_dir(biofefi_experiments_base_dir() / experiment_name)
    if fuzzy_plots.exists():
        plot_box(fuzzy_plots, "Fuzzy plots")
