from multiprocessing import Process
from pathlib import Path

import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.forms import ml_options_form
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.plots import display_metrics_table, plot_box
from biofefi.machine_learning import train
from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import (
    ExecutionStateKeys,
    MachineLearningStateKeys,
    PlotOptionKeys,
)
from biofefi.options.execution import ExecutionOptions
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    execution_options_path,
    log_dir,
    ml_metrics_path,
    ml_model_dir,
    ml_options_path,
    ml_plot_dir,
    plot_options_path,
)
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.configuration import (
    load_execution_options,
    load_plot_options,
    save_options,
)
from biofefi.services.experiments import get_experiments
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import models_exist, save_model, save_models_metrics
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import cancel_pipeline, delete_directory, set_seed


def build_configuration() -> (
    tuple[MachineLearningOptions, ExecutionOptions, PlottingOptions, str]
):
    """Build the configuration options to run the Machine Learning pipeline.

    Returns:
        tuple[MachineLearningOptions, ExecutionOptions, PlottingOptions, str]:
        The machine learning options, general execution options, plotting options,
        experiment name
    """

    experiment_name = st.session_state[ExecutionStateKeys.ExperimentName]

    path_to_plot_opts = plot_options_path(
        biofefi_experiments_base_dir() / experiment_name
    )
    plot_opt = load_plot_options(path_to_plot_opts)

    path_to_exec_opts = execution_options_path(
        biofefi_experiments_base_dir() / experiment_name
    )
    exec_opt = load_execution_options(path_to_exec_opts)
    ml_opt = MachineLearningOptions(
        save_actual_pred_plots=st.session_state[PlotOptionKeys.SavePlots],
        model_types=st.session_state[MachineLearningStateKeys.ModelTypes],
        ml_plot_dir=str(ml_plot_dir(biofefi_experiments_base_dir() / experiment_name)),
        ml_log_dir=str(
            log_dir(biofefi_experiments_base_dir() / experiment_name) / "ml"
        ),
        save_models=st.session_state[MachineLearningStateKeys.SaveModels],
    )

    return ml_opt, exec_opt, plot_opt, experiment_name


def pipeline(
    ml_opts: MachineLearningOptions,
    exec_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    experiment_name: str,
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        ml_opts (MachineLearningOptions): Options for machine learning.
        exec_opts (ExecutionOptions): General execution options.
        plotting_opts (PlottingOptions): Options for plotting.
        experiment_name (str): The name of the experiment.
    """
    seed = exec_opts.random_state
    set_seed(seed)
    logger_instance = Logger(Path(ml_opts.ml_log_dir))
    logger = logger_instance.make_logger()

    data = DataBuilder(
        data_path=exec_opts.data_path,
        random_state=exec_opts.random_state,
        normalization=exec_opts.normalization,
        n_bootstraps=exec_opts.n_bootstraps,
        logger=logger,
        data_split=exec_opts.data_split,
        problem_type=exec_opts.problem_type,
    ).ingest()

    # Machine learning
    trained_models, metrics_stats = train.run(
        ml_opts=ml_opts,
        exec_opts=exec_opts,
        plot_opts=plotting_opts,
        data=data,
        logger=logger,
    )
    if ml_opts.save_models:
        for model_name in trained_models:
            for i, model in enumerate(trained_models[model_name]):
                save_path = (
                    ml_model_dir(biofefi_experiments_base_dir() / experiment_name)
                    / f"{model_name}-{i}.pkl"
                )
                save_model(model, save_path)

    save_models_metrics(
        metrics_stats,
        ml_metrics_path(biofefi_experiments_base_dir() / experiment_name),
    )
    # Close the logger
    close_logger(logger_instance, logger)


st.set_page_config(
    page_title="Train Models",
    page_icon=sidebar_logo(),
)
sidebar_logo()

st.header("Train Models")
st.write(
    """
    This page is where you can train new machine learning models.
    First, you select an experiment to add your data.
    Then, you can give a name to your dependent variable. This will appear on your
    plots.
    Next, you choose a CSV containing your data and specify how you wish it
    to be standardised and spit into training and test data.
    After that, you select the type of problem you are trying to solve,
    followed by the models you wish to train - you may choose more than one.
    Finally, you choose which outputs to save and hit **"Run Training"**,
    and wait for the pipeline to finish.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
if experiment_name:
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name
    biofefi_base_dir = biofefi_experiments_base_dir()
    experiment_dir = biofefi_base_dir / experiment_name
    path_to_exec_opts = execution_options_path(experiment_dir)
    exec_opt = load_execution_options(path_to_exec_opts)

    if models_exist(
        ml_model_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ExecutionStateKeys.ExperimentName]
        )
    ):
        st.warning("⚠️ You have trained models in this experiment.")
        st.checkbox(
            "Would you like to rerun the experiments? This will overwrite the existing models.",
            value=True,
            key=MachineLearningStateKeys.RerunML,
        )
    else:
        st.session_state[MachineLearningStateKeys.RerunML] = True

    if st.session_state[MachineLearningStateKeys.RerunML]:
        ml_options_form()

    if st.session_state[MachineLearningStateKeys.RerunML] and (
        st.button("Run Training", type="primary")
    ):

        if experiment_dir.exists():
            delete_directory(ml_model_dir(experiment_dir))
        if experiment_dir.exists():
            delete_directory(ml_plot_dir(experiment_dir))

        config = build_configuration()
        save_options(ml_options_path(experiment_dir), config[0])
        process = Process(target=pipeline, args=config, daemon=True)
        process.start()
        cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
        with st.spinner("Model training in progress. Check the logs for progress."):
            # wait for the process to finish or be cancelled
            process.join()
        try:
            st.session_state[MachineLearningStateKeys.MLLogBox] = get_logs(
                log_dir(experiment_dir) / "ml"
            )
            log_box(
                box_title="Machine Learning Logs", key=MachineLearningStateKeys.MLLogBox
            )
        except NotADirectoryError:
            pass
        metrics = ml_metrics_path(experiment_dir)
        if metrics.exists():
            display_metrics_table(metrics)
        ml_plots = ml_plot_dir(experiment_dir)
        if ml_plots.exists():
            plot_box(ml_plots, "Machine learning plots")

    elif not st.session_state[MachineLearningStateKeys.RerunML]:
        st.info(
            "You have chosen not to rerun the machine learning experiments. "
            "You can proceed to feature importance analysis."
        )
