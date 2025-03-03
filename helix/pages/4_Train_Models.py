from multiprocessing import Process
from pathlib import Path

import pandas as pd
import streamlit as st

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector
from helix.components.forms import ml_options_form
from helix.components.images.logos import sidebar_logo
from helix.components.logs import log_box
from helix.components.plots import display_metrics_table, display_predictions, plot_box
from helix.machine_learning import train
from helix.options.data import DataOptions
from helix.options.enums import (
    DataSplitMethods,
    ExecutionStateKeys,
    MachineLearningStateKeys,
    PlotOptionKeys,
)
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    log_dir,
    ml_metrics_path,
    ml_model_dir,
    ml_options_path,
    ml_plot_dir,
    ml_predictions_path,
    plot_options_path,
)
from helix.options.ml import MachineLearningOptions
from helix.options.plotting import PlottingOptions
from helix.services.configuration import (
    load_data_options,
    load_execution_options,
    load_plot_options,
    save_options,
)
from helix.services.data import DataBuilder, TabularData, ingest_data
from helix.services.experiments import get_experiments
from helix.services.logs import get_logs
from helix.services.ml_models import (
    models_exist,
    save_model,
    save_model_predictions,
    save_models_metrics,
)
from helix.utils.logging_utils import Logger, close_logger
from helix.utils.utils import cancel_pipeline, delete_directory, set_seed


def build_configuration() -> (
    tuple[MachineLearningOptions, ExecutionOptions, PlottingOptions, DataOptions, str]
):
    """Build the configuration options to run the Machine Learning pipeline.

    Returns:
        tuple[MachineLearningOptions, ExecutionOptions, PlottingOptions, DataOptions, str]:
        The machine learning options, general execution options, plotting options,
        data options, experiment name.
    """

    experiment_name = st.session_state[ExecutionStateKeys.ExperimentName]

    path_to_plot_opts = plot_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    plot_opt = load_plot_options(path_to_plot_opts)

    path_to_exec_opts = execution_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    path_to_data_opts = data_options_path(
        helix_experiments_base_dir() / experiment_name
    )

    exec_opt = load_execution_options(path_to_exec_opts)
    ml_opt = MachineLearningOptions(
        save_actual_pred_plots=st.session_state[PlotOptionKeys.SavePlots],
        model_types=st.session_state[MachineLearningStateKeys.ModelTypes],
        ml_plot_dir=str(ml_plot_dir(helix_experiments_base_dir() / experiment_name)),
        ml_log_dir=str(log_dir(helix_experiments_base_dir() / experiment_name) / "ml"),
        save_models=st.session_state[MachineLearningStateKeys.SaveModels],
        use_hyperparam_search=st.session_state.get(
            ExecutionStateKeys.UseHyperParamSearch, True
        ),
    )
    data_opts = load_data_options(path_to_data_opts)
    # update data opts
    data_opts.data_split = st.session_state.get(ExecutionStateKeys.DataSplit)
    save_options(path_to_data_opts, data_opts)

    return ml_opt, exec_opt, plot_opt, data_opts, experiment_name


def pipeline(
    ml_opts: MachineLearningOptions,
    exec_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
    data_opts: DataOptions,
    experiment_name: str,
    data: TabularData,
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        ml_opts (MachineLearningOptions): Options for machine learning.
        exec_opts (ExecutionOptions): General execution options.
        plotting_opts (PlottingOptions): Options for plotting.
        experiment_name (str): The name of the experiment.
        data (TabularData): The data that will be used in the pipeline.
    """
    seed = exec_opts.random_state
    set_seed(seed)
    logger_instance = Logger(Path(ml_opts.ml_log_dir))
    logger = logger_instance.make_logger()

    # Machine learning
    trained_models, metrics_stats = train.run(
        ml_opts=ml_opts,
        data_opts=data_opts,
        plot_opts=plotting_opts,
        data=data,
        exec_opts=exec_opts,
        logger=logger,
    )
    if ml_opts.save_models:
        predictions = pd.DataFrame(
            columns=["Y True", "Y Prediction", "Model Name", "Set", "Bootstrap"]
        )

        for model_name in trained_models:

            for i, model in enumerate(trained_models[model_name]):
                save_path = (
                    ml_model_dir(helix_experiments_base_dir() / experiment_name)
                    / f"{model_name}-{i}.pkl"
                )
                save_model(model, save_path)

                predictions_train = model.predict(data.X_train[i])
                predictions_train = {
                    "Y True": data.y_train[i],
                    "Y Prediction": predictions_train,
                    "Model Name": model_name,
                    "Set": "Train",
                    "Bootstrap": i,
                }
                df_train = pd.DataFrame(predictions_train)
                predictions_test = model.predict(data.X_test[i])
                predictions_test = {
                    "Y True": data.y_test[i],
                    "Y Prediction": predictions_test,
                    "Model Name": model_name,
                    "Set": "Test",
                    "Bootstrap": i,
                }
                df_test = pd.DataFrame(predictions_test)
                predictions = pd.concat(
                    [predictions, df_train, df_test], ignore_index=True
                )
        st.session_state[MachineLearningStateKeys.Predictions] = predictions

    if ml_opts.use_hyperparam_search:
        predictions = predictions[["Y True", "Y Prediction", "Model Name", "Set"]]
    elif data_opts.data_split.method == DataSplitMethods.KFold:
        predictions = predictions.rename(columns={"Bootstrap": "Fold"})

    save_models_metrics(
        metrics_stats,
        ml_metrics_path(helix_experiments_base_dir() / experiment_name),
    )
    save_model_predictions(
        predictions,
        ml_predictions_path(helix_experiments_base_dir() / experiment_name),
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
    biofefi_base_dir = helix_experiments_base_dir()
    experiment_dir = biofefi_base_dir / experiment_name
    display_options(experiment_dir)
    path_to_exec_opts = execution_options_path(experiment_dir)
    exec_opt = load_execution_options(path_to_exec_opts)

    already_trained_models = models_exist(
        ml_model_dir(
            helix_experiments_base_dir()
            / st.session_state[ExecutionStateKeys.ExperimentName]
        )
    )
    if already_trained_models:
        st.warning("⚠️ You have trained models in this experiment.")
        st.checkbox(
            "Would you like to rerun the experiments? This will overwrite the existing models.",
            value=False,
            key=MachineLearningStateKeys.RerunML,
        )
    else:
        st.session_state[MachineLearningStateKeys.RerunML] = False

    if not already_trained_models or st.session_state[MachineLearningStateKeys.RerunML]:
        ml_options_form()
    else:
        st.info(
            "You have chosen not to rerun the machine learning experiments. "
            "You can proceed to feature importance analysis."
        )
        st.stop()

    if st.button("Run Training", type="primary"):

        if experiment_dir.exists():
            delete_directory(ml_model_dir(experiment_dir))
        if experiment_dir.exists():
            delete_directory(ml_plot_dir(experiment_dir))

        ml_opts, exec_opts, plot_opts, data_opts, exp_name = build_configuration()
        # Create the logger
        logger_instance = Logger(Path(ml_opts.ml_log_dir))
        logger = logger_instance.make_logger()
        # Retrieve data from state or load it from disk
        data = ingest_data(exec_opts, data_opts, logger)
        # Save ML options
        save_options(ml_options_path(experiment_dir), ml_opts)
        process = Process(
            target=pipeline,
            args=(ml_opts, exec_opts, plot_opts, data_opts, exp_name, data),
            daemon=True,
        )
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
        display_metrics_table(metrics)

        if st.session_state.get(MachineLearningStateKeys.Predictions):
            display_predictions(
                st.session_state.get(MachineLearningStateKeys.Predictions)
            )
        else:
            predictions = ml_predictions_path(biofefi_base_dir / experiment_name)
            if predictions.exists():
                preds = pd.read_csv(predictions)
                display_predictions(preds)

        ml_plots = ml_plot_dir(experiment_dir)
        plot_box(ml_plots, "Machine learning plots")
