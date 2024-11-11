from argparse import Namespace
from multiprocessing import Process
import streamlit as st
from biofefi.components.experiments import experiment_selector
from biofefi.components.forms import ml_options_form
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.plots import plot_box
from biofefi.machine_learning import train
from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.ml_options import MLOptions
from biofefi.options.choices import NORMALISATIONS
from biofefi.options.enums import (
    ConfigStateKeys,
    PlotOptionKeys,
    ProblemTypes,
    ViewExperimentKeys,
)
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    log_dir,
    ml_model_dir,
    ml_plot_dir,
    uploaded_file_path,
)
from biofefi.services.experiments import get_experiments
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import save_model
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import cancel_pipeline, save_upload, set_seed


def build_configuration() -> tuple[Namespace, str]:
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[Namespace, str]: The configuration for the ML pipeline
        and the experiment name.
    """

    ml_opt = MLOptions()
    ml_opt.initialize()
    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName].name,
        biofefi_experiments_base_dir()
        / st.session_state[ViewExperimentKeys.ExperimentName],
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
        ml_log_dir=ml_plot_dir(
            biofefi_experiments_base_dir()
            / st.session_state[ConfigStateKeys.ExperimentName]
        ),
        problem_type=st.session_state.get(
            ConfigStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        random_state=st.session_state[ConfigStateKeys.RandomSeed],
        is_machine_learning=True,
        save_models=st.session_state[ConfigStateKeys.SaveModels],
    )
    ml_opt = ml_opt.parse()

    return ml_opt, st.session_state[ConfigStateKeys.ExperimentName]


def pipeline(ml_opts: Namespace, experiment_name: str):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        ml_opts (Namespace): Options for machine learning.
        experiment_name (str): The name of the experiment.
    """
    seed = ml_opts.random_state
    set_seed(seed)
    logger_instance = Logger(log_dir(biofefi_experiments_base_dir() / experiment_name))
    logger = logger_instance.make_logger()

    data = DataBuilder(ml_opts, logger).ingest()

    # Machine learning
    if ml_opts.is_machine_learning:
        trained_models = train.run(ml_opts, data, logger)
        if ml_opts.save_models:
            for model_name in trained_models:
                for i, model in enumerate(trained_models[model_name]):
                    save_path = (
                        ml_model_dir(biofefi_experiments_base_dir() / experiment_name)
                        / f"{model_name}-{i}.pkl"
                    )
                    save_model(model, save_path)

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
    This page is where you can train new machine learning models. First, you select an experiment
    to add your data. Then, you can give a name to your dependent variable. This will appear on your
    plots. Next, you choose a CSV containing your data and specify how you wish it to be standardised
    and spit into training and test data. After that, you select the type of problem you are trying
    to solve, followed by the models you wish to train - you may choose more than one. Finally,
    you choose which outputs to save and hit **"Run Training"**, and wait for the pipeline to finish.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
if experiment_name:
    st.session_state[ConfigStateKeys.ExperimentName] = experiment_name

    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )

    st.subheader("Data preparation")
    st.write(
        """
        Upload your data file as a CSV and then define how the data will be normalised and split between
        training and test data.
        """
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=ConfigStateKeys.Normalization,
    )

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
    st.number_input(
        "Number of bootstraps",
        min_value=1,
        value=10,
        key=ConfigStateKeys.NumberOfBootstraps,
    )
    seed = st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )

    ml_options_form()

    if st.button("Run Training", type="primary") and (
        uploaded_file := st.session_state.get(ConfigStateKeys.UploadedFileName)
    ):
        biofefi_base_dir = biofefi_experiments_base_dir()
        upload_path = uploaded_file_path(
            uploaded_file.name, biofefi_base_dir / experiment_name
        )
        save_upload(upload_path, uploaded_file.read().decode("utf-8-sig"))
        config = build_configuration()
        process = Process(target=pipeline, args=config, daemon=True)
        process.start()
        cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
        with st.spinner("Model training in progress. Check the logs for progress."):
            # wait for the process to finish or be cancelled
            process.join()
        try:
            st.session_state[ConfigStateKeys.LogBox] = get_logs(
                log_dir(biofefi_experiments_base_dir() / experiment_name)
            )
            log_box()
        except NotADirectoryError:
            pass
        ml_plots = ml_plot_dir(biofefi_experiments_base_dir() / experiment_name)
        if ml_plots.exists():
            plot_box(ml_plots, "Machine learning plots")
