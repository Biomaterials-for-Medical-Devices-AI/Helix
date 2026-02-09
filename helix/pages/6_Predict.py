from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector, model_selector
from helix.components.images.logos import sidebar_logo
from helix.options.enums import (
    ExecutionStateKeys,
    Normalisations,
    PredictStateKeys,
    ProblemTypes,
)
from helix.options.file_paths import (
    data_options_path,
    data_preprocessing_options_path,
    execution_options_path,
    helix_experiments_base_dir,
    ml_model_dir,
)
from helix.options.preprocessing import PreprocessingOptions
from helix.services.configuration import (
    load_data_options,
    load_data_preprocessing_options,
    load_execution_options,
)
from helix.services.data import read_data, read_uploaded_data
from helix.services.experiments import get_experiments
from helix.services.ml_models import load_models
from helix.services.preprocessing import find_non_numeric_columns
from helix.utils.logging_utils import Logger

# Set page contents
st.set_page_config(
    page_title="Predict",
    page_icon=sidebar_logo(),
)


def preprocess_data(
    X: pd.DataFrame,
    predict_data: pd.DataFrame,
    options: PreprocessingOptions | None,
) -> pd.DataFrame:
    if options is None or not options.data_is_preprocessed:
        return predict_data

    columns_to_drop = find_non_numeric_columns(X)
    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)
        predict_data = predict_data.drop(columns=columns_to_drop)

    scaler = get_scaler(options.independent_variable_normalisation)
    if scaler is None:
        return predict_data

    scaler.fit(X)
    return pd.DataFrame(
        scaler.transform(predict_data),
        columns=predict_data.columns,
        index=predict_data.index,
    )


def get_scaler(normalisation_method):
    match normalisation_method:
        case Normalisations.NoNormalisation:
            return None
        case Normalisations.Standardisation:
            return StandardScaler()
        case Normalisations.MinMax:
            return MinMaxScaler()


def ensemble_predictions(
    predictions_df: pd.DataFrame,
    problem_type: ProblemTypes,
) -> pd.Series:
    if problem_type == ProblemTypes.Regression:
        return pd.Series(
            predictions_df.mean(axis=1),
            index=predictions_df.index,
            name="Mean Prediction",
        )

    if problem_type == ProblemTypes.Classification:
        values, _ = mode(predictions_df.values, axis=1, keepdims=False)
        return pd.Series(
            values,
            index=predictions_df.index,
            name="Majority Vote",
        )

    raise ValueError(f"Unsupported problem type: {problem_type}")


def get_model_predictions(models_to_use, trained_models, predict_data):
    predictions = {}

    for model_name, model in trained_models.items():
        if model_name not in models_to_use:
            continue

        display_name = model_name.split(".")[0].replace("-", " ").capitalize()
        predictions[display_name] = model.predict(predict_data)

    return pd.DataFrame(predictions)


def get_predictions(
    raw_data: pd.DataFrame,
    independent_variable_col_names: list,
    predict_data: pd.DataFrame,
    preprocessing_options: PreprocessingOptions | None,
    models: list,
    problem_type: ProblemTypes,
    id_column: str | None = None,
):

    X = raw_data[independent_variable_col_names]
    predict_data = predict_data[independent_variable_col_names]
    target_col = raw_data[id_column] if id_column else None

    predict_data = preprocess_data(X, predict_data, preprocessing_options)

    trained_models = load_models(
        ml_model_dir(
            helix_experiments_base_dir()
            / st.session_state[ExecutionStateKeys.ExperimentName]
        )
    )

    predictions_df = get_model_predictions(models, trained_models, predict_data)

    ensemble = ensemble_predictions(predictions_df, problem_type)

    result = pd.concat([predict_data, predictions_df, ensemble], axis=1)
    if target_col is not None:
        result = pd.concat([target_col, result], axis=1)

    st.dataframe(result)


st.header("Predict")
st.write(
    """
    This page allows the prediction of the target variable using the trained models.
    The idea is to use this page to predict the target variable on datasets where this value is unknown.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)
base_dir = helix_experiments_base_dir()
# Create the logger
logger_instance = Logger()
logger = logger_instance.make_logger()


if experiment_name:
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    display_options(base_dir / experiment_name)

    path_to_exec_opts = execution_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    exec_opt = load_execution_options(path_to_exec_opts)

    path_to_data_options = data_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    data_options = load_data_options(path_to_data_options)

    path_to_preprocessing_options = data_preprocessing_options_path(
        helix_experiments_base_dir() / experiment_name
    )
    if path_to_preprocessing_options.exists():
        preprocessing_options = load_data_preprocessing_options(
            path_to_preprocessing_options
        )
        is_preprocessed = preprocessing_options.data_is_preprocessed
        raw_data_path = data_options.data_path.replace("_preprocessed", "")
    else:
        preprocessing_options = None
        is_preprocessed = False
        raw_data_path = data_options.data_path
    # This is the data that the user provided initially.
    # Having this is useful as it is needed to fit the scalers if needed.
    raw_data = read_data(Path(raw_data_path), logger)

    # This is the data that was actually used for the training (after preprocessing)
    # This is only needed to get the names of the variables used for model training.
    if data_options.feature_columns is not None:
        independent_variables = data_options.feature_columns
    else:
        independent_variables = read_data(Path(data_options.data_path), None).columns[
            :-1
        ]

    st.write(
        "For this experiment, the following columns were used as independent variables:"
    )

    uploaded_file = st.file_uploader(
        "Upload the data ",
        type=["csv", "xlsx"],
        key=PredictStateKeys.PredictFile,
        help="Updload a CSV or Excel (.xslx) file containing your data.",
    )

    if uploaded_file:
        predict_data = read_uploaded_data(uploaded_file)
        st.write("#### View the provided data")
        st.dataframe(predict_data)

    model_dir = ml_model_dir(base_dir / experiment_name)
    if model_dir.exists():
        model_choices = list(
            filter(lambda x: x.endswith(".pkl"), [x.name for x in model_dir.iterdir()])
        )
    else:
        model_choices = []

    if len(model_choices) == 0:
        st.info(
            "You don't have any trained models in this experiment. "
            "Go to **Train Models** to create some models to evaulate."
        )

    else:
        models = model_selector(
            options=model_choices,
            gui_text="Select the models to use for the predictions of your new dataset",
            placeholder="Models for predictions",
            key=PredictStateKeys.PredictModels,
        )

    if uploaded_file:

        missing_cols = []
        for col in independent_variables:
            if col not in predict_data.columns:
                missing_cols.append(col)
                st.error(f"The columns {col} was not found in the provided data.")

        if missing_cols:
            st.error(
                "Dependent variables necessary for the prediciton were not found. Please provide the data with all the variables that you provided in your original data."
            )

        predict_data = predict_data[independent_variables]
        st.success(
            "All the needed independent variables for the predictions were found successfully."
        )

    if st.button(
        "Predict",
        key=PredictStateKeys.PredictButton,
        disabled=not models or not uploaded_file,
    ):
        get_predictions(
            raw_data=raw_data,
            independent_variable_col_names=independent_variables,
            predict_data=predict_data,
            preprocessing_options=preprocessing_options,
            models=models,
            problem_type=exec_opt.problem_type,
            id_column=data_options.id_column,
        )
