import streamlit as st
from helix.components.images.logos import sidebar_logo
from helix.services.experiments import (
    get_experiments,
)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector, model_selector

from helix.options.file_paths import (
    data_options_path,
    execution_options_path,
    data_preprocessing_options_path,
    helix_experiments_base_dir,
    ml_model_dir,
)
from scipy.stats import mode

from helix.services.configuration import (
    load_data_options,
    load_execution_options,
    load_data_preprocessing_options,
)
from helix.services.ml_models import load_models
from helix.options.preprocessing import PreprocessingOptions
from helix.services.preprocessing import find_non_numeric_columns
from helix.options.enums import (
    Normalisations,
    ExecutionStateKeys,
    ProblemTypes,
    PredictStateKeys,
)

# Set page contents
st.set_page_config(
    page_title="Predict",
    page_icon=sidebar_logo(),
)


def get_predictions(
    raw_data: pd.DataFrame,
    independent_variable_col_names: list,
    predict_data: pd.DataFrame,
    preprocessing_options: PreprocessingOptions,
    models: list,
    problem_type: ProblemTypes,
):

    X = raw_data[independent_variable_col_names]
    predict_data = predict_data[independent_variable_col_names]

    if preprocessing_options.data_is_preprocessed:

        columns_to_drop = find_non_numeric_columns(X)
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)

        normalisation_method = preprocessing_options.independent_variable_normalisation
        if normalisation_method == Normalisations.NoNormalisation:
            scaler = None
        elif normalisation_method == Normalisations.Standardisation:
            scaler = StandardScaler()
        elif normalisation_method == Normalisations.MinMax:
            scaler = MinMaxScaler()

        if scaler is not None:
            scaler.fit(X)
            predict_data = scaler.transform(predict_data)
            predict_data = pd.DataFrame(
                predict_data, columns=independent_variable_col_names
            )

    trained_models = load_models(
        ml_model_dir(
            helix_experiments_base_dir()
            / st.session_state[ExecutionStateKeys.ExperimentName]
        )
    )

    predictions_df = pd.DataFrame()

    for model_name, model in trained_models.items():
        if model_name in models:
            df_model_name = model_name.split(".")[0].replace("-", " ").capitalize()
            predictions_df[df_model_name] = model.predict(predict_data)

    if problem_type == ProblemTypes.Regression:
        ensemble_prediction = predictions_df.mean(axis=1)
        method = "Mean Prediction"

    elif problem_type == ProblemTypes.Classification:
        ensemble_prediction, _ = mode(predictions_df.values, axis=1, keepdims=False)
        method = "Majority Vote"

    ensemble_prediction = pd.Series(
        ensemble_prediction, index=predictions_df.index, name=method
    )

    predictions_df = pd.concat(
        [predict_data, predictions_df, ensemble_prediction], axis=1
    )

    st.dataframe(predictions_df)


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
    raw_data = pd.read_csv(raw_data_path)

    # This is the data that was actually used for the training (after preprocessing)
    # This is only needed to get the names of the variables used for model training.
    independent_variables = pd.read_csv(data_options.data_path).columns[:-1]
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
        if uploaded_file.name.endswith(".csv"):
            predict_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            predict_data = pd.read_excel(uploaded_file)
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
        )
