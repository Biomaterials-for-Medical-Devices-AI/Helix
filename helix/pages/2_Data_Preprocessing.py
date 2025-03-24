from pathlib import Path

import streamlit as st

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector
from helix.components.forms import preprocessing_opts_form
from helix.components.images.logos import sidebar_logo
from helix.components.preprocessing import original_view, preprocessed_view
from helix.options.enums import DataPreprocessingStateKeys, ExecutionStateKeys
from helix.options.file_paths import (
    data_options_path,
    data_preprocessing_options_path,
    helix_experiments_base_dir,
    plot_options_path,
    preprocessed_data_path,
)
from helix.options.preprocessing import PreprocessingOptions
from helix.services.configuration import (
    load_data_options,
    load_data_preprocessing_options,
    load_plot_options,
    save_options,
)
from helix.services.data import read_data, save_data
from helix.services.experiments import get_experiments
from helix.services.preprocessing import find_non_numeric_columns, run_preprocessing
from helix.utils.logging_utils import Logger, close_logger


def build_config() -> PreprocessingOptions:
    """
    Build the configuration object for preprocessing.
    """

    preprocessing_options = PreprocessingOptions(
        feature_selection_methods={
            DataPreprocessingStateKeys.VarianceThreshold: st.session_state[
                DataPreprocessingStateKeys.VarianceThreshold
            ],
            DataPreprocessingStateKeys.CorrelationThreshold: st.session_state[
                DataPreprocessingStateKeys.CorrelationThreshold
            ],
            DataPreprocessingStateKeys.LassoFeatureSelection: st.session_state[
                DataPreprocessingStateKeys.LassoFeatureSelection
            ],
        },
        variance_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdVariance
        ],
        correlation_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdCorrelation
        ],
        lasso_regularisation_term=st.session_state[
            DataPreprocessingStateKeys.RegularisationTerm
        ],
        independent_variable_normalisation=st.session_state[
            DataPreprocessingStateKeys.IndependentNormalisation
        ].lower(),
        dependent_variable_transformation=st.session_state[
            DataPreprocessingStateKeys.DependentNormalisation
        ].lower(),
    )
    return preprocessing_options


st.set_page_config(
    page_title="Data Preprocessing",
    page_icon=sidebar_logo(),
)

sidebar_logo()

st.header("Data Preprocessing")
st.write(
    """
    Here you can make changes to your data before running machine learning models. This includes feature selection and scalling of variables.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = helix_experiments_base_dir()

if experiment_name:
    logger_instance = Logger()
    logger = logger_instance.make_logger()

    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    display_options(biofefi_base_dir / experiment_name)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)
    data_opts = load_data_options(path_to_data_opts)

    path_to_preproc_opts = data_preprocessing_options_path(
        biofefi_base_dir / experiment_name
    )

    data_is_preprocessed = False
    if path_to_preproc_opts.exists():
        preproc_opts = load_data_preprocessing_options(path_to_preproc_opts)
        data_is_preprocessed = preproc_opts.data_is_preprocessed

    # Check if the user has already preprocessed their data
    if data_is_preprocessed:
        st.warning("Your data are already preprocessed. Would you like to start again?")
        preproc_again = st.checkbox("Redo preprocessing", value=False)
    else:
        # allow the user to perform preprocessing if the data are unprocessed
        preproc_again = True

    if not preproc_again:
        try:
            data = read_data(Path(data_opts.data_path), logger)
            preprocessed_view(data)
        except Exception:
            st.error("Unable to read data.", icon="🔥")
        finally:
            close_logger(logger_instance, logger)

    else:
        # remove preprocessed suffix to point to original data file
        data_opts.data_path = data_opts.data_path.replace("_preprocessed", "")

        try:
            data = read_data(Path(data_opts.data_path), logger)
            non_numeric = find_non_numeric_columns(data.iloc[:, :-1])

            if non_numeric:
                st.warning(
                    f"The following columns contain non-numeric values: {', '.join(non_numeric)}. These will be eliminated."
                )
            else:
                st.success("All the independent variable columns are numeric.")

            non_numeric_y = find_non_numeric_columns(data.iloc[:, -1])

            if non_numeric_y:
                st.warning(
                    "The dependent variable contains non-numeric values. This will be transformed to allow training."
                )

            plot_opt = load_plot_options(path_to_plot_opts)

            st.header(
                f"Original Data ({data.shape[1]} columns, including dependent variable)"
            )
            original_view(data)

            preprocessing_opts_form(data)

            if st.button("Run Data Preprocessing", type="primary"):

                config = build_config()

                processed_data = run_preprocessing(
                    data,
                    biofefi_base_dir / experiment_name,
                    config,
                )

                path_to_preprocessed_data = preprocessed_data_path(
                    Path(data_opts.data_path).name,
                    biofefi_base_dir / experiment_name,
                )

                save_data(path_to_preprocessed_data, processed_data, logger)

                # Update data opts to point to the pre-processed data
                data_opts.data_path = str(path_to_preprocessed_data)
                save_options(path_to_data_opts, data_opts)

                # Update config to show preprocessing is complete
                config.data_is_preprocessed = True
                save_options(path_to_preproc_opts, config)

                st.success("Data Preprocessing Complete")
                st.header(f"Preprocessed Data ({processed_data.shape[1]} columns)")
                preprocessed_view(processed_data)
        except Exception:
            st.error("Unable to read data.", icon="🔥")
        finally:
            close_logger(logger_instance, logger)
