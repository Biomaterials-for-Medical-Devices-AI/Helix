from pathlib import Path

import streamlit as st

from helix.components.configuration import display_options
from helix.components.experiments import experiment_selector
from helix.components.forms.forms_preprocessing import preprocessing_opts_form
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
    Here you can preprocess your data before running machine learning models. This includes feature selection and scalling of variables.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = helix_experiments_base_dir()


def validate_data(data, logger) -> tuple[list, bool]:
    """Validate data for preprocessing.

    Args:
        data: The input data to validate
        logger: Logger instance

    Returns:
        tuple containing list of non-numeric columns and whether y has non-numeric values
    """
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
            "The dependent variable contains non-numeric values. This will be transformed to allow for training."
        )

    return non_numeric, bool(non_numeric_y)


def run_preprocessing_pipeline(
    data,
    config,
    experiment_dir: Path,
    data_opts,
    path_to_data_opts,
    path_to_preproc_opts,
    logger,
) -> None:
    """Run the preprocessing pipeline and save results.

    Args:
        data: Input data to preprocess
        config: Preprocessing configuration
        experiment_dir: Path to experiment directory
        data_opts: Data options
        path_to_data_opts: Path to data options file
        path_to_preproc_opts: Path to preprocessing options file
        logger: Logger instance
    """
    processed_data = run_preprocessing(
        data,
        experiment_dir,
        config,
    )

    path_to_preprocessed_data = preprocessed_data_path(
        Path(data_opts.data_path).name,
        experiment_dir,
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


if experiment_name:
    logger_instance = Logger()
    logger = logger_instance.make_logger()

    try:
        st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name
        display_options(biofefi_base_dir / experiment_name)

        path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)
        path_to_data_opts = data_options_path(biofefi_base_dir / experiment_name)
        data_opts = load_data_options(path_to_data_opts)

        path_to_preproc_opts = data_preprocessing_options_path(
            biofefi_base_dir / experiment_name
        )

        # Check preprocessing status
        data_is_preprocessed = False
        if path_to_preproc_opts.exists():
            preproc_opts = load_data_preprocessing_options(path_to_preproc_opts)
            data_is_preprocessed = preproc_opts.data_is_preprocessed

        if data_is_preprocessed:
            st.warning(
                "Your data are already preprocessed. Would you like to start again?"
            )
            preproc_again = st.checkbox("Redo preprocessing", value=False)
        else:
            preproc_again = True

        if not preproc_again:
            data = read_data(Path(data_opts.data_path), logger)
            preprocessed_view(data)
        else:
            # remove preprocessed suffix to point to original data file
            data_opts.data_path = data_opts.data_path.replace("_preprocessed", "")
            data = read_data(Path(data_opts.data_path), logger)

            # Validate data
            validate_data(data, logger)
            plot_opt = load_plot_options(path_to_plot_opts)
            original_view(data)
            preprocessing_opts_form(data)

            if st.button("Run Data Preprocessing", type="primary"):
                config = build_config()
                run_preprocessing_pipeline(
                    data,
                    config,
                    biofefi_base_dir / experiment_name,
                    data_opts,
                    path_to_data_opts,
                    path_to_preproc_opts,
                    logger,
                )

    except ValueError as e:
        st.error(str(e), icon="🔥")
    except Exception:
        st.error("Something went wrong.", icon="🔥")
    finally:
        close_logger(logger_instance, logger)
