import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from helix.options.data import DataOptions, DataSplitOptions
from helix.options.enums import (
    DataSplitMethods,
    FeatureImportanceStateKeys,
    FeatureImportanceTypes,
    ViewExperimentKeys,
)
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_options_path,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_options_path,
    fuzzy_plot_dir,
    fuzzy_result_dir,
    helix_experiments_base_dir,
    log_dir,
    ml_model_dir,
)
from helix.services.configuration import save_options
from helix.services.ml_models import save_model
from tests.utils import get_element_by_key, get_element_by_label

from .fixtures import (
    classification_data_opts,
    classification_execution_opts,
    dummy_classification_data,
    new_classification_experiment,
    plotting_opts,
)


@pytest.fixture
def models_to_evaluate(
    new_classification_experiment: str,
    dummy_classification_data: np.ndarray,
    classification_execution_opts: ExecutionOptions,
    classification_data_opts: DataOptions,
):
    save_dir = ml_model_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    classification_data_opts.data_split = DataSplitOptions(
        n_bootstraps=3, method=DataSplitMethods.Holdout.capitalize()
    )
    # update the data options
    save_options(
        data_options_path(helix_experiments_base_dir() / new_classification_experiment),
        classification_data_opts,
    )
    X, y = dummy_classification_data[:, :-1], dummy_classification_data[:, -1]
    for i in range(classification_data_opts.data_split.n_bootstraps):
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=classification_data_opts.data_split.test_size,
            random_state=classification_execution_opts.random_state + i,
        )
        model = LogisticRegression()
        model.fit(X_train, y_train)
        save_model(model, save_dir / f"{model.__class__.__name__}-{i}.pkl")


def test_page_loads_without_exception():
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error


def test_page_can_find_experiment(new_classification_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert at.info  # the info saying there's no models should appear
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_can_find_models(
    new_classification_experiment: str, models_to_evaluate: None
):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert not at.info  # the info box saying there's no models shouldn't appear


# TODO: rename once global and ensemble nomenclature sorted
def test_ensemble_methods_disabled_without_global(
    new_classification_experiment: str, models_to_evaluate: None
):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    mean_checkbox = get_element_by_label(at, "checkbox", "Mean")
    maj_vote_checkbox = get_element_by_label(at, "checkbox", "Majority vote")

    # Assert
    assert not at.exception
    assert not at.error
    # check these methods are disabled
    assert mean_checkbox.disabled
    assert maj_vote_checkbox.disabled


# TODO: rename once global and ensemble nomenclature sorted
# def test_fuzzy_unavailable_without_ensemble_and_local_methods(
#     new_classification_experiment: str, models_to_evaluate: None
# ):
#     # Arrange
#     at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
#     at.run()

#     # Act
#     # Select the experiment
#     exp_selector = get_element_by_key(
#         at, "selectbox", ViewExperimentKeys.ExperimentName
#     )
#     exp_selector.select(new_classification_experiment).run()
#     # Select explain all models
#     all_model_toggle = get_element_by_key(
#         at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
#     )
#     all_model_toggle.set_value(True).run()
#     fuzzy_checkbox = get_element_by_label(
#         at, "checkbox", "Enable Fuzzy Feature Importance"
#     )

#     # Assert
#     assert not at.exception
#     assert not at.error
#     assert (
#         at.warning[1].value
#         == "You must configure both ensemble and local importance methods to use fuzzy feature selection."
#     )
#     # check these methods are disabled
#     assert fuzzy_checkbox.disabled


def test_permutation_importance(
    new_classification_experiment: str, models_to_evaluate: None
):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    fi_results = fi_result_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation importance")
    perm_imp_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert list(
        filter(
            lambda x: x.endswith("-bar.png")
            and x.startswith(FeatureImportanceTypes.PermutationImportance.value),
            map(str, fi_plots.iterdir()),
        )
    )
    assert list(
        filter(
            lambda x: x.endswith("all-folds-mean.png")
            and x.startswith(FeatureImportanceTypes.PermutationImportance.value),
            map(str, fi_plots.iterdir()),
        )
    )
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_global_shap(new_classification_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    # fi_results = fi_result_dir(helix_experiments_base_dir() / new_experiment)
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select global SHAP importance
    shap_checkbox = get_element_by_label(at, "checkbox", "SHAP")
    shap_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert (fi_plots / "SHAP-global-LogisticRegression-bar.png").exists()
    # TODO: check that global SHAP results should be getting saved, and fix
    # assert fi_results.exists()
    # assert list(
    #     filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    # )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_ensemble_mean(new_classification_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    fi_results = fi_result_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance; global method required for ensemble
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation Importance")
    perm_imp_checkbox.check().run()
    # Select ensemble mean
    ens_mean_checkbox = get_element_by_label(at, "checkbox", "Mean")
    ens_mean_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert (fi_plots / "Ensemble Mean-bar.png").exists()
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_ensemble_majority_vote(
    new_classification_experiment: str, models_to_evaluate: None
):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    fi_results = fi_result_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance; global method required for ensemble
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation Importance")
    perm_imp_checkbox.check().run()
    # Select ensemble mean
    ens_maj_vote_checkbox = get_element_by_label(at, "checkbox", "Majority Vote")
    ens_maj_vote_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert (fi_plots / "Ensemble Majority Vote-bar.png").exists()
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_local_lime(new_classification_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    fi_results = fi_result_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance; global method required for ensemble
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation Importance")
    perm_imp_checkbox.check().run()
    # Select local LIME
    local_lime_checkbox = get_element_by_label(at, "checkbox", "LIME")
    local_lime_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert (fi_plots / "LIME-LogisticRegression-violin.png").exists()
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_local_shap(new_classification_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
    fi_results = fi_result_dir(
        helix_experiments_base_dir() / new_classification_experiment
    )
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance; global method required for ensemble
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation Importance")
    perm_imp_checkbox.check().run()
    # Select local SHAP
    local_lime_checkbox = get_element_by_label(at, "checkbox", "Local SHAP")
    local_lime_checkbox.check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert (fi_plots / "SHAP-local-LogisticRegression-beeswarm.png").exists()
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
# def test_fuzzy_analysis(new_classification_experiment: str, models_to_evaluate: None):
#     # Arrange
#     fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_classification_experiment)
#     fi_results = fi_result_dir(
#         helix_experiments_base_dir() / new_classification_experiment
#     )
#     fuzzy_plots = fuzzy_plot_dir(
#         helix_experiments_base_dir() / new_classification_experiment
#     )
#     fuzzy_results = fuzzy_result_dir(
#         helix_experiments_base_dir() / new_classification_experiment
#     )
#     fuzzy_options = fuzzy_options_path(
#         helix_experiments_base_dir() / new_classification_experiment
#     )
#     at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=180.0)
#     at.run()

#     # Act
#     # Select the experiment
#     exp_selector = get_element_by_key(
#         at, "selectbox", ViewExperimentKeys.ExperimentName
#     )
#     exp_selector.select(new_classification_experiment).run()
#     # Select explain all models
#     all_model_toggle = get_element_by_key(
#         at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
#     )
#     all_model_toggle.set_value(True).run()
#     # Select permutation importance; global method required for ensemble
#     perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation Importance")
#     perm_imp_checkbox.check().run()
#     # Select ensemble mean; required for fuzzy
#     ens_mean_checkbox = get_element_by_label(at, "checkbox", "Mean")
#     ens_mean_checkbox.check().run()
#     # Select local SHAP; local also required for fuzzy
#     local_lime_checkbox = get_element_by_label(at, "checkbox", "Local SHAP")
#     local_lime_checkbox.check().run()
#     # Select fuzzy
#     fuzzy_checkbox = get_element_by_label(
#         at, "checkbox", "Enable Fuzzy Feature Importance"
#     )
#     fuzzy_checkbox.check().run()
#     # Select granualr analysis
#     granular_checkbox = get_element_by_label(at, "checkbox", "Granular features")
#     granular_checkbox.check().run()
#     # Leave additional configs as the defaults
#     # Leave save output toggles as true, the default
#     # Run
#     button = get_element_by_label(at, "button", "Run Feature Importance")
#     button.click().run()

#     # Assert
#     assert not at.exception
#     assert not at.error
#     assert fi_plots.exists()
#     assert (fi_plots / "SHAP-local-LogisticRegression-beeswarm.png").exists()
#     assert fi_results.exists()
#     assert list(
#         filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
#     )  # directory is not empty
#     assert list(
#         filter(lambda x: x.endswith(".png"), map(str, fuzzy_plots.iterdir()))
#     )  # directory is not empty
#     assert fuzzy_options.exists()
#     assert (fuzzy_results / "top contextual rules.csv").exists()


def test_page_makes_one_log_per_run(
    new_classification_experiment: str, models_to_evaluate: None
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_classification_experiment
    expected_fi_log_dir = log_dir(exp_dir) / "fi"
    expected_fuzzy_log_dir = log_dir(exp_dir) / "fi"
    expected_n_log_files = 1
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=180.0)
    at.run()

    # Act
    # Select the experiment
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(new_classification_experiment).run()
    # Select explain all models
    all_model_toggle = get_element_by_key(
        at, "toggle", FeatureImportanceStateKeys.ExplainAllModels
    )
    all_model_toggle.set_value(True).run()
    # Select permutation importance; global method required for ensemble
    at.checkbox[0].check().run()
    # Select ensemble mean; required for fuzzy
    at.checkbox[2].check().run()
    # Select local SHAP; local also required for fuzzy
    at.checkbox[5].check().run()
    # Select fuzzy
    at.checkbox[6].check().run()
    # Select granualr analysis
    at.checkbox[7].check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    button = get_element_by_label(at, "button", "Run Feature Importance")
    button.click().run()

    # log dir contents
    fi_log_dir_contents = list(
        filter(lambda x: x.endswith(".log"), map(str, expected_fi_log_dir.iterdir()))
    )
    fuzzy_log_dir_contents = list(
        filter(lambda x: x.endswith(".log"), map(str, expected_fuzzy_log_dir.iterdir()))
    )

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_fi_log_dir.exists()
    assert fi_log_dir_contents  # directory is not empty
    assert len(fi_log_dir_contents) == expected_n_log_files
    assert expected_fuzzy_log_dir.exists()
    assert fuzzy_log_dir_contents  # directory is not empty
    assert len(fuzzy_log_dir_contents) == expected_n_log_files
