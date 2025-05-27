import json
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
    ml_metrics_full_path,
    ml_metrics_mean_std_path,
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


@pytest.fixture
def mock_clf_metrics(
    new_classification_experiment: str,
):
    metrics = {
        "linear model": [
            {
                "accuracy": {
                    "train": {"value": 0.8967391304347826},
                    "test": {"value": 0.8859934853420195},
                },
                "f1_score": {
                    "train": {"value": 0.8781270044900578},
                    "test": {"value": 0.866581956797967},
                },
                "precision_score": {
                    "train": {"value": 0.8207434052757794},
                    "test": {"value": 0.8042452830188679},
                },
                "recall_score": {
                    "train": {"value": 0.9441379310344827},
                    "test": {"value": 0.9393939393939394},
                },
                "roc_auc_score": {
                    "train": {"value": 0.9660524199783517},
                    "test": {"value": 0.9631011977053033},
                },
            },
            {
                "accuracy": {
                    "train": {"value": 0.8616847826086956},
                    "test": {"value": 0.8523344191096635},
                },
                "f1_score": {
                    "train": {"value": 0.8443900947722409},
                    "test": {"value": 0.8325123152709359},
                },
                "precision_score": {
                    "train": {"value": 0.7583745194947831},
                    "test": {"value": 0.7527839643652561},
                },
                "recall_score": {
                    "train": {"value": 0.9524137931034483},
                    "test": {"value": 0.931129476584022},
                },
                "roc_auc_score": {
                    "train": {"value": 0.9631159734034329},
                    "test": {"value": 0.953972767755759},
                },
            },
            {
                "accuracy": {
                    "train": {"value": 0.8875},
                    "test": {"value": 0.8990228013029316},
                },
                "f1_score": {
                    "train": {"value": 0.8673076923076923},
                    "test": {"value": 0.8806161745827985},
                },
                "precision_score": {
                    "train": {"value": 0.8101796407185629},
                    "test": {"value": 0.8245192307692307},
                },
                "recall_score": {
                    "train": {"value": 0.9331034482758621},
                    "test": {"value": 0.9449035812672176},
                },
                "roc_auc_score": {
                    "train": {"value": 0.9637263027678986},
                    "test": {"value": 0.9651006645141543},
                },
            },
        ]
    }
    metrics_path = ml_metrics_full_path(
        helix_experiments_base_dir() / new_classification_experiment
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as mp:
        json.dump(metrics, mp)


@pytest.fixture
def mock_clf_metrics_mean_std(
    new_classification_experiment: str,
):
    metrics_mean_std = {
        "linear model": {
            "train": {
                "accuracy": {"mean": 0.8819746376811594, "std": 0.014834622721725895},
                "f1_score": {"mean": 0.8632749305233304, "std": 0.014065137634925323},
                "precision_score": {
                    "mean": 0.7964325218297085,
                    "std": 0.027254442288661425,
                },
                "recall_score": {
                    "mean": 0.9432183908045978,
                    "std": 0.007910184153602411,
                },
                "roc_auc_score": {
                    "mean": 0.9642982320498944,
                    "std": 0.0012651763907432546,
                },
            },
            "test": {
                "accuracy": {"mean": 0.8791169019182048, "std": 0.019670905887370485},
                "f1_score": {"mean": 0.8599034822172339, "std": 0.020198132541037736},
                "precision_score": {
                    "mean": 0.7938494927177849,
                    "std": 0.030194275986245824,
                },
                "recall_score": {
                    "mean": 0.9384756657483929,
                    "std": 0.005660618919163446,
                },
                "roc_auc_score": {
                    "mean": 0.9607248766584057,
                    "std": 0.004843738078166715,
                },
            },
        }
    }
    metrics_path = ml_metrics_mean_std_path(
        helix_experiments_base_dir() / new_classification_experiment
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as mp:
        json.dump(metrics_mean_std, mp)


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
    new_classification_experiment: str,
    models_to_evaluate: None,
    mock_clf_metrics: None,
    mock_clf_metrics_mean_std: None,
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
            and FeatureImportanceTypes.PermutationImportance.value in x,
            map(str, fi_plots.iterdir()),
        )
    )
    assert list(
        filter(
            lambda x: x.endswith("all-folds-mean.png")
            and FeatureImportanceTypes.PermutationImportance.value in x,
            map(str, fi_plots.iterdir()),
        )
    )
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_global_shap(
    new_classification_experiment: str,
    models_to_evaluate: None,
    mock_clf_metrics: None,
    mock_clf_metrics_mean_std: None,
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
    assert list(
        filter(
            lambda x: x.endswith("-bar.png") and FeatureImportanceTypes.SHAP.value in x,
            map(str, fi_plots.iterdir()),
        )
    )
    assert list(
        filter(
            lambda x: x.endswith("all-folds-mean.png")
            and FeatureImportanceTypes.SHAP.value in x,
            map(str, fi_plots.iterdir()),
        )
    )
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


def test_ensemble_mean(
    new_classification_experiment: str,
    models_to_evaluate: None,
    mock_clf_metrics: None,
    mock_clf_metrics_mean_std: None,
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
    perm_imp_checkbox = get_element_by_label(at, "checkbox", "Permutation importance")
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
    assert list(
        filter(
            lambda x: x.endswith(f"{FeatureImportanceTypes.Mean.value}.png"),
            map(str, fi_plots.iterdir()),
        )
    )
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
