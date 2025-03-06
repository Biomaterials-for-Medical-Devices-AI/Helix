# 1. Fixture for models to perform feature importance
# 2. Test page loads without error
# 3. Test page finds experiment
# 4. Test what happens when no models to test
# 5. Test plots are produced
# 6. Test for results


from pickle import dump
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from helix.options.data import DataOptions, DataSplitOptions
from helix.options.enums import DataSplitMethods
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_options_path,
    fi_plot_dir,
    fi_result_dir,
    helix_experiments_base_dir,
    ml_model_dir,
)
from helix.services.configuration import save_options
from helix.services.ml_models import save_model

from .fixtures import (
    data_opts,
    dummy_data,
    execution_opts,
    new_experiment,
    plotting_opts,
)


@pytest.fixture
def models_to_evaluate(
    new_experiment: str,
    dummy_data: np.ndarray,
    execution_opts: ExecutionOptions,
    data_opts: DataOptions,
):
    save_dir = ml_model_dir(helix_experiments_base_dir() / new_experiment)
    data_opts.data_split = DataSplitOptions(
        n_bootstraps=3, method=DataSplitMethods.Holdout.capitalize()
    )
    # update the data options
    save_options(
        data_options_path(helix_experiments_base_dir() / new_experiment), data_opts
    )
    X, y = dummy_data[:, :-1], dummy_data[:, -1]
    for i in range(data_opts.data_split.n_bootstraps):
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=data_opts.data_split.test_size,
            random_state=execution_opts.random_state + i,
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


def test_page_can_find_experiment(new_experiment: str):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert at.info  # the info saying there's no models should appear
    with pytest.raises(ValueError):
        # check for error for non existent experiment
        at.selectbox[0].select("non-existent").run()


def test_page_can_find_models(new_experiment: str, models_to_evaluate: None):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert not at.info  # the info box saying there's no models shouldn't appear


# TODO: rename once global and ensemble nomenclature sorted
def test_ensemble_methods_disabled_without_global(
    new_experiment: str, models_to_evaluate: None
):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Select explain all models
    at.toggle[0].set_value(True).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert (
        at.warning[0].value
        == "You must configure at least one global feature importance method to perform ensemble methods."
    )
    # check these methods are disabled
    assert at.checkbox[2].disabled
    assert at.checkbox[3].disabled


# TODO: rename once global and ensemble nomenclature sorted
def test_fuzzy_unavailable_without_ensemble_and_local_methods(
    new_experiment: str, models_to_evaluate: None
):
    # Arrange
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Select explain all models
    at.toggle[0].set_value(True).run()

    # Assert
    assert not at.exception
    assert not at.error
    assert (
        at.warning[1].value
        == "You must configure both ensemble and local importance methods to use fuzzy feature selection."
    )
    # check these methods are disabled
    assert at.checkbox[6].disabled


def test_permutation_importance(new_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_experiment)
    fi_results = fi_result_dir(helix_experiments_base_dir() / new_experiment)
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Select explain all models
    at.toggle[0].set_value(True).run()
    # Select permutation importance
    at.checkbox[0].check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, fi_plots.iterdir()))
    )  # directory is not empty
    assert fi_results.exists()
    assert list(
        filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    )  # directory is not empty


# TODO: rename once global and ensemble nomenclature sorted
def test_global_shap(new_experiment: str, models_to_evaluate: None):
    # Arrange
    fi_plots = fi_plot_dir(helix_experiments_base_dir() / new_experiment)
    fi_results = fi_result_dir(helix_experiments_base_dir() / new_experiment)
    at = AppTest.from_file("helix/pages/5_Feature_Importance.py", default_timeout=60.0)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Select explain all models
    at.toggle[0].set_value(True).run()
    # Select global SHAP importance
    at.checkbox[1].check().run()
    # Leave additional configs as the defaults
    # Leave save output toggles as true, the default
    # Run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert fi_plots.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, fi_plots.iterdir()))
    )  # directory is not empty
    # TODO: check that global SHAP results should be getting saved, and fix
    # assert fi_results.exists()
    # assert list(
    #     filter(lambda x: x.endswith(".csv"), map(str, fi_results.iterdir()))
    # )  # directory is not empty
