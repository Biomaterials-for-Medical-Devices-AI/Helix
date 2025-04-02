from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator
from helix.feature_importance.interpreter import FeatureImportanceEstimator
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.plotting import PlottingOptions
from helix.utils.logging_utils import Logger
from helix.services.feature_importance.global_methods import (
    calculate_global_shap_values,
    calculate_permutation_importance,
)
from helix.services.feature_importance.local_methods import (
    calculate_local_shap_values,
    calculate_lime_values,
)


def run(
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data,
    models,
    logger,
):

    # Interpret the model results
    interpreter = FeatureImportanceEstimator(
        fi_opt=fi_opt, exec_opt=exec_opt, plot_opt=plot_opt, logger=logger
    )
    # TODO: Add indices to the dataframe results-> global + ensemble
    # TODO: Resolve majority vote issue
    gloabl_importance_results, local_importance_results, ensemble_results = (
        interpreter.interpret(models, data)
    )

    return gloabl_importance_results, local_importance_results, ensemble_results
