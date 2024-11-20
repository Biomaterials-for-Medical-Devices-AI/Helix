import os
import sys

from biofefi.options.execution import ExecutionOptions
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.plotting import PlottingOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biofefi.feature_importance.fuzzy import Fuzzy


def run(
    fuzzy_opt: FuzzyOptions,
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data,
    models,
    ensemble_results,
    logger,
):

    # Interpret the feature synergy importance
    fuzzy = Fuzzy(
        fuzzy_opt=fuzzy_opt,
        fi_opt=fi_opt,
        exec_opt=exec_opt,
        plot_opt=plot_opt,
        logger=logger,
    )
    fuzzy_rules = fuzzy.interpret(models, ensemble_results, data)

    return fuzzy_rules
