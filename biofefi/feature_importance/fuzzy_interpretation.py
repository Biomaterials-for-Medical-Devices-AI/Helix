import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biofefi.feature_importance.fuzzy import Fuzzy


def run(fuzzy_opt, ml_opt, data, models, ensemble_results, logger):

    # Interpret the feature synergy importance
    fuzzy = Fuzzy(opt=fuzzy_opt, ml_opt=ml_opt, logger=logger)
    fuzzy_rules = fuzzy.interpret(models, ensemble_results, data)

    return fuzzy_rules
