import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_importance.interpreter import Interpreter
from feature_importance.fuzzy import Fuzzy


def run(fi_opt, data, models, logger):

    # Interpret the model results
    interpreter = Interpreter(fi_opt, logger=logger)
    # TODO: Add indices to the dataframe results-> global + ensemble
    # TODO: Resolve majority vote issue
    gloabl_importance_results, local_importance_results, ensemble_results = (
        interpreter.interpret(models, data)
    )

    return gloabl_importance_results, local_importance_results, ensemble_results
