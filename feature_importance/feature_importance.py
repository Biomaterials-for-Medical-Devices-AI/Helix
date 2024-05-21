import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_importance.interpreter import Interpreter
from feature_importance.fuzzy import Fuzzy


def run(fi_opt, data, models, logger):


    
    # Interpret the model results
    interpreter = Interpreter(fi_opt,logger=logger)
    fuzzy = Fuzzy(fi_opt, logger=logger)
    gloabl_importance_results, ensemble_results = interpreter.interpret(models, data)
    # ensemble_results = {'Majority Vote': pd.DataFrame()}
    # data = {0: [1.000000, 0.035145, 0.169412, 0.221469, 0.037242, 0.125516, 0.025160,  0.198182, 0.023214, 0.056053,
    #               0.036624, 0.033836, 0.094505, 0.040900, 0.071979, 0.075352, 0.001124, 0.000000]}
    # index = ["Average Feature Area_small",    "Average Feature Area_mod",    "Average Feature Area_large",    "Maximum Inscribed Circle Radius_small",
    #              "Maximum Inscribed Circle Radius_mod",    "Maximum Inscribed Circle Radius_large",    "Variance Feature Orientation_small",
    #                 "Variance Feature Orientation_mod",    "Variance Feature Orientation_large",    "Std Dev Inscribed Circle Radius_small",
    #                 "Std Dev Inscribed Circle Radius_mod",    "Std Dev Inscribed Circle Radius_large",    "Maximum Feature Area_small",
    #                 "Maximum Feature Area_mod",    "Maximum Feature Area_large",    "Feature Unit Cell Size_small",    "Feature Unit Cell Size_mod",
    #                 "Feature Unit Cell Size_large"]
    # ensemble_results['Majority Vote'] = pd.DataFrame(data, index=index)

    X_train, X_test = fuzzy.interpret(models, ensemble_results, data)
    print(X_train)
    print(X_test.shape)


    #return  gloabl_importance_results, ensemble_results, local_importance_results
    return gloabl_importance_results, ensemble_results