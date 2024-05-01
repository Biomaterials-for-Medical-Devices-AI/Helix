import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from feature_importance_options import FeatureImportanceOptions
from call_methods import load_model


def run() -> None:

    opt = FeatureImportanceOptions().parse()
    #data = load_data(opt)
    #models = load_models(opt)
    data = pd.read_csv('feature_importance/data/granular_surface_macrophage.csv') # Just for testing
    # split data into X and y
    # X starts from the second column
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    # Load saved models
    xgb1 = load_model('xgb', 'feature_importance/models/') # Just for testing
    xgb2 = load_model('xgb2', 'feature_importance/models/') # Just for testing
    models = {'XG Boost1': xgb1, 'XG Boost2': xgb2} # Just for testing
    # Interpret the model results
    from interpreter import Interpreter
    interpreter = Interpreter(opt)
    feature_importance_results, ensemble_results = interpreter.interpret(models, X, y)           
        




if __name__ == "__main__":
    run()