import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_importance.interpreter import Interpreter


def run(opt, data, models, logger):


    #data = load_data(opt)
    #models = load_models(opt)
    # data = pd.read_csv('feature_importance/data/granular_surface_macrophage.csv') # Just for testing
    # # split data into X and y
    # # X starts from the second column
    # X = data.iloc[:, 1:-1]
    # y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test
    # Load saved models
    # xgb1 = load_model('xgb', 'feature_importance/models/') # Just for testing
    # xgb2 = load_model('xgb2', 'feature_importance/models/') # Just for testing
    # models = {'XG Boost1': xgb1, 'XG Boost2': xgb2} # Just for testing
    # initiate logger
    # logger = Logger(opt.log_dir, opt.experiment_name).make_logger()
    # Interpret the model results
    interpreter = Interpreter(opt,logger=logger)
    feature_importance_results, ensemble_results = interpreter.interpret(models, X_train, y_train)

    return feature_importance_results, ensemble_results