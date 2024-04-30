import os
import sys


import pandas as pd
from call_methods import load_model
from feature_importance_methods import calculate_permutation_importance, calculate_shap_values
from options.feature_importance_options import FeatureImportanceOptions
from ensemble_methods import calculate_ensemble_mean, calculate_ensemble_majorityvote


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(os.getcwd())

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
    feature_importance_results = {}

    for model_type, model in models.items():
        # condition when all methods are False
        if not any(opt.feature_importance_methods.values()):
            print("No feature importance methods selected")
            break

        print(f"Calculating feature importance for {model_type}...")
        feature_importance_results[model_type] = {}        

        # Run methods with TRUE values in the dictionary of feature importance methods
        if opt.feature_importance_methods['Permutation Importance']:
            # Run Permutation Importance
            permutation_importance = calculate_permutation_importance(model, model_type, X, y, opt)
            feature_importance_results[model_type]['Permutation Importance'] = permutation_importance

        
        if opt.feature_importance_methods['SHAP']:
            # Run SHAP
            shap_values = calculate_shap_values(model,model_type, X,opt)
            feature_importance_results[model_type]['SHAP'] = shap_values
            


    ######################### Run Ensemble Feature Importance Methods ######################### 

    # condition when all ensemble methods are False
    if not any(opt.feature_importance_ensemble.values()):
        print("No ensemble feature importance method selected")
    else:            
        ensemble_results = {}
        print("------ Ensemble of feature importance results ------")
        if opt.feature_importance_ensemble['Mean']:
            # Calculate mean of feature importance results            
            ensemble_results['Mean'] = calculate_ensemble_mean(feature_importance_results, opt)
        
        if opt.feature_importance_ensemble['Majority Vote']:
            # Calculate majority vote of feature importance results            
            ensemble_results['Majority Vote'] = calculate_ensemble_majorityvote(feature_importance_results, opt)
            


    print(ensemble_results['Mean']) # Just for testing

    print(ensemble_results['Majority Vote']) # Just for testing


           
        




if __name__ == "__main__":
    run()