import os
import sys


import pandas as pd
from call_methods import load_model
from feature_importance_methods import calculate_permutation_importance, calculate_shap_values
from options.feature_importance_options import FeatureImportanceOptions
from options.permutation_importance_options import PermutationImportanceOptions
from options.shap_options import SHAPOptions

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
            opt = PermutationImportanceOptions().parse()
            permutation_importance = calculate_permutation_importance(model, model_type, X, y, opt)
            feature_importance_results[model_type]['Permutation Importance'] = permutation_importance

        
        if opt.feature_importance_methods['SHAP']:
            # Run SHAP
            opt = SHAPOptions().parse()
            shap_values = calculate_shap_values(model,model_type, X,opt)
            feature_importance_results[model_type]['SHAP'] = shap_values
            


    # Check if fi_results is not empty and print the shape of the results
    if feature_importance_results:
        print(feature_importance_results['XG Boost1']['Permutation Importance'].shape) # Just for testing
        print(feature_importance_results['XG Boost2']['Permutation Importance'].shape) # Just for testing
        print(feature_importance_results['XG Boost1']['SHAP'].shape) # Just for testing
        print(feature_importance_results['XG Boost2']['SHAP'].shape) # Just for testing
    else:
        print("No feature importance results found")

           
        




if __name__ == "__main__":
    run()