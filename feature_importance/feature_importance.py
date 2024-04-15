import os
import sys
from call_methods import load_data, load_models
from fi_methods import calculate_permutation_importance, calculate_shap_values


from fi_options import FIOptions

def run() -> None:

    opt = FIOptions().parse()
    #data = load_data(opt)
    #models = load_models(opt)
    data = 'data' # Just for testing
    models = {'Random Forest': 'rf', 'XG Boost': 'xgb'} # Just for testing
    fi_results = {}

    for model_type, model in models.items():
        # condition when all methods are False
        if not any(opt.fi_methods.values()):
            print("No feature importance methods selected")
            break

        print(f"Calculating feature importance for {model_type}")
        fi_results[model_type] = {}        

        # Run methods with TRUE values in the dictionary of feature importance methods
        if opt.fi_methods['Permutation Importance']:
            # Run Permutation Importance
            permutation_importance = calculate_permutation_importance(model, data)
            fi_results[model_type]['Permutation Importance'] = permutation_importance
        
        if opt.fi_methods['SHAP']:
            # Run SHAP
            shap_values = calculate_shap_values(model, data)
            fi_results[model_type]['SHAP'] = shap_values        
        




if __name__ == "__main__":
    run()