import argparse

import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(model, X, y, opt: argparse.Namespace, logger):
    '''Calculate permutation importance for a given model and dataset
    Args:
        model: Model object
        X: Input features
        y: Target variable
        opt: Options
        logger: Logger
    Returns:
        permutation_importance: Permutation importance values
    '''

    logger.info(f"Calculating Permutation Importance..")
        
    # Use permutation importance in sklearn.inspection to calculate feature importance
    permutation_importance_results = permutation_importance(model, X=X, y=y, scoring=opt.permutation_importance_scoring,
                                                             n_repeats=opt.permutation_importance_repeat, 
                                                             random_state=opt.random_state)
    # Create a DataFrame with the results
    permutation_importance_df = pd.DataFrame(permutation_importance_results.importances_mean, 
                                              index=X.columns)
    

    # Return the DataFrame
    return permutation_importance_df


def calculate_shap_values(model, X, opt: argparse.Namespace, logger):
    '''Calculate SHAP values for a given model and dataset
    Args:
        model: Model object
        X: Dataset
        opt: Options
        logger: Logger
    Returns:
        shap_df: Average SHAP values
    '''
    logger.info(f"Calculating SHAP Importance..")

    if opt.shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else: 
        X_reduced = shap.utils.sample(X, int(X.shape[0]*opt.shap_reduce_data/100))
        explainer = shap.Explainer(model.predict, X_reduced)   
    
    shap_values = explainer(X)     
    
    # Calculate Average Importance + set column names as index
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame() 

    
    # Return the DataFrame
    return shap_df, shap_values
    