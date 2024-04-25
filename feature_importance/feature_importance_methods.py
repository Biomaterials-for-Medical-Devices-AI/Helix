import argparse
import os
import pandas as pd
from sklearn.inspection import permutation_importance
from utils.utils import log_metrics
import shap
import matplotlib.pyplot as plt

def calculate_permutation_importance(model, model_type, X, y, opt: argparse.Namespace):
    '''Calculate permutation importance for a given model and dataset
    Args:
        model: Model object
        X: Input features
        y: Target variable
    Returns:
        permutation_importance: Permutation importance values
    '''
    print("Calculating Permutation Importance...")

        
    # Use permutation importance in sklearn.inspection to calculate feature importance
    permutation_importance_results = permutation_importance(model, X=X, y=y, scoring=opt.permutation_importance_scoring,
                                                             n_repeats=opt.permutation_importance_repeat, 
                                                             random_state=opt.random_state)
    # Create a DataFrame with the results
    permutation_importance_df = pd.DataFrame(permutation_importance_results.importances_mean, 
                                              index=X.columns)
    # Plot results
    # Create results directory if it doesn't exist
    directory = f'results/{opt.experiment_name}/{model_type}/feature_importance/Permutation Importance/'
    if not os.path.exists(directory):
            os.makedirs(directory)
            
    if opt.save_feature_importance_plots:
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        permutation_importance_df.sort_values(by=0, ascending=False).head(opt.num_features_to_plot).plot(kind='bar', legend=False)
        # rotate x-axis labels for better readability
        plt.xticks(rotation=opt.angle_rotate_xaxis_labels)
        plt.title(f'Permutation Importance - {model_type}')
        plt.ylabel('Importance')
        plt.savefig(f"{directory}permutation_bar.png")
        plt.show()
        plt.close()
    
    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results:
        permutation_importance_df.to_csv(f"{directory}importance.csv") 
    # Save the metrics to a log file
    if opt.save_feature_importance_metrics:
        log_metrics(directory, opt)

    # Return the DataFrame
    return permutation_importance_df


def calculate_shap_values(model, model_type, X, opt: argparse.Namespace):
    '''Calculate SHAP values for a given model and dataset
    Args:
        model: Model object
        X: Dataset
    Returns:
        shap_df: Average SHAP values
    '''
    print("Calculating SHAP Values...")
   
    explainer = shap.Explainer(model)
    
    shap_values = explainer(X)  

    # Plot results
    # Create results directory if it doesn't exist
    directory = f'results/{opt.experiment_name}/{model_type}/feature_importance/SHAP/'
    if not os.path.exists(directory):
            os.makedirs(directory)

    if  opt.save_feature_importance_plots:
        # Plot bee swarm plot
        shap.plots.beeswarm(shap_values, max_display=opt.num_features_to_plot,show=False)
        plt.yticks(rotation=opt.angle_rotate_yaxis_labels)
        plt.savefig(f"{directory}shap_beeswarm.png")
        plt.show()
  
        #Plot bar plot
        shap.plots.bar(shap_values, max_display=opt.num_features_to_plot, show=False)
        plt.yticks(rotation=opt.angle_rotate_yaxis_labels)
        plt.savefig(f"{directory}shap_bar.png")
        plt.show()
        plt.close()   
    
    
    # Calculate Average Importance + set column names as index
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame() 

    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results:
        shap_df.to_csv(f'{directory}importance.csv') 
    
    # Save the metrics to a log file
    if opt.save_feature_importance_metrics:
        log_metrics(directory, opt)
    
    # Return the DataFrame
    return shap_df
    