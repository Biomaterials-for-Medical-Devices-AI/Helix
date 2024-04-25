import argparse
import os
import pandas as pd
from utils.utils import log_metrics
import matplotlib.pyplot as plt

def calculate_ensemble_mean(feature_importance_results, opt: argparse.Namespace):
    '''Calculate mean of feature importance results
    Args:
        feature_importance_results: Dictionary containing feature importance results for each model
    Returns:
        ensemble_mean: Mean of feature importance results
    '''
    print("Calculating mean of feature importance results...")
    # create a dataframe to store the mean of feature importance results
    # with the feature names as the index
    ensemble_mean = pd.DataFrame()
    
    # Loop through each model and scale the feature importance values between 0 and 1
    for model_type, feature_importance in feature_importance_results.items():
        for method, result in feature_importance.items():
            # Scale the feature importance values between 0 and 1
            result = (result - result.min()) / (result.max() - result.min())
            # Add the scaled values to the ensemble_mean dataframe
            ensemble_mean = pd.concat([ensemble_mean, result], axis=1)
    
    # Calculate the mean of the feature importance values across models
    ensemble_mean = ensemble_mean.mean(axis=1).to_frame()

    # Add the feature names as index to the ensemble_mean dataframe
    ensemble_mean.index = result.index 

    # Plot results
    # Create results directory if it doesn't exist
    directory = f'results/{opt.experiment_name}/Ensemble_importance/Mean/'
    if not os.path.exists(directory):
            os.makedirs(directory)

    if opt.save_feature_importance_plots:
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        ensemble_mean.sort_values(by=0, ascending=False).head(opt.num_features_to_plot).plot(kind='bar', legend=False)
        # rotate x-axis labels for better readability
        plt.xticks(rotation=opt.angle_rotate_xaxis_labels)
        plt.title(f'Mean Feature Importance')
        plt.ylabel('Importance')
        plt.savefig(f"{directory}mean_bar.png")
        plt.show()
        plt.close()
    
    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results:
        ensemble_mean.to_csv(f"{directory}importance.csv") 
    # Save the metrics to a log file
    if opt.save_feature_importance_metrics:
        log_metrics(directory, opt)     
    

    # Return the DataFrame
    return ensemble_mean


def calculate_ensemble_majorityvote(feature_importance_results, opt: argparse.Namespace):
    raise NotImplementedError("Majority vote ensemble method is not implemented yet")


def calculate_ensemble_fuzzy(feature_importance_results, opt: argparse.Namespace):
    raise NotImplementedError("Fuzzy ensemble method is not implemented yet")