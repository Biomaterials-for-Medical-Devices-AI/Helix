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
    for _, feature_importance in feature_importance_results.items():
        for _, result in feature_importance.items():
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
    '''Calculate majority vote of feature importance results. 
    For majority vote, each vector in the feature importance matrix has their features ranked based on their importance. 
    Subsequently, the final feature importance is the average of the most common rank order for each feature. 
    For example, feature Xi has a final rank vector of [1, 1, 1, 2], where each rank rk is established by a different feature importance method k. 
    The final feature importance value for feature Xi is the average value from the three feature importance methods that ranked it as 1.
    Args:
        feature_importance_results: Dictionary containing feature importance results for each model
    Returns:
        ensemble_majorityvote: Majority vote of feature importance results
    
    '''
    ensemble_majorityvote = pd.DataFrame()
    # Loop through each model and scale the feature importance values between 0 and 1
    for _ , feature_importance in feature_importance_results.items():
        for _, result in feature_importance.items():
            # Scale the feature importance values between 0 and 1
            result = (result - result.min()) / (result.max() - result.min())
            # Add the scaled values to the ensemble_mean dataframe
            ensemble_majorityvote = pd.concat([ensemble_majorityvote, result], axis=1)
    
    # Rank the features based on their importance in each model and method
    rank_feature_importance = ensemble_majorityvote.rank(axis=0, method='dense', ascending=False)
    # Find the top two most common rank order for each feature
    majority_votes_1 = rank_feature_importance.mode(axis=1)[0]
    majority_votes_2 = rank_feature_importance.mode(axis=1)[1]

    # Assign True to rank values that are the most common
    majority_votes_rank_1 = rank_feature_importance.eq(majority_votes_1 , axis=0)
    majority_votes_rank_2 = rank_feature_importance.eq(majority_votes_2 , axis=0)

    final_majority_votes = majority_votes_rank_1 + majority_votes_rank_2

    # Calculate the mean of feature importance values in ensemble_majorityvote where majority votes is True in final_majority_votes dataframe
    ensemble_majorityvote = ensemble_majorityvote.where(final_majority_votes, 0).mean(axis=1).to_frame()
    

    return ensemble_majorityvote




def calculate_ensemble_fuzzy(feature_importance_results, opt: argparse.Namespace):
    raise NotImplementedError("Fuzzy ensemble method is not implemented yet")