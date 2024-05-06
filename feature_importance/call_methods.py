import argparse
import pickle
import os
import sys
import matplotlib.pyplot as plt
from utils.utils import log_options
import shap



def load_model(model_name, folder):
    '''Load a saved model
    Args:
        model_name: Name of the model
        folder: Folder where the model is saved
    Returns:
        model: Model object
    '''
    with open(f"{folder}{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        return model

def load_data(opt: argparse.Namespace):   
    raise NotImplementedError(f"Funtion load_data is not implemented")

def save_importance_results(feature_importance_df, model_type, feature_importance_type, opt: argparse.Namespace, logger, shap_values=None):
    '''Save the feature importance results to a CSV file
    Args:
        feature_importance_df: DataFrame of feature importance results
        model_type: Type of model
        feature_importance_type: Type of feature importance method
        opt: Options
        logger: Logger
        shap_values: SHAP values
    Returns:
        None
    '''
    logger.info(f"Saving importance results and plots of {feature_importance_type}...")

    # Create results directory if it doesn't exist
    if model_type == None:
        # directory for ensemble feature importance results
        directory = f'./log/{opt.experiment_name}/fi/results/Ensemble_importances/{feature_importance_type}/'
    else:
        # directory for individual model feature importance results
        directory = f'./log/{opt.experiment_name}/fi/results/{model_type}/indiviudal_feature_importances/{feature_importance_type}/'

    if not os.path.exists(directory):
        os.makedirs(directory)
            
    if opt.save_feature_importance_plots:
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        feature_importance_df.sort_values(by=0, ascending=False).head(opt.num_features_to_plot).plot(kind='bar', legend=False)
        # rotate x-axis labels for better readability
        plt.xticks(rotation=opt.angle_rotate_xaxis_labels)
        plt.title(f'{feature_importance_type} - {model_type}')
        plt.ylabel('Importance')
        plt.savefig(f"{directory}bar.png")
        plt.show()
        plt.close()

        if feature_importance_type == 'SHAP':
            # Plot bee swarm plot
            shap.plots.beeswarm(shap_values, max_display=opt.num_features_to_plot,show=False)
            plt.yticks(rotation=opt.angle_rotate_yaxis_labels)
            plt.savefig(f"{directory}beeswarm.png")
            plt.show()
        

    
    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results:
        feature_importance_df.to_csv(f"{directory}importance.csv") 
        
    # Save the metrics to a log file
    if opt.save_feature_importance_metrics:
        log_options(directory, opt)