import argparse
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import log_options
import shap


def load_model(model_name, folder):
    """Load a saved model
    Args:
        model_name: Name of the model
        folder: Folder where the model is saved
    Returns:
        model: Model object
    """
    with open(f"{folder}{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
        return model


def load_data(opt: argparse.Namespace):
    raise NotImplementedError(f"Funtion load_data is not implemented")


def save_importance_results(
    feature_importance_df,
    model_type,
    importance_type,
    feature_importance_type,
    opt: argparse.Namespace,
    logger,
    shap_values=None,
):
    """Save the feature importance results to a CSV file
    Args:
        feature_importance_df: DataFrame of feature importance results
        model_type: Type of model
        feature_importance_type: Type of feature importance method
        opt: Options
        logger: Logger
        shap_values: SHAP values
    Returns:
        None
    """
    logger.info(f"Saving importance results and plots of {feature_importance_type}...")

    # Create results directory if it doesn't exist
    if importance_type == "fuzzy":
        # directory for fuzzy feature importance results
        directory = f"./log/{opt.experiment_name}/{opt.fuzzy_log_dir}/results/"
    elif model_type == None:
        # directory for ensemble feature importance results
        directory = f"./log/{opt.experiment_name}/{opt.fi_log_dir}/results/Ensemble_importances/{feature_importance_type}/"
    elif importance_type == "local":
        # directory for local model feature importance results
        directory = f"./log/{opt.experiment_name}/{opt.fi_log_dir}/results/{model_type}/local_feature_importances/{feature_importance_type}/"
    else:
        # directory for individual model feature importance results
        directory = f"./log/{opt.experiment_name}/{opt.fi_log_dir}/results/{model_type}/global_feature_importances/{feature_importance_type}/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save plots when the flag is set to True and importance type is not fuzzy
    if opt.save_feature_importance_plots and importance_type != "fuzzy":
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        feature_importance_df.sort_values(by=0, ascending=False).head(
            opt.num_features_to_plot
        ).plot(kind="bar", legend=False)
        # rotate x-axis labels for better readability
        plt.xticks(rotation=opt.angle_rotate_xaxis_labels)
        plt.title(f"{feature_importance_type} - {model_type}")
        plt.ylabel("Importance")
        plt.savefig(f"{directory}bar.png")
        # plt.show()
        plt.close()

        if feature_importance_type == "SHAP":
            # Plot bee swarm plot
            shap.plots.beeswarm(
                shap_values, max_display=opt.num_features_to_plot, show=False
            )
            plt.yticks(rotation=opt.angle_rotate_yaxis_labels)
            plt.savefig(f"{directory}beeswarm.png")
            # plt.show()

    if opt.save_feature_importance_plots and importance_type == "fuzzy":
        # save fuzzy sets
        raise NotImplementedError(f"Plotting fuzzy sets not implemented yet")

    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results and importance_type != "fuzzy":
        feature_importance_df.to_csv(f"{directory}importance.csv")

    if opt.save_feature_importance_results and importance_type == "fuzzy":
        feature_importance_df.to_csv(f"{directory}{feature_importance_type}.csv")

    # Save the metrics to a log file
    if opt.save_feature_importance_options:
        log_options(directory, opt)


def save_fuzzy_sets_plots(
    universe, membership_functions, x_cols, opt: argparse.Namespace, logger
):
    # Plot the membership functions
    if opt.save_fuzzy_set_plots:
        logger.info(f"Saving fuzzy set plots ...")
        directory = (
            f"./log/{opt.experiment_name}/{opt.fuzzy_log_dir}/results/fuzzy sets/"
        )
        if not os.path.exists(directory):
            os.makedirs(directory)

        for feature in x_cols:
            plt.figure(figsize=(5, 5))
            plt.plot(
                universe[feature],
                membership_functions[feature]["low"],
                "r",
                label="Small",
            )
            plt.plot(
                universe[feature],
                membership_functions[feature]["medium"],
                "g",
                label="Moderate",
            )
            plt.plot(
                universe[feature],
                membership_functions[feature]["high"],
                "b",
                label="Large",
            )
            plt.title(f"{feature} Membership Functions")
            plt.legend()
            plt.savefig(f"{directory}{feature}.png")
        plt.close()


def save_target_clusters_plots(df_cluster, opt: argparse.Namespace, logger):
    # Plot the target clusters
    if opt.save_fuzzy_set_plots:
        logger.info(f"Saving target clusters plot ...")
        directory = f"./log/{opt.experiment_name}/{opt.fuzzy_log_dir}/results/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Plot boxplot of the target (continuous values) and target clusters (categories) using seaborn
        plt.figure(figsize=(5, 5))
        sns.boxplot(data=df_cluster, x="cluster", y="target")
        plt.title("Target Clusters")
        plt.savefig(f"{directory}target_clusters.png")
        plt.close()
