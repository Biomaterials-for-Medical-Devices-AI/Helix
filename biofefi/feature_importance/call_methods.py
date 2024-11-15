import argparse
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_options_dir,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_plot_dir,
    fuzzy_result_dir,
)
from biofefi.utils.utils import log_options
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

    # Save plots when the flag is set to True and importance type is not fuzzy
    if opt.save_feature_importance_plots and importance_type != "fuzzy":
        save_dir = fi_plot_dir(biofefi_experiments_base_dir() / opt.experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        plt.style.use(opt.plot_colour_scheme)
        fig, ax = plt.subplots(layout="constrained")

        feature_importance_df.sort_values(by=0, ascending=False).head(
            opt.num_features_to_plot
        ).plot(
            kind="bar",
            legend=False,
            ax=ax,
            title=f"{feature_importance_type} - {model_type}",
            ylabel="Importance",
        )
        # rotate x-axis labels for better readability
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=opt.angle_rotate_xaxis_labels,
            family=opt.plot_font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=opt.angle_rotate_yaxis_labels,
            family=opt.plot_font_family,
        )
        ax.set_xlabel(ax.get_xlabel(), family=opt.plot_font_family)
        ax.set_ylabel(ax.get_ylabel(), family=opt.plot_font_family)
        ax.set_title(ax.get_title(), family=opt.plot_font_family)
        fig.savefig(save_dir / f"{model_type}-bar.png")

        if feature_importance_type == "SHAP":
            # Plot bee swarm plot
            fig, ax = plt.subplots(layout="constrained")
            ax.set_title(
                f"{feature_importance_type} - {model_type}", family=opt.plot_font_family
            )
            shap.plots.beeswarm(
                shap_values, max_display=opt.num_features_to_plot, show=False
            )
            ax.set_xlabel(ax.get_xlabel(), family=opt.plot_font_family)
            ax.set_ylabel(ax.get_ylabel(), family=opt.plot_font_family)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                family=opt.plot_font_family,
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                family=opt.plot_font_family,
            )
            fig.savefig(save_dir / f"{model_type}-beeswarm.png")

    if opt.save_feature_importance_plots and importance_type == "fuzzy":
        # save fuzzy sets
        raise NotImplementedError(f"Plotting fuzzy sets not implemented yet")

    # Save the results to a CSV file - create folders if they don't exist
    if opt.save_feature_importance_results and importance_type != "fuzzy":
        save_dir = fi_result_dir(biofefi_experiments_base_dir() / opt.experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    if opt.save_feature_importance_results and importance_type == "fuzzy":
        save_dir = fuzzy_result_dir(
            biofefi_experiments_base_dir() / opt.experiment_name
        )
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    # Save the metrics to a log file
    if opt.save_feature_importance_options:
        options_path = fi_options_dir(
            biofefi_experiments_base_dir() / opt.experiment_name
        )
        if not options_path.exists():
            options_path.mkdir(parents=True, exist_ok=True)
        log_options(options_path, opt)


def save_fuzzy_sets_plots(
    universe, membership_functions, x_cols, opt: argparse.Namespace, logger
):
    # Plot the membership functions
    if opt.save_fuzzy_set_plots:
        logger.info(f"Saving fuzzy set plots ...")
        save_dir = fuzzy_plot_dir(biofefi_experiments_base_dir() / opt.experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)

        plt.style.use(opt.plot_colour_scheme)
        for feature in x_cols:
            fig, ax = plt.subplots(layout="constrained")
            ax.plot(
                universe[feature],
                membership_functions[feature]["low"],
                "r",
                label="Small",
            )
            ax.plot(
                universe[feature],
                membership_functions[feature]["medium"],
                "g",
                label="Moderate",
            )
            ax.plot(
                universe[feature],
                membership_functions[feature]["high"],
                "b",
                label="Large",
            )
            ax.set_title(
                f"{feature} Membership Functions",
                family=opt.plot_font_family,
            )
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=opt.angle_rotate_xaxis_labels,
                family=opt.plot_font_family,
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=opt.angle_rotate_yaxis_labels,
                family=opt.plot_font_family,
            )
            ax.legend(prop={"family": opt.plot_font_family})
            fig.savefig(save_dir / f"{feature}.png")
        plt.close()


def save_target_clusters_plots(df_cluster, opt: argparse.Namespace, logger):
    # Plot the target clusters
    if opt.save_fuzzy_set_plots:
        logger.info(f"Saving target clusters plot ...")
        save_dir = fuzzy_plot_dir(biofefi_experiments_base_dir() / opt.experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)

        # Plot boxplot of the target (continuous values) and target clusters (categories) using seaborn
        plt.style.use(opt.plot_colour_scheme)
        fig, ax = plt.subplots(layout="constrained")
        sns.boxplot(data=df_cluster, x="cluster", y="target", ax=ax)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=opt.angle_rotate_xaxis_labels,
            family=opt.plot_font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=opt.angle_rotate_yaxis_labels,
            family=opt.plot_font_family,
        )
        ax.set_title(
            "Target Clusters",
            family=opt.plot_font_family,
        )
        fig.savefig(save_dir / f"target_clusters.png")
        plt.close()
