import argparse
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from biofefi.options.execution import ExecutionOptions
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    fi_options_dir,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_plot_dir,
    fuzzy_result_dir,
)
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.utils.logging_utils import Logger
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
    feature_importance_df: pd.DataFrame,
    model_type,
    importance_type: str,
    feature_importance_type: str,
    experiment_name: str,
    fi_opt: FeatureImportanceOptions,
    plot_opt: PlottingOptions,
    logger: Logger,
    shap_values=None,
):
    """Save the feature importance results to a CSV file and the plots.

    Args:
        feature_importance_df (pd.DataFrame): DataFrame of feature importance results.
        model_type (_type_): Type of model.
        importance_type (str): Type of feature importance method.
        feature_importance_type (str): Type of feature importance method (Again for some reason).
        experiment_name (str): Name of the experiment, to know where to save outputs.
        fi_opt (FeatureImportanceOptions): Feature importance options.
        plot_opt (PlottingOptions): Plotting options.
        logger (Logger): The logger.
        shap_values (_type_, optional): SHAP values. Defaults to None.
    """

    # Return early if feature_importance_df is empty
    if feature_importance_df.empty:
        logger.info(f"No importance results for {feature_importance_type}...")
        return

    biofefi_base_dir = biofefi_experiments_base_dir()
    logger.info(f"Saving importance results and plots of {feature_importance_type}...")

    # Save plots when the flag is set to True and importance type is not fuzzy
    if fi_opt.save_feature_importance_plots and importance_type != "fuzzy":
        save_dir = fi_plot_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        # Plot bar plot - sort values in descending order and plot top n features
        # rotate x-axis labels for better readability
        plt.style.use(plot_opt.plot_colour_scheme)
        fig, ax = plt.subplots(layout="constrained")

        feature_importance_df.sort_values(by=0, ascending=False).head(
            fi_opt.num_features_to_plot
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
            rotation=plot_opt.angle_rotate_xaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=plot_opt.angle_rotate_yaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.set_xlabel(ax.get_xlabel(), family=plot_opt.plot_font_family)
        ax.set_ylabel(ax.get_ylabel(), family=plot_opt.plot_font_family)
        ax.set_title(ax.get_title(), family=plot_opt.plot_font_family)
        fig.savefig(save_dir / f"{model_type}-bar.png")

        if feature_importance_type == "SHAP":
            # Plot bee swarm plot
            fig, ax = plt.subplots(layout="constrained")
            ax.set_title(
                f"{feature_importance_type} - {model_type}",
                family=plot_opt.plot_font_family,
            )
            shap.plots.beeswarm(
                shap_values, max_display=fi_opt.num_features_to_plot, show=False
            )
            ax.set_xlabel(ax.get_xlabel(), family=plot_opt.plot_font_family)
            ax.set_ylabel(ax.get_ylabel(), family=plot_opt.plot_font_family)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                family=plot_opt.plot_font_family,
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                family=plot_opt.plot_font_family,
            )
            fig.savefig(save_dir / f"{model_type}-beeswarm.png")

    # Save the results to a CSV file - create folders if they don't exist
    if fi_opt.save_feature_importance_results and importance_type != "fuzzy":
        save_dir = fi_result_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    if fi_opt.save_feature_importance_results and importance_type == "fuzzy":
        save_dir = fuzzy_result_dir(biofefi_base_dir / experiment_name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        feature_importance_df.to_csv(save_dir / f"{feature_importance_type}.csv")

    # Save the metrics to a log file
    if fi_opt.save_feature_importance_options:
        options_path = fi_options_dir(biofefi_base_dir / experiment_name)
        if not options_path.exists():
            options_path.mkdir(parents=True, exist_ok=True)
        log_options(options_path, fi_opt)


def save_fuzzy_sets_plots(
    universe,
    membership_functions,
    x_cols,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    logger: Logger,
):
    # Plot the membership functions
    logger.info(f"Saving fuzzy set plots ...")
    save_dir = fuzzy_plot_dir(biofefi_experiments_base_dir() / exec_opt.experiment_name)
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    plt.style.use(plot_opt.plot_colour_scheme)
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
            family=plot_opt.plot_font_family,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=plot_opt.angle_rotate_xaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=plot_opt.angle_rotate_yaxis_labels,
            family=plot_opt.plot_font_family,
        )
        ax.legend(prop={"family": plot_opt.plot_font_family})
        fig.savefig(save_dir / f"{feature}.png")
    plt.close()


def save_target_clusters_plots(
    df_cluster, exec_opt: ExecutionOptions, plot_opt: PlottingOptions, logger: Logger
):
    # Plot the target clusters
    logger.info(f"Saving target clusters plot ...")
    save_dir = fuzzy_plot_dir(biofefi_experiments_base_dir() / exec_opt.experiment_name)
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    # Plot boxplot of the target (continuous values) and target clusters (categories) using seaborn
    plt.style.use(plot_opt.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")
    sns.boxplot(data=df_cluster, x="cluster", y="target", ax=ax)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=plot_opt.angle_rotate_xaxis_labels,
        family=plot_opt.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=plot_opt.angle_rotate_yaxis_labels,
        family=plot_opt.plot_font_family,
    )
    ax.set_title(
        "Target Clusters",
        family=plot_opt.plot_font_family,
    )
    fig.savefig(save_dir / f"target_clusters.png")
    plt.close()
