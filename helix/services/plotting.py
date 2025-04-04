from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from helix.options.plotting import PlottingOptions


def plot_lime_importance(
    df: pd.DataFrame,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot LIME importance.

    Args:
        df (pd.DataFrame): The LIME data to plot
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The top number of features to plot.
        title (str): The title of the plot.

    Returns:
        Figure: The LIME plot.
    """
    # Calculate most important features
    most_importance_features = (
        df.abs()
        .mean()
        .sort_values(ascending=False)
        .head(num_features_to_plot)
        .index.to_list()
    )

    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)

    sns.violinplot(data=df.loc[:, most_importance_features], fill=True, ax=ax)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=plot_opts.angle_rotate_xaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=plot_opts.angle_rotate_yaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_xlabel("Feature Name", family=plot_opts.plot_font_family)
    ax.set_ylabel("Importance", family=plot_opts.plot_font_family)
    ax.set_title(title, family=plot_opts.plot_font_family, wrap=True)
    return fig


def plot_local_shap_importance(
    shap_values: shap.Explainer,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Plot a beeswarm plot of the local SHAP values.

    Args:
        shap_values (shap.Explainer): The SHAP explainer to produce the plot from.
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The number of top features to plot.
        title (str): The plot title.

    Returns:
        Figure: The beeswarm plot of local SHAP values.
    """
    # Plot bee swarm plot
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)
    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
        wrap=True,
    )
    shap.plots.beeswarm(
        shap_values,
        max_display=num_features_to_plot,
        show=False,
        color=plot_opts.plot_colour_map,
    )
    ax.set_xlabel(ax.get_xlabel(), family=plot_opts.plot_font_family)
    ax.set_ylabel(ax.get_ylabel(), family=plot_opts.plot_font_family)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        family=plot_opts.plot_font_family,
    )

    return fig


def plot_global_shap_importance(
    shap_values: pd.DataFrame,
    plot_opts: PlottingOptions,
    num_features_to_plot: int,
    title: str,
) -> Figure:
    """Produce a bar chart of global SHAP values.

    Args:
        shap_values (pd.DataFrame): The `DataFrame` containing the global SHAP values.
        plot_opts (PlottingOptions): The plotting options.
        num_features_to_plot (int): The number of top features to plot.
        title (str): The plot title.

    Returns:
        Figure: The bar chart of global SHAP values.
    """
    # Plot bar chart
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)
    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
        wrap=True,
    )
    plot_data = (
        shap_values.sort_values(by="SHAP Importance", ascending=False).head(num_features_to_plot).T
    )
    sns.barplot(data=plot_data, fill=True, ax=ax)
    ax.set_xlabel("Feature Name", family=plot_opts.plot_font_family)
    ax.set_ylabel("Abs. SHAP Importance", family=plot_opts.plot_font_family)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=plot_opts.angle_rotate_xaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=plot_opts.angle_rotate_yaxis_labels,
        family=plot_opts.plot_font_family,
    )

    return fig


def plot_auc_roc(
    y_classes_labels: np.ndarray,
    y_score_probs: np.ndarray,
    set_name: str,
    model_name: str,
    directory: Path,
    plot_opts: PlottingOptions | None = None,
):
    """
    Plot the ROC curve for a multi-class classification model.
    Args:

        y_classes_labels (numpy.ndarray): The true labels of the classes.
        y_score_probs (numpy.ndarray): The predicted probabilities of the classes.
        set_name (string): The name of the set (train or test).
        model_name (string): The name of the model.
        directory (Path): The directory path to save the plot.
        Returns:
        None
    """

    num_classes = y_score_probs.shape[1]
    start_index = 1 if num_classes == 2 else 0

    # Set colour scheme
    plt.style.use(plot_opts.plot_colour_scheme)

    for i in range(start_index, num_classes):

        auroc = RocCurveDisplay.from_predictions(
            y_classes_labels[:, i],
            y_score_probs[:, i],
            name=f"Class {i} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )

        auroc.ax_.set_xlabel(
            "False Positive Rate",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        auroc.ax_.set_ylabel(
            "True Positive Rate",
            fontsize=plot_opts.plot_axis_font_size,
            family=plot_opts.plot_font_family,
        )

        figure_title = (
            f"{model_name} {set_name} One-vs-Rest ROC curves:\n {i} Class vs Rest"
        )
        auroc.ax_.set_title(
            figure_title,
            family=plot_opts.plot_font_family,
            fontsize=plot_opts.plot_title_font_size,
            wrap=True,
        )

        auroc.ax_.legend(
            prop={
                "family": plot_opts.plot_font_family,
                "size": plot_opts.plot_axis_tick_size,
            },
            loc="lower right",
        )

        auroc.figure_.savefig(
            directory / f"{model_name}-{set_name}-{i}_vs_rest.png", dpi=plot_opts.dpi
        )

        plt.close()


def plot_confusion_matrix(
    estimator,
    X,
    y,
    set_name: str,
    model_name: str,
    directory: Path,
    plot_opts: PlottingOptions | None = None,
):
    """
    Plot the confusion matrix for a multi-class or binary classification model.

    Args:
        estimator: The trained model.
        X: The features.
        y: The true labels.
        set_name: The name of the set (train or test).
        model_name: The name of the model.
        directory: The directory path to save the plot.
        plot_opts: Options for styling the plot. Defaults to None.

    Returns:
        None
    """

    plt.style.use(plot_opts.plot_colour_scheme)

    disp = ConfusionMatrixDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        normalize=None,
        colorbar=False,
        cmap=plot_opts.plot_colour_map,
    )

    disp.ax_.set_xlabel(
        "Predicted label",
        fontsize=plot_opts.plot_axis_font_size,
        fontfamily=plot_opts.plot_font_family,
        rotation=plot_opts.angle_rotate_xaxis_labels,
    )
    disp.ax_.set_ylabel(
        "True label",
        fontsize=plot_opts.plot_axis_font_size,
        fontfamily=plot_opts.plot_font_family,
        rotation=plot_opts.angle_rotate_yaxis_labels,
    )

    disp.ax_.set_title(
        "Confusion Matrix {} - {}".format(model_name, set_name),
        fontsize=plot_opts.plot_title_font_size,
        fontfamily=plot_opts.plot_font_family,
    )

    disp.figure_.savefig(
        directory / f"{model_name}-{set_name}-confusion_matrix.png", dpi=plot_opts.dpi
    )

    plt.close()
    plt.clf()


def plot_scatter(
    y,
    yp,
    r2: float,
    set_name: str,
    dependent_variable: str,
    model_name: str,
    plot_opts: PlottingOptions | None = None,
):
    """_summary_

    Args:
        y (_type_): True y values.
        yp (_type_): Predicted y values.
        r2 (float): R-squared between `y`and `yp`.
        set_name (str): "Train" or "Test".
        dependent_variable (str): The name of the dependent variable.
        model_name (str): Name of the model.
        plot_opts (PlottingOptions | None, optional):
        Options for styling the plot. Defaults to None.
    """

    # Create a scatter plot using Seaborn
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)
    sns.scatterplot(x=y, y=yp, ax=ax)

    # Add the best fit line
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)

    # Set labels and title
    ax.set_xlabel(
        "Measured " + dependent_variable,
        fontsize=plot_opts.plot_axis_font_size,
        family=plot_opts.plot_font_family,
    )
    ax.set_ylabel(
        "Predicted " + dependent_variable,
        fontsize=plot_opts.plot_axis_font_size,
        family=plot_opts.plot_font_family,
    )
    figure_title = "Prediction Error for " + model_name + " - " + set_name
    ax.set_title(
        figure_title,
        fontsize=plot_opts.plot_title_font_size,
        family=plot_opts.plot_font_family,
        wrap=True,
    )

    # Add legend
    legend = f"R2: {(r2['value']):.3f}"
    ax.legend(
        ["Best fit", legend],
        prop={
            "family": plot_opts.plot_font_family,
            "size": plot_opts.plot_axis_tick_size,
        },
        loc="upper left",
    )

    # Add grid
    ax.grid(visible=True, axis="both")

    return fig


def plot_permutation_importance(
    df: pd.DataFrame, plot_opts: PlottingOptions, n_features: int, title: str
) -> Figure:
    """Plot a bar chart of the top n features in the feature importance dataframe,
    with the given title and styled with the given options.

    Args:
        df (pd.DataFrame): The dataframe containing the permutation importance.
        plot_opts (PlottingOptions): The options for how to configure the plot.
        n_features (int): The top number of features to plot.
        title (str): The title of the plot.

    Returns:
        Figure: The bar chart of the top n features.
    """

    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)

    top_features = df.sort_values(by="Permutation Importance", ascending=False).head(n_features).T
    sns.barplot(top_features, ax=ax)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=plot_opts.angle_rotate_xaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=plot_opts.angle_rotate_yaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_xlabel("Feature", family=plot_opts.plot_font_family)
    ax.set_ylabel("Importance", family=plot_opts.plot_font_family)
    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
        wrap=True,
    )

    return fig
