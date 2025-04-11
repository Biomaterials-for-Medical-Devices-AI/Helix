from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
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
        shap_values.sort_values(by="SHAP Importance", ascending=False)
        .head(num_features_to_plot)
        .T
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set_name: str,
    model_name: str,
    directory: Path,
    plot_opts: PlottingOptions | None = None,
):
    """
    Plot the confusion matrix for a multi-class or binary classification model.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        set_name: The name of the set (train or test).
        model_name: The name of the model.
        directory: The directory path to save the plot.
        plot_opts: Options for styling the plot. Defaults to None.

    Returns:
        None
    """

    plt.style.use(plot_opts.plot_colour_scheme)

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
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


def plot_beta_coefficients(
    coefficients: np.ndarray,
    feature_names: list,
    plot_opts: PlottingOptions,
    model_name: str,
    dependent_variable: str | None = None,
    standard_errors: np.ndarray | None = None,
) -> Figure:
    """Create a bar plot of model coefficients with different colors for positive/negative values.

    Args:
        coefficients (np.ndarray): The model coefficients. For logistic regression, uses first class coefficients
        feature_names (list): Names of the features corresponding to coefficients
        plot_opts (PlottingOptions): The plotting options
        model_name (str): Name of the model for the plot title
        dependent_variable (str | None, optional): Name of the dependent variable. Defaults to None.
        standard_errors (np.ndarray | None, optional): Standard errors of coefficients. Defaults to None.

    Returns:
        Figure: The coefficient plot
    """
    plt.style.use(plot_opts.plot_colour_scheme)

    # For logistic regression, use coefficients for first class
    if len(coefficients.shape) > 1:
        coefficients = coefficients[0]  # Take first class coefficients

    # Calculate figure height based on number of coefficients
    # Base height of 3 inches, plus 0.25 inches per coefficient
    n_coefs = min(len(coefficients), 20)  # We show max 20 coefficients
    fig_height = 3 + (0.16 * n_coefs)

    fig, ax = plt.subplots(
        layout="constrained", dpi=plot_opts.dpi, figsize=(10, fig_height)
    )

    # Sort coefficients and keep track of indices
    coef_with_idx = list(enumerate(coefficients))
    coef_with_idx.sort(key=lambda x: abs(x[1]), reverse=True)
    sorted_indices = [x[0] for x in coef_with_idx]
    sorted_coefs = [x[1] for x in coef_with_idx]

    # Get corresponding feature names and errors
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_errors = None
    if standard_errors is not None:
        sorted_errors = [standard_errors[i] for i in sorted_indices]

    # Take top 20 coefficients by magnitude if there are more
    if len(sorted_coefs) > 20:
        sorted_coefs = sorted_coefs[:20]
        sorted_features = sorted_features[:20]
        if sorted_errors:
            sorted_errors = sorted_errors[:20]

    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_coefs))
    colors = ["blue" if c >= 0 else "red" for c in sorted_coefs]
    ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.6)

    # Add error bars if available
    if sorted_errors:
        ax.errorbar(
            sorted_coefs,
            y_pos,
            xerr=1.96 * np.array(sorted_errors),  # 95% confidence interval
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
            elinewidth=1,
            zorder=3,
        )

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        sorted_features,
        family=plot_opts.plot_font_family,
        fontsize=plot_opts.plot_axis_tick_size,
    )
    ax.set_xlabel(
        "Coefficient Value",
        family=plot_opts.plot_font_family,
        fontsize=plot_opts.plot_axis_font_size,
    )

    # Create title with dependent variable if provided
    model_name = " ".join(word.capitalize() for word in model_name.split())
    title = f"Beta Coefficients - {model_name}"
    if dependent_variable:
        title += f"\nDependent Variable: {dependent_variable}"

    ax.set_title(
        title,
        family=plot_opts.plot_font_family,
        fontsize=plot_opts.plot_title_font_size,
        wrap=True,
    )

    # Add gridlines
    ax.grid(True, axis="x", linestyle="--", alpha=0.3, zorder=0)

    # Add zero line
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, zorder=1)

    # Adjust layout
    fig.tight_layout()


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

    top_features = (
        df.sort_values(by="Permutation Importance", ascending=False).head(n_features).T
    )
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


def plot_bar_chart(
    df: pd.DataFrame,
    sort_key: Any,
    plot_opts: PlottingOptions,
    title: str,
    x_label: str,
    y_label: str,
    n_features: int = 10,
    error_bars: pd.DataFrame | None = None,
) -> Figure:
    """Plot a bar chart of the top n features from the given dataframe.

    Args:
        df (pd.DataFrame): The data to be plotted.
        plot_opts (PlottingOptions): The options for styling the plot.
        sort_key (str): The key by which to sort the data. This can be the name of a column.
        title (str): The title of the plot.
        x_label (str): The label for the X axis.
        y_label (str): The label for the Y axis.
        n_features (int, optional): The top number of featurs to plot. Defaults to 10.

    Returns:
        Figure: The bar chart of the top n features.
    """

    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained", dpi=plot_opts.dpi)

    # Get top features
    top_df = df.sort_values(by=sort_key, ascending=False).head(n_features)
    x = top_df.index.tolist()
    y = top_df[sort_key].values

    # If error bars provided, align them
    yerr = None
    if error_bars is not None and sort_key in error_bars.columns:
        yerr = error_bars.loc[x, sort_key].values

    # Plot with error bars
    ax.bar(x, y, yerr=yerr, capsize=5)

    # Label formatting
    ax.set_xticklabels(
        x,
        rotation=plot_opts.angle_rotate_xaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_yticklabels(
        ax.get_yticks(),
        rotation=plot_opts.angle_rotate_yaxis_labels,
        family=plot_opts.plot_font_family,
    )
    ax.set_xlabel(x_label, family=plot_opts.plot_font_family)
    ax.set_ylabel(y_label, family=plot_opts.plot_font_family)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(title, family=plot_opts.plot_font_family, wrap=True)

    return fig
