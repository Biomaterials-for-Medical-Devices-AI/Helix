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

    sns.violinplot(data=df.loc[:, most_importance_features], fill=True, ax=ax, palette=plot_opts.plot_colour_map)

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
    model_name: str,
    set_name: str,
    directory: Path,
    plot_opts: PlottingOptions,
) -> None:
    """Plot the ROC curve for a multi-class classification model.

    Args:
        y_classes_labels (numpy.ndarray): The true labels of the classes.
        y_score_probs (numpy.ndarray): The predicted probabilities of the classes.
        model_name (string): The name of the model.
        set_name (string): The name of the set (train or test).
        directory (Path): The directory path to save the plot.
        plot_opts (PlottingOptions): The plotting options.
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

        display_model_name = " ".join(word.capitalize() for word in model_name.split())
        figure_title = f"{display_model_name} {set_name} One-vs-Rest ROC curves:\n {i} Class vs Rest"
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

        # Save plot with standardized filename
        auroc.figure_.savefig(
            directory / f"{model_name}-{set_name.lower()}-{i}_vs_rest-roc.png",
            dpi=plot_opts.dpi,
        )

        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    set_name: str,
    directory: Path,
    plot_opts: PlottingOptions,
) -> None:
    """Plot the confusion matrix for a multi-class or binary classification model.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        model_name (str): The name of the model.
        set_name (str): The name of the set (train or test).
        directory (Path): The directory path to save the plot.
        plot_opts (PlottingOptions): The plotting options.
    """
    plt.style.use(plot_opts.plot_colour_scheme)

    # Create title with capitalized model name
    display_model_name = " ".join(word.capitalize() for word in model_name.split())

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        normalize=None,
        colorbar=False,
        cmap=plot_opts.plot_colour_map,
    )

    title = f"Confusion Matrix - {display_model_name} - {set_name}"
    disp.ax_.set_title(
        title,
        fontsize=plot_opts.plot_title_font_size,
        fontfamily=plot_opts.plot_font_family,
        wrap=True,
    )

    disp.ax_.set_xlabel(
        "Predicted label",
        fontsize=plot_opts.plot_axis_font_size,
        family=plot_opts.plot_font_family,
    )
    disp.ax_.set_ylabel(
        "True label",
        fontsize=plot_opts.plot_axis_font_size,
        family=plot_opts.plot_font_family,
        rotation=plot_opts.angle_rotate_yaxis_labels,
    )

    # Save plot with standardized filename
    disp.figure_.savefig(
        directory / f"{model_name}-{set_name.lower()}-confusion.png", dpi=plot_opts.dpi
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
    plot_opts: PlottingOptions,
) -> Figure:
    """Create a scatter plot comparing predicted vs actual values.

    Args:
        y (_type_): True y values.
        yp (_type_): Predicted y values.
        r2 (float): R-squared between `y`and `yp`.
        set_name (str): "Train" or "Test".
        dependent_variable (str): The name of the dependent variable.
        model_name (str): Name of the model.
        plot_opts (PlottingOptions): Options for styling the plot.

    Returns:
        Figure: The scatter plot figure
    """

    # Create a scatter plot using Seaborn
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=plot_opts.dpi)
    sns.scatterplot(x=y, y=yp, ax=ax)

    # Add the best fit line
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2, label="Best fit")

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

    # Create title
    if model_name.lower() == "linear model":
        display_model_name = "Linear Regression"
    else:
        display_model_name = " ".join(word.capitalize() for word in model_name.split())

    figure_title = f"Prediction Error for {display_model_name} - {set_name}"
    ax.set_title(
        figure_title,
        fontsize=plot_opts.plot_title_font_size,
        family=plot_opts.plot_font_family,
        wrap=True,
    )

    # Add legend with R2 value
    if isinstance(r2, dict) and "value" in r2:
        r2_value = r2["value"]
    else:
        r2_value = r2
    legend = f"R²: {r2_value:.3f}"
    ax.plot([], [], " ", label=legend)  # Add empty line for R2 in legend
    ax.legend(
        prop={
            "family": plot_opts.plot_font_family,
            "size": plot_opts.plot_axis_tick_size,
        },
        loc="upper left",
    )

    # Add grid
    ax.grid(visible=True, axis="both", linestyle="--", alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_beta_coefficients(
    coefficients: np.ndarray,
    feature_names: list,
    plot_opts: PlottingOptions,
    model_name: str,
    dependent_variable: str | None = None,
    standard_errors: np.ndarray | None = None,
    is_classification: bool = False,
) -> Figure:
    """Create a bar plot of model coefficients with different colors for positive/negative values.

    Args:
        coefficients (np.ndarray): The model coefficients. For logistic regression, uses first class coefficients
        feature_names (list): Names of the features
        plot_opts (PlottingOptions): Plot styling options
        model_name (str): Name of the model for the plot title
        dependent_variable (str | None, optional): Name of the dependent variable. Defaults to None.
        standard_errors (np.ndarray | None, optional): Standard errors of coefficients. Defaults to None.
        is_classification (bool, optional): Whether this is a classification model. Defaults to False.

    Returns:
        Figure: The coefficient plot
    """
    plt.style.use(plot_opts.plot_colour_scheme)

    # Handle different coefficient shapes
    if len(coefficients.shape) > 1:
        coefficients = coefficients[0]

    # Ensure coefficients and feature names match in length
    if len(coefficients) != len(feature_names):
        raise ValueError(
            f"Number of coefficients ({len(coefficients)}) does not match "
            f"number of feature names ({len(feature_names)})"
        )

    # Sort coefficients by their values
    sorted_indices = np.argsort(coefficients)
    coefficients = coefficients[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    if standard_errors is not None:
        standard_errors = standard_errors[sorted_indices]

    # Calculate figure dimensions based on number of coefficients
    n_coef = len(coefficients)
    base_height = 4  # minimum height in inches
    height_per_coef = 0.25  # additional height per coefficient
    fig_height = min(base_height + height_per_coef * n_coef, 20)  # cap at 20 inches

    # Only scale feature names font size based on number of coefficients
    font_scale = max(0.5, min(1.0, 25 / n_coef))  # scale between 0.5 and 1.0
    axis_font_size = int(plot_opts.plot_axis_font_size * font_scale)

    # Create figure with calculated dimensions
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=plot_opts.dpi)

    # Get colors from the palette
    palette = sns.color_palette(plot_opts.plot_colour_map)
    negative_color = palette[0]  # First color for negative values
    positive_color = palette[-1]  # Last color for positive values

    # Create color list based on coefficient signs
    colors = [negative_color if c < 0 else positive_color for c in coefficients]

    # Plot horizontal bars
    y_pos = np.arange(len(coefficients))
    bars = ax.barh(y_pos, coefficients, color=colors)

    # Add error bars if provided
    if standard_errors is not None:
        error_positions = np.array(
            [
                (
                    bar.get_width()
                    if bar.get_width() >= 0
                    else bar.get_width() + standard_errors[i]
                )
                for i, bar in enumerate(bars)
            ]
        )
        ax.errorbar(
            error_positions,
            y_pos,
            xerr=standard_errors,
            fmt="none",
            color="black",
            capsize=5,
            zorder=2,
        )

    # Customize axes with scaled font size only for feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        feature_names,
        family=plot_opts.plot_font_family,
        fontsize=axis_font_size,
    )

    # Create title
    if model_name.lower() == "linear model":
        display_model_name = (
            "Logistic Regression" if is_classification else "Linear Regression"
        )
    else:
        display_model_name = " ".join(word.capitalize() for word in model_name.split())

    title = f"Beta Coefficients - {display_model_name}"
    if dependent_variable:
        title += f"\nDependent Variable: {dependent_variable}"

    # Use original title font size
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
    plt.tight_layout()

    return fig


def plot_permutation_importance(
    df: pd.DataFrame,
    plot_opts: PlottingOptions,
    n_features: int,
    title: str,
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
        error_bars (pd.DataFrame | None, optional): Error bars for the plot. Defaults to None.
        directory (Path | None, optional): The directory to save the plot. Defaults to None.
        model_name (str | None, optional): The name of the model. Defaults to None.
        set_name (str | None, optional): The name of the set (train or test). Defaults to None.

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
