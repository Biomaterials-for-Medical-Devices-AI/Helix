import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder

from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import Metrics, ProblemTypes
from biofefi.options.execution import ExecutionOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions


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

        auroc.figure_.savefig(directory / f"{model_name}-{set_name}-{i}_vs_rest.png")

        plt.close()


def plot_scatter(
    y,
    yp,
    r2: float,
    set_name: str,
    dependent_variable: str,
    model_name: str,
    directory: Path,
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
        directory (str): The directory to save the plot.
        plot_opts (PlottingOptions | None, optional):
        Options for styling the plot. Defaults to None.
    """

    # Create a scatter plot using Seaborn
    plt.style.use(plot_opts.plot_colour_scheme)
    fig, ax = plt.subplots(layout="constrained")
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
    legend = "R2: " + str(float("{0:.2f}".format(r2["value"])))
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

    # Save the figure
    fig.savefig(directory / f"{model_name}-{set_name}.png")
    plt.close()


def save_actual_pred_plots(
    data: DataBuilder,
    ml_results,
    opt: ExecutionOptions,
    logger,
    ml_metric_results,
    ml_metric_results_stats,
    n_bootstraps: int,
    plot_opts: PlottingOptions | None = None,
    ml_opts: MachineLearningOptions | None = None,
    trained_models: dict | None = None,
) -> None:
    """Save Actual vs Predicted plots for Regression models
    Args:
        data: Data object
        ml_results: Results of the model
        opt: Options
        logger: Logger
        ml_metric_results: metrics of machine learning models
        ml_metric_results_stats: metrics mean and std
    Returns:
        None
    """
    if opt.problem_type == ProblemTypes.Regression:
        metric = Metrics.R2
    elif opt.problem_type == ProblemTypes.Classification:
        metric = Metrics.ROC_AUC

    model_boots_plot = {}

    for model_name, stats in ml_metric_results_stats.items():
        # Extract the mean R² for the test set
        mean_r2_test = stats["test"][metric]["mean"]

        # Find the bootstrap index closest to the mean R²
        dif = float("inf")
        closest_index = -1
        for i, bootstrap in enumerate(ml_metric_results[model_name]):
            r2_test_value = bootstrap[metric]["test"]["value"]
            current_dif = abs(r2_test_value - mean_r2_test)
            if current_dif < dif:
                dif = current_dif
                closest_index = i

        # Store the closest index
        model_boots_plot[model_name] = closest_index

    # Create results directory if it doesn't exist
    directory = Path(ml_opts.ml_plot_dir)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # Convert train and test sets to numpy arrays for easier handling
    y_test = [np.array(df) for df in data.y_test]
    y_train = [np.array(df) for df in data.y_train]

    # Scatter plot of actual vs predicted values
    for model_name, model_options in ml_opts.model_types.items():
        if model_options["use"]:
            logger.info(f"Saving actual vs prediction plots of {model_name}...")

            for i in range(n_bootstraps):
                if i != model_boots_plot[model_name]:
                    continue
                y_pred_test = ml_results[i][model_name]["y_pred_test"]
                y_pred_train = ml_results[i][model_name]["y_pred_train"]

                # Plotting the training and test results
                if opt.problem_type == ProblemTypes.Regression:
                    plot_scatter(
                        y_test[i],
                        y_pred_test,
                        ml_metric_results[model_name][i]["R2"]["test"],
                        "Test",
                        opt.dependent_variable,
                        model_name,
                        directory,
                        plot_opts=plot_opts,
                    )
                    plot_scatter(
                        y_train[i],
                        y_pred_train,
                        ml_metric_results[model_name][i]["R2"]["train"],
                        "Train",
                        opt.dependent_variable,
                        model_name,
                        directory,
                        plot_opts=plot_opts,
                    )

                else:

                    model = trained_models[model_name][i]
                    y_score_train = model.predict_proba(data.X_train[i])
                    encoder = OneHotEncoder()
                    encoder.fit(y_train[i].reshape(-1, 1))
                    y_train_labels = encoder.transform(
                        y_train[i].reshape(-1, 1)
                    ).toarray()

                    plot_auc_roc(
                        y_classes_labels=y_train_labels,
                        y_score_probs=y_score_train,
                        set_name="Train",
                        model_name=model_name,
                        directory=directory,
                        plot_opts=plot_opts,
                    )

                    y_score_test = model.predict_proba(data.X_test[i])
                    y_test_labels = encoder.transform(
                        y_test[i].reshape(-1, 1)
                    ).toarray()

                    plot_auc_roc(
                        y_classes_labels=y_test_labels,
                        y_score_probs=y_score_test,
                        set_name="Test",
                        model_name=model_name,
                        directory=directory,
                        plot_opts=plot_opts,
                    )
