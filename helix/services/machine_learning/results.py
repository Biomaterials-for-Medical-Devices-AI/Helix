import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from helix.options.enums import Metrics, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.ml import MachineLearningOptions
from helix.options.plotting import PlottingOptions
from helix.services.data import TabularData
from helix.services.plotting import (
    plot_auc_roc,
    plot_beta_coefficients,
    plot_confusion_matrix,
    plot_scatter,
)
from helix.utils.logging_utils import Logger

warnings.filterwarnings("ignore", message="X has feature names")


def _get_metric_for_problem_type(problem_type: ProblemTypes) -> str:
    """Get the appropriate metric based on the problem type.

    Args:
        problem_type (ProblemTypes): The type of problem (Regression or Classification)

    Returns:
        str: The metric to use
    """
    return Metrics.R2 if problem_type == ProblemTypes.Regression else Metrics.ROC_AUC


def _find_closest_bootstrap_index(
    ml_metric_results: dict,
    ml_metric_results_stats: dict,
    model_name: str,
    metric: str,
) -> int:
    """Find the bootstrap index closest to the mean metric value.

    Args:
        ml_metric_results (dict): The machine learning results
        ml_metric_results_stats (dict): The statistics for the ML metrics
        model_name (str): Name of the model
        metric (str): Metric to use for comparison

    Returns:
        int: Index of the bootstrap closest to the mean
    """
    mean_metric_test = ml_metric_results_stats[model_name]["test"][metric]["mean"]
    closest_index = -1
    min_diff = float("inf")

    for i, bootstrap in enumerate(ml_metric_results[model_name]):
        metric_test_value = bootstrap[metric]["test"]["value"]
        current_diff = abs(metric_test_value - mean_metric_test)
        if current_diff < min_diff:
            min_diff = current_diff
            closest_index = i

    return closest_index


def _save_regression_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_results: dict,
    split_type: str,
    dependent_variable: str,
    model_name: str,
    directory: Path,
    bootstrap_index: int,
    plot_opts: PlottingOptions,
) -> None:
    """Save regression plots for a specific split.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        metric_results (dict): Results for the metrics
        split_type (str): Type of split (Train/Test)
        dependent_variable (str): Name of dependent variable
        model_name (str): Name of the model
        directory (Path): Directory to save plots
        bootstrap_index (int): Index of the bootstrap
        plot_opts (PlottingOptions): Plot options
    """
    plot = plot_scatter(
        y_true,
        y_pred,
        metric_results["R2"][split_type.lower()],
        split_type,
        dependent_variable,
        model_name,
        plot_opts=plot_opts,
    )
    plot.savefig(directory / f"{model_name}-{bootstrap_index}-{split_type}.png")


def _save_coefficient_plot(
    model,
    feature_names: list,
    plot_opts: PlottingOptions,
    model_name: str,
    dependent_variable: str,
    directory: Path,
    bootstrap_index: int,
) -> None:
    """Save coefficient plot for linear regression models.

    Args:
        model: The trained model
        feature_names (list): List of feature names
        plot_opts (PlottingOptions): Plot options
        model_name (str): Name of the model
        dependent_variable (str): Name of dependent variable
        directory (Path): Directory to save plots
        bootstrap_index (int): Index of the bootstrap
    """
    if hasattr(model, "coef_"):
        coef_plot = plot_beta_coefficients(
            coefficients=model.coef_,
            feature_names=feature_names,
            plot_opts=plot_opts,
            model_name=model_name,
            dependent_variable=dependent_variable,
        )
        coef_plot.savefig(
            directory / f"{model_name}-{bootstrap_index}-Coefficients.png"
        )


def _save_classification_plots(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model: object,
    split_type: str,
    model_name: str,
    directory: Path,
    plot_opts: PlottingOptions,
    logger: Logger,
    feature_names: list | None = None,  # Add feature_names parameter
) -> None:
    """Save classification plots for a specific split.

    Args:
        y_true (np.ndarray): True values
        y_pred_proba (np.ndarray): Predicted probabilities
        model: The trained model
        split_type (str): Type of split (Train/Test)
        model_name (str): Name of the model
        directory (Path): Directory to save plots
        plot_opts (PlottingOptions): Plot options
        logger (Logger): The logger instance
        feature_names (list | None, optional): List of feature names. Defaults to None.
    """
    try:
        encoder = OneHotEncoder()
        encoder.fit(y_true.reshape(-1, 1))
        y_true_labels = encoder.transform(y_true.reshape(-1, 1)).toarray()

        plot_auc_roc(
            y_classes_labels=y_true_labels,
            y_score_probs=y_pred_proba,
            set_name=split_type,
            model_name=model_name,
            directory=directory,
            plot_opts=plot_opts,
        )
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")

    try:
        # Get predictions for confusion matrix
        y_pred = (
            np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            y_true=y_true,  # True labels
            y_pred=y_pred,  # Predicted labels
            set_name=split_type,
            model_name=model_name,
            directory=directory,
            plot_opts=plot_opts,
        )
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")


def save_actual_pred_plots(
    data: TabularData,
    ml_results: dict,
    logger: Logger,
    ml_metric_results: dict,
    ml_metric_results_stats: dict,
    n_bootstraps: int,
    exec_opts: ExecutionOptions,
    plot_opts: PlottingOptions | None = None,
    ml_opts: MachineLearningOptions | None = None,
    trained_models: dict | None = None,
) -> None:
    """Save actual vs prediction plots for classification and regression.

    Args:
        data (DataBuilder): The data
        ml_results (dict): The machine learning results
        logger (Logger): The logger
        ml_metric_results (dict): The results for the ML metrics
        ml_metric_results_stats (dict): The statistics for the ML metrics
        n_bootstraps (int): The number of bootstraps or k-folds
        exec_opts (ExecutionOptions): The execution options
        plot_opts (PlottingOptions | None, optional): The plot options. Defaults to None
        ml_opts (MachineLearningOptions | None, optional): The ML options. Defaults to None
        trained_models (dict | None, optional): The ML models. Defaults to None
    """
    metric = _get_metric_for_problem_type(exec_opts.problem_type)
    # Create results directory
    directory = Path(ml_opts.ml_plot_dir)
    os.makedirs(directory, exist_ok=True)
    # Convert data to numpy arrays
    y_test = [np.array(df) for df in data.y_test]
    y_train = [np.array(df) for df in data.y_train]
    # Process each model
    for model_name, model_options in ml_opts.model_types.items():
        if not model_options["use"]:
            continue
        logger.info(f"Saving actual vs prediction plots of {model_name}...")
        # Find best bootstrap index
        closest_index = _find_closest_bootstrap_index(
            ml_metric_results, ml_metric_results_stats, model_name, metric
        )
        # Get predictions for best bootstrap
        y_pred_test = ml_results[closest_index][model_name]["y_pred_test"]
        y_pred_train = ml_results[closest_index][model_name]["y_pred_train"]
        if exec_opts.problem_type == ProblemTypes.Regression:
            # Save regression plots
            _save_regression_plots(
                y_test[closest_index],
                y_pred_test,
                ml_metric_results[model_name][closest_index],
                "Test",
                exec_opts.dependent_variable,
                model_name,
                directory,
                closest_index,
                plot_opts,
            )
            _save_regression_plots(
                y_train[closest_index],
                y_pred_train,
                ml_metric_results[model_name][closest_index],
                "Train",
                exec_opts.dependent_variable,
                model_name,
                directory,
                closest_index,
                plot_opts,
            )
            # Save coefficient plots for linear models
            if model_name in [
                "linear model",
                "multiple linear regression with expectation maximisation",
            ]:
                _save_coefficient_plot(
                    trained_models[model_name][closest_index],
                    data.X_train[closest_index].columns.tolist(),
                    plot_opts,
                    model_name,
                    exec_opts.dependent_variable,
                    directory,
                    closest_index,
                )
        else:
            # Save classification plots
            y_pred_test_proba = ml_results[closest_index][model_name][
                "y_pred_test_proba"
            ]
            y_pred_train_proba = ml_results[closest_index][model_name][
                "y_pred_train_proba"
            ]
            _save_classification_plots(
                y_test[closest_index],
                y_pred_test_proba,
                trained_models[model_name][closest_index],
                "Test",
                model_name,
                directory,
                plot_opts,
                logger,
            )
            _save_classification_plots(
                y_train[closest_index],
                y_pred_train_proba,
                trained_models[model_name][closest_index],
                "Train",
                model_name,
                directory,
                plot_opts,
                logger,
            )
            # Save coefficient plots for linear classification models
            if model_name == "linear model" and hasattr(
                trained_models[model_name][closest_index], "coef_"
            ):
                _save_coefficient_plot(
                    trained_models[model_name][closest_index],
                    data.X_train[closest_index].columns.tolist(),
                    plot_opts,
                    model_name,
                    exec_opts.dependent_variable,
                    directory,
                    closest_index,
                )
