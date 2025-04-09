import os
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='X has feature names')

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from helix.options.enums import Metrics, ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.ml import MachineLearningOptions
from helix.options.plotting import PlottingOptions
from helix.services.data import DataBuilder
from helix.services.plotting import plot_auc_roc, plot_beta_coefficients, plot_confusion_matrix, plot_scatter
from helix.utils.logging_utils import Logger


def save_actual_pred_plots(
    data: DataBuilder,
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

    TODO: There must be a way to break this down. There's lots of parameters and
    the function is massive.

    Args:
        data (DataBuilder): The data.
        ml_results (dict): The machine learning results.
        logger (Logger): The logger
        ml_metric_results (dict): The the results for the ML metrics.
        ml_metric_results_stats (dict): The statistics for the ML metrics.
        n_bootstraps (int): The number of bootstraps or the number of k-folds.
        exec_opts (ExecutionOptions): The execution options.
        plot_opts (PlottingOptions | None, optional): The plot options. Defaults to None.
        ml_opts (MachineLearningOptions | None, optional): The machine learning options. Defaults to None.
        trained_models (dict | None, optional): The machine learning models. Defaults to None.
    """
    if exec_opts.problem_type == ProblemTypes.Regression:
        metric = Metrics.R2
    elif exec_opts.problem_type == ProblemTypes.Classification:
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
                if exec_opts.problem_type == ProblemTypes.Regression:
                    test_plot = plot_scatter(
                        y_test[i],
                        y_pred_test,
                        ml_metric_results[model_name][i]["R2"]["test"],
                        "Test",
                        exec_opts.dependent_variable,
                        model_name,
                        plot_opts=plot_opts,
                    )
                    test_plot.savefig(directory / f"{model_name}-{i}-Test.png")
                    train_plot = plot_scatter(
                        y_train[i],
                        y_pred_train,
                        ml_metric_results[model_name][i]["R2"]["train"],
                        "Train",
                        exec_opts.dependent_variable,
                        model_name,
                        plot_opts=plot_opts,
                    )
                    train_plot.savefig(directory / f"{model_name}-{i}-Train.png")

                    # Add beta coefficients plot for linear regression models
                    if model_name in ["linear model", "multiple linear regression with expectation maximisation"]:
                        model = trained_models[model_name][i]
                        if hasattr(model, "coef_"):
                            coef_plot = plot_beta_coefficients(
                                coefficients=model.coef_,
                                feature_names=data.X_train[i].columns.tolist(),
                                plot_opts=plot_opts,
                                model_name=model_name,
                                dependent_variable=exec_opts.dependent_variable
                            )
                            coef_plot.savefig(directory / f"{model_name}-{i}-Coefficients.png")
                            plt.close(coef_plot)

                    plt.close(test_plot)
                    plt.close(train_plot)

                else:

                    model = trained_models[model_name][i]
                    y_score_train = ml_results[i][model_name]["y_pred_train_proba"]
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

                    plot_confusion_matrix(
                        estimator=model,
                        X=data.X_train[i],
                        y=y_train[i],
                        set_name="Train",
                        model_name=model_name,
                        directory=directory,
                        plot_opts=plot_opts,
                    )

                    y_score_test = ml_results[i][model_name]["y_pred_test_proba"]
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

                    plot_confusion_matrix(
                        estimator=model,
                        X=data.X_test[i],
                        y=y_test[i],
                        set_name="Test",
                        model_name=model_name,
                        directory=directory,
                        plot_opts=plot_opts,
                    )
