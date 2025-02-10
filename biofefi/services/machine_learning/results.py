import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from biofefi.machine_learning.data import DataBuilder
from biofefi.options.enums import Metrics, ProblemTypes
from biofefi.options.execution import ExecutionOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.plotting import plot_auc_roc, plot_confusion_matrix, plot_scatter


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
                    test_plot = plot_scatter(
                        y_test[i],
                        y_pred_test,
                        ml_metric_results[model_name][i]["R2"]["test"],
                        "Test",
                        opt.dependent_variable,
                        model_name,
                        plot_opts=plot_opts,
                    )
                    test_plot.savefig(directory / f"{model_name}-{i}-Test.png")
                    train_plot = plot_scatter(
                        y_train[i],
                        y_pred_train,
                        ml_metric_results[model_name][i]["R2"]["train"],
                        "Train",
                        opt.dependent_variable,
                        model_name,
                        plot_opts=plot_opts,
                    )
                    train_plot.savefig(directory / f"{model_name}-{i}-Train.png")

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
