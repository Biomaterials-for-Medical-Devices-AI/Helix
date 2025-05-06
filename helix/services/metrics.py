from helix.options.choices.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from helix.options.enums import ProblemTypes


def get_metrics(problem_type: ProblemTypes, logger: object = None) -> dict:
    """Get the metrics functions for a given problem type.

    For classification:
    - Accuracy
    - F1
    - Precision
    - Recall
    - ROC AUC

    For Regression
    - R2
    - MAE
    - RMSE

    Args:
        problem_type (ProblemTypes): Where the problem is classification or regression.
        logger (object, optional): The logger. Defaults to None.

    Raises:
        ValueError: When you give an incorrect problem type.

    Returns:
        dict: A `dict` of score names and functions.
    """
    if problem_type.lower() == ProblemTypes.Classification:
        metrics = CLASSIFICATION_METRICS
    elif problem_type.lower() == ProblemTypes.Regression:
        metrics = REGRESSION_METRICS
    else:
        raise ValueError(f"Problem type {problem_type} not recognized")

    logger.info(f"Using metrics: {list(metrics.keys())}")
    return metrics


def find_mean_model_index(
    full_metrics: dict, aggregated_metrics: dict, metric_name: str
) -> int:
    """
    Find the index of the model with the mean of the metric.
    """

    for model_name, stats in aggregated_metrics.items():
        # Extract the mean metric for the test set
        mean_metric_test = stats["test"][metric_name]["mean"]

        # Find the bootstrap index closest to the mean metric
        dif = float("inf")
        closest_index = -1
        for i, bootstrap in enumerate(full_metrics[model_name]):
            metric_value = bootstrap[metric_name]["test"]["value"]
            current_dif = abs(metric_value - mean_metric_test)
            if current_dif < dif:
                dif = current_dif
                closest_index = i

    return closest_index
