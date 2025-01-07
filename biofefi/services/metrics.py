from biofefi.options.choices import CLASSIFICATION_METRICS, REGRESSION_METRICS
from biofefi.options.enums import ProblemTypes


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
