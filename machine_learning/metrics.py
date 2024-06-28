from typing import Dict


def get_metrics(problem_type: str, logger: object = None) -> Dict:
    if problem_type.lower() == "classification":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score

        )



        metrics = {
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "roc_auc_score": roc_auc_score,
        }
    elif problem_type.lower() == "regression":
        from sklearn.metrics import (
            mean_absolute_error,
            r2_score,
            root_mean_squared_error,
        )

        metrics = {
            "MAE": mean_absolute_error,
            "RMSE": root_mean_squared_error,
            "R2": r2_score,
        }
    else:
        raise ValueError(f"Problem type {problem_type} not recognized")

    logger.info(f"Using metrics: {list(metrics.keys())}")
    return metrics
