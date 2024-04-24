

def get_metrics(problem_type: str):
    if problem_type.lower() == "classification":
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score,
                                     roc_curve)
        return {
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
            "roc_auc_score": roc_auc_score,
            "roc_curve": roc_curve,
        }
    elif problem_type.lower() == "regression":
        from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                     r2_score)
        return {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
        }
    else:
        raise ValueError(f"Problem type {problem_type} not recognized")