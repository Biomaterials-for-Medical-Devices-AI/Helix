from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "roc_auc_score": roc_auc_score,
}
REGRESSION_METRICS = {
    "MAE": mean_absolute_error,
    "RMSE": root_mean_squared_error,
    "R2": r2_score,
}
