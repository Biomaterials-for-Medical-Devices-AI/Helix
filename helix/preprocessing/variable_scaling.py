import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanCenterPoissonScaler(BaseEstimator, TransformerMixin):
    """
    Mean Centering + Poisson Scaling transformer.

    - Mean centering: subtracts feature-wise mean.
    - Poisson scaling: divides by sqrt(mean) of each feature.

    Compatible with scikit-learn pipelines.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute mean and Poisson scaling factor for each feature."""
        X = self._validate_data(X, dtype=np.float64, force_all_finite="allow-nan")
        self.mean_ = np.nanmean(X, axis=0)
        # Avoid division by zero in case mean is 0
        self.scale_ = np.sqrt(np.where(self.mean_ == 0, 1, self.mean_))
        return self

    def transform(self, X):
        """Apply mean centering and Poisson scaling."""
        X = self._validate_data(
            X, dtype=np.float64, reset=False, force_all_finite="allow-nan"
        )
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        """Reconstruct the original data."""
        X = np.asarray(X)
        return X * self.scale_ + self.mean_


class ParetoScaler(BaseEstimator, TransformerMixin):
    """
    Pareto Scaling transformer.

    - Scales each feature by sqrt(std).
    - Unlike standardization, does NOT mean-center the data.

    Compatible with scikit-learn pipelines.
    """

    def __init__(self):
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute Pareto scaling factor for each feature."""
        X = self._validate_data(X, dtype=np.float64, force_all_finite="allow-nan")
        std = np.nanstd(X, axis=0, ddof=0)
        # Avoid division by zero in case std is 0
        self.scale_ = np.sqrt(np.where(std == 0, 1, std))
        return self

    def transform(self, X):
        """Apply Pareto scaling."""
        X = self._validate_data(
            X, dtype=np.float64, reset=False, force_all_finite="allow-nan"
        )
        return X / self.scale_

    def inverse_transform(self, X):
        """Reconstruct the original data."""
        X = np.asarray(X)
        return X * self.scale_
