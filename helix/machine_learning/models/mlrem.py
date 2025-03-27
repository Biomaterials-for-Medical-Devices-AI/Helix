import numpy as np
import numpy.linalg.linalg as LA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_X_y


class EMLinearRegression(RegressorMixin, BaseEstimator):
    """Multiple Linear Regression with Expectation-Maximisation.

    This implementation uses the EM algorithm for feature selection and weight optimisation.

    Parameters
    ----------
    alpha : float, default=0.
        Regularisation parameter.
    max_beta : float, default=50
        Maximum beta value to optimise over.
    weight_threshold : float, default=1e-3
        Threshold for feature removal.
    max_iterations : int, default=300
        Maximum number of EM algorithm iterations.
    tolerance : float, default=0.01
        Convergence tolerance for relative change in SSD.
    """

    def __init__(
        self,
        alpha=0.5,
        max_beta=20,
        weight_threshold=1e-3,
        max_iterations=300,
        tolerance=0.01,
    ):
        self.alpha = alpha
        self.max_beta = max_beta
        self.weight_threshold = weight_threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Attributes set during fit
        self.best_beta = None
        self.weights_ = None
        self.coefficients_ = None
        self.intercept_ = None
        self.p_values_ = None

    def fit(self, X, y):
        """Fit the EM linear regression model with optimised beta.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        feature_names : list of str, optional
            Names of features. If None, will use indices.

        Returns
        -------
        self : object
            Fitted model for the best value of beta within the specified range.
        """
        # Scale inputs
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Input validation
        X, y = check_X_y(X, y, y_numeric=True)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        # Add intercept
        H = np.ones((n_samples, n_features + 1), float)
        H[:, 1:] = X

        best_beta = None
        best_weights = None
        best_ssd = float("inf")

        # Beta optimisation loop
        for beta in np.arange(0.01, self.max_beta, 0.01):
            weights, ssd = self._em_algorithm(H, y, beta)
            if ssd < best_ssd:
                best_ssd = ssd
                best_beta = beta
                best_weights = weights

        self.best_beta = best_beta
        self.weights_ = best_weights
        self.intercept_ = best_weights[0, 0]
        self.coefficients_ = best_weights[1:].flatten()
        return self

    def _em_algorithm(self, H, y, beta):
        """Expectation-Maximisation algorithm for feature selection and weight estimation."""
        n_samples, n_features = H.shape
        HT = H.T
        HTy = HT @ y
        weights = LA.pinv(HT @ H) @ HTy

        ssd2 = 1.0  # Initial sum of squared differences
        change = 1.0
        iteration = 0

        # The loop below iterates through the EM process until convergence criteria are met.
        # It stops if the maximum number of iterations is reached, if the relative change in SSD is below tolerance,
        # or if SSD becomes very small, indicating convergence.
        #
        # In simpler terms:
        # - The loop ensures the algorithm keeps refining the model until it finds the best possible estimates.
        # - It stops when changes between iterations are small enough (convergence) or if the model has already stabilised.

        while (
            iteration < self.max_iterations and change > self.tolerance and ssd2 > 1e-15
        ):
            iteration += 1
            ssd1 = ssd2

            # Update matrices
            U = np.diag(
                np.abs(weights).flatten()
            )  # Diagonal matrix of absolute weight values to scale feature contributions
            Ic = np.eye(n_features) * (
                self.alpha + beta * ssd1**2
            )  # Regularisation matrix incorporating alpha and beta

            # M-step weight updates
            R = LA.pinv(
                Ic + U @ HT @ H @ U
            )  # Intermediate inverse matrix for weight computation
            weights = U @ R @ U @ HTy  # Updated weight vector

            # Compute predictions and residuals
            predictions = H @ weights
            residuals = predictions - y
            ssd2 = np.sqrt(np.sum(residuals**2) / n_samples)

            # Check convergence
            change = 100 * np.abs(ssd2 - ssd1) / ssd1

        return (
            weights,
            ssd2,
        )  # Return the final weights and sum of squared differences for the EM algorithm

    def predict(self, X):
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        X = check_array(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        n_samples = X.shape[0]

        X = np.hstack((np.ones((n_samples, 1)), X))
        predictions = X @ np.vstack(
            ([self.intercept_], self.coefficients_.reshape(-1, 1))
        )
        return predictions.ravel()

    def score(self, X, y, sample_weight=None):
        r2 = super().score(X, y, sample_weight)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        self.r2_ = r2
        self.adjusted_r2_ = adjusted_r2
        return r2
