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

        Args:
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        feature_names : list of str, optional
            Names of features. If None, will use indices.

        Returns:
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
        HT = H.T
        HTy = HT @ y
        G = HT @ H
        weights = LA.pinv(HT @ H) @ HTy

        best_beta = None
        best_weights = None
        best_ssd = float("inf")

        # Beta optimisation loop
        for beta in np.logspace(-2, np.log10(self.max_beta), 50):
            weights, ssd = self._em_algorithm(H, HTy, G, y, beta, weights)
            if ssd < best_ssd:
                best_ssd = ssd
                best_beta = beta
                best_weights = weights

        self.best_beta = best_beta
        self.weights_ = best_weights
        self.intercept_ = best_weights[0, 0]
        self.coefficients_ = best_weights[1:].flatten()
        return self

    def _em_algorithm(self, H, HTy, G, y, beta, weights):
        """Expectation-Maximisation algorithm for feature selection and weight estimation."""
        n_samples, n_features = H.shape

        ssd2 = 1.0
        change = 1.0
        iteration = 0

        while (
            iteration < self.max_iterations and change > self.tolerance and ssd2 > 1e-15
        ):

            iteration += 1
            ssd1 = ssd2

            # ---- 1. Diagonal weights (vector form, no diag matrix needed) ----
            u = np.abs(weights).flatten()

            # Early stop if weights collapse
            if np.all(u < 1e-12):
                break

            # ---- 2. Regularisation matrix Ic ----
            Ic = np.eye(n_features) * (self.alpha + beta * ssd1**2)

            # ---- 3. Build A = Ic + U G U  using vectorized diagonal mult ----
            # A[i,j] = Ic[i,j] + u[i] * G[i,j] * u[j]
            A = Ic + (u[:, None] * G * u[None, :])

            # ---- 4. b = U * HTy (diagonal mult) ----
            b = u[:, None] * HTy

            # ---- 5. Solve A w = b (avoid pinv) ----
            try:
                w_new = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # fallback to pseudo-inverse if A is nearly singular
                w_new = np.linalg.pinv(A) @ b

            # Multiply final U again: weights = U w_new
            weights = u[:, None] * w_new

            # Check for divergence
            if not np.all(np.isfinite(weights)):
                break

            # ---- 6. Compute error ----
            residuals = H @ weights - y
            ssd2 = np.linalg.norm(residuals) / np.sqrt(n_samples)

            # ---- 7. Convergence ----
            change = abs(ssd2 - ssd1) / max(ssd1, 1e-12)

        return weights, ssd2

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

    @property
    def coef_(self):
        """Get the coefficients. This property exists for scikit-learn compatibility."""
        return self.coefficients_

    def score(self, X, y, sample_weight=None):
        r2 = super().score(X, y, sample_weight)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        self.r2_ = r2
        self.adjusted_r2_ = adjusted_r2
        return r2
