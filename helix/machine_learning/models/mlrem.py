import numpy as np
import numpy.linalg.linalg as LA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y

class EMLinearRegression(RegressorMixin, BaseEstimator):
    """Multiple Linear Regression with Expectation-Maximisation.
    
    This implementation uses the EM algorithm for feature selection and weight optimisation.
    
    Parameters
    ----------
    alpha : float, default=0
        Regularisation parameter.
    beta : float, default=1
        EM algorithm scaling parameter.
    weight_threshold : float, default=1e-3
        Threshold for feature removal.
    max_iterations : int, default=300
        Maximum number of EM algorithm iterations.
    tolerance : float, default=0.01
        Convergence tolerance for relative change in SSD.
    """
    def __init__(self, alpha=0, beta=1, weight_threshold=1e-3, max_iterations=300, tolerance=0.01):
        self.alpha = alpha
        self.beta = beta
        self.weight_threshold = weight_threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Attributes set during fit
        self.coefficients_ = None
        self.intercept_ = None
        self.p_values_ = None
        self.active_features_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        """Fit the EM linear regression model.
        
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
            Fitted model
        """
        # Input validation
        X, y = check_X_y(X, y, y_numeric=True)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Store feature names
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(n_features)]
        self.feature_names_ = ['Intercept'] + list(feature_names)

        # Add intercept
        # Add intercept
        H = np.ones((n_samples, n_features + 1), float)
        H[:, 1:] = X
        active_features = list(range(n_features + 1))
        
        # Initial calculations
        HT = H.T
        HTy = HT @ y
        Ainv = LA.pinv(HT @ H)
        weights = Ainv @ HTy
        
        # Initialize variables
        ssd2 = 1.0  # Initial sum of squared differences
        n_active = len(active_features)

        # EM algorithm
        iteration = 0
        change = 1.0
        self.residuals_ = y - X @ weights

        while (iteration < self.max_iterations and 
               change > self.tolerance and 
               ssd2 > 1e-15 and 
               n_active > 5):
            iteration += 1
            n_active_prev = n_active
            active_features_prev = active_features.copy()
            
            # E-step
            U = np.eye(n_active_prev)
            Ic = np.eye(n_active_prev)
            
            iteration += 1
            ssd1 = ssd2
            
            # Update matrices
            U = U * np.abs(weights)
            Ic = Ic * (self.alpha + self.beta * ssd1 * ssd1)
            
            # M-step weight updates
            R1 = HT @ H @ U
            R2 = U @ R1
            R3 = LA.pinv(Ic + R2)
            R4 = U @ R3
            R5 = U @ HTy
            weights = R4 @ R5

            # Compute predictions and residuals
            predictions = H @ weights
            residuals = predictions - y
            ssd2 = np.sqrt(np.sum(residuals ** 2) / n_samples)
            
            # Check convergence
            change = 100 * np.abs(ssd2 - ssd1) / ssd1

            # Feature pruning
            n_active = 0
            for i in range(len(active_features_prev)):
                if np.abs(weights[i]) > self.weight_threshold:
                    n_active += 1
                    
            if n_active < len(active_features_prev):
                # Create reduced matrices
                weights_new = np.zeros((n_active, 1))
                H_new = np.zeros((n_samples, n_active))
                active_features = []
                k = -1
                
                for i in range(len(active_features_prev)):
                    if np.abs(weights[i]) > self.weight_threshold:
                        k += 1
                        weights_new[k] = weights[i]
                        H_new[:, k] = H[:, i]
                        active_features.append(active_features_prev[i])
                
                # Update matrices
                H = H_new
                weights = weights_new
                HT = H.T
                HTy = HT @ y

        # Store final results
        self.weights_ = weights
        self.intercept_ = weights[0, 0]
        self.coefficients_ = weights[1:].flatten()
        self.active_features_ = active_features[1:]
        self.active_feature_names_ = [self.feature_names_[i] for i in active_features]
        
        # Compute final predictions and statistics
        predictions = H @ weights
        self.residuals_ = predictions - y
        
        # Calculate p-values
        C = LA.pinv(HT @ H).diagonal()
        degrees_of_freedom = max(1, y.shape[0] - len(active_features))
        standard_error = np.sqrt(np.sum(self.residuals_ ** 2) / degrees_of_freedom)
        
        confidence_intervals = np.sqrt(C * standard_error**2).reshape(-1, 1)
        t_values = np.abs(weights / confidence_intervals)
        s = np.random.standard_t(len(active_features), size=10000)
        self.p_values_ = np.sum(s > t_values.reshape(-1, 1), axis=1) / 10000.0
        self.p_values_ = p_values
        return self

    def predict(self, X):
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        X = check_array(X)
        n_samples = X.shape[0]

        if self.active_features_ is not None:
            X = X[:, self.active_features_]

        X = np.hstack((np.ones((n_samples, 1)), X))
        predictions = X @ np.vstack(([self.intercept_], self.coefficients_.reshape(-1, 1)))
        return predictions.ravel()

    def score(self, X, y, sample_weight=None):
        r2 = super().score(X, y, sample_weight)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        self.r2_ = r2
        self.adjusted_r2_ = adjusted_r2
        return r2
