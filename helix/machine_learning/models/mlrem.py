import numpy as np
import numpy.linalg.linalg as LA
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
import csv

class EMLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0, beta=1, weight_threshold=1e-3, max_iterations=300, tolerance=0.01):
        self.alpha = alpha  # Regularisation parameter
        self.beta = beta  # Scaling parameter
        self.weight_threshold = weight_threshold  # Threshold for weight pruning
        self.max_iterations = max_iterations  # Maximum iterations for EM
        self.tolerance = tolerance  # Convergence tolerance
        self.coefficients_ = None  # Model coefficients
        self.intercept_ = None  # Intercept term
        self.p_values_ = None  # P-values for significance testing
        self.residuals_ = None  # Store residuals for later computations

    def fit(self, X, y):
        """
        Fit the multiple linear regression model using the Expectation-Maximisation (EM) algorithm.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Append a column of ones to X to account for the intercept
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        # Initialisation
        X_transpose = X.T
        Xt_y = X_transpose @ y
        inv_XtX = LA.pinv(X_transpose @ X)
        weights = inv_XtX @ Xt_y
        
        # Expectation-Maximisation Algorithm
        prev_residual_variance = 1.0
        change = 1.0
        iteration = 0
        active_features = list(range(n_features + 1))

        # Ensure residuals are initialised
        self.residuals_ = y - X @ weights 

        while iteration < self.max_iterations and change > self.tolerance and prev_residual_variance > 1e-15 and len(active_features) > 5:
            iteration += 1
            
            # Compute expected values
            U = np.eye(len(active_features)) * np.abs(weights)
            I_reg = np.eye(len(active_features)) * (self.alpha + self.beta * prev_residual_variance**2)
            
            # Update weights
            R1 = X_transpose @ X @ U
            R2 = U @ R1
            R3 = LA.pinv(I_reg + R2)
            weights = (U @ R3 @ U @ Xt_y)
            
            # Compute residual variance
            predictions = X @ weights
            self.residuals_ = y - predictions  # Store residuals
            residual_variance = np.sqrt((self.residuals_.T @ self.residuals_) / n_samples)
            change = 100 * np.abs(residual_variance - prev_residual_variance) / prev_residual_variance
            prev_residual_variance = residual_variance
            
            # Feature pruning based on weight threshold
            active_indices = [i for i in range(len(active_features)) if np.abs(weights[i]) > self.weight_threshold]
            if len(active_indices) < len(active_features):
                X = X[:, active_indices]
                X_transpose = X.T
                Xt_y = X_transpose @ y
                weights = weights[active_indices]
                active_features = [active_features[i] for i in active_indices]
        
        # Store final model parameters
        self.intercept_ = weights[0, 0]
        self.coefficients_ = weights[1:].flatten()
        
        # Compute standard errors and p-values
        C = LA.pinv(X_transpose @ X).diagonal()
        degrees_of_freedom = max(y.shape[0] - len(active_features), 1)
        standard_error = np.sqrt(np.sum((self.residuals_ * self.residuals_)) / degrees_of_freedom)
        confidence_intervals = np.sqrt(C * standard_error**2).reshape(-1, 1)
        t_values = np.abs(weights / confidence_intervals)
        p_values = np.sum(np.random.standard_t(len(active_features), size=10000) > t_values, axis=1) / 10000
        self.p_values_ = p_values
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Append a column of ones to X to account for the intercept
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        return X @ np.vstack(([self.intercept_], self.coefficients_.reshape(-1, 1)))
    
    def plot_coefficients(self):
        """
        Plot the beta coefficients of the best linear model found, 
        with positive coefficients in blue and negative ones in red.
        """
        if self.coefficients_ is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Coefficients: ", self.coefficients_)

        plt.figure(figsize=(8, 5))
        colors = ['blue' if coef > 0 else 'red' for coef in self.coefficients_]
        plt.bar(range(len(self.coefficients_)), self.coefficients_, color=colors)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.title("Beta Coefficients of EM Linear Regression")
        plt.show()

def save_coefficients(self, filename):
    """
    Save the coefficients found in self.coefficients_ to a csv file.
    """
    if self.coefficients_ is None or self.p_values_ is None:
        raise ValueError("Model has not been trained yet or p-values are not available.")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Beta coefficients", "p-value"])
        writer.writerows(zip(self.coefficients_, self.p_values_))
