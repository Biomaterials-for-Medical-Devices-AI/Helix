import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BayesianNeuralNetwork(BaseEstimator, RegressorMixin):
    """Bayesian Regularised Neural Network following MacKay's evidence framework.

    Args:
        hidden_units (int): Number of neurons in the single hidden layer. Defaults to 10.
        activation (str): Activation function for the hidden layer. Must be one of:
            'relu', 'tanh', 'sigmoid'. Defaults to 'tanh'.
        max_iter (int): Maximum number of iterations. Defaults to 200.
        random_state (int, optional): Random seed for weight initialisation. Defaults to None.
        tolerance (float): Convergence tolerance for evidence maximisation. Defaults to 1e-4.

    Attributes:
        n_features_in_ (int): Number of features seen during fit.
        n_samples_ (int): Number of samples seen during fit.
        model_ (nn.Sequential): The PyTorch neural network model.
        evidence_ (float): The current evidence value.
    """

    def __init__(
        self,
        hidden_units=10,
        activation="tanh",
        max_iter=200,
        random_state=None,
        tolerance=1e-4,
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.tolerance = tolerance

    def _init_model(self, n_features, n_outputs):
        """Initialises the neural network model.

        Args:
            n_features (int): Number of input features.
            n_outputs (int): Number of output dimensions.
        """
        layers = [
            nn.Linear(n_features, self.hidden_units),
            self._get_activation(self.activation),
            nn.Linear(self.hidden_units, n_outputs),
        ]
        self.model_ = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Returns the activation function layer.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: The activation function layer.

        Raises:
            ValueError: If activation function is not supported.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError("Activation function not supported")

    def _calculate_evidence(self, alpha, beta, eigenvalues):
        """Calculates the evidence value.

        Args:
            alpha (float): Hyperparameter alpha.
            beta (float): Hyperparameter beta.
            eigenvalues (list): Eigenvalues of the Hessian matrix.

        Returns:
            float: The evidence value.
        """
        N = len(eigenvalues)
        evidence = -0.5 * (beta * self.error_ + alpha * self.regularisation_)
        evidence -= 0.5 * np.sum(np.log(alpha + beta * eigenvalues))
        evidence += 0.5 * (N * np.log(alpha) + self.n_samples_ * np.log(beta))
        return evidence

    def fit(self, X, y):
        """Fits the Bayesian neural network.

        Args:
            X (array-like): Input data.
            y (array-like): Target data.

        Returns:
            self: The fitted estimator.
        """
        X, y = check_X_y(X, y, multi_output=True)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self._init_model(self.n_features_in_, n_outputs)

        # Initialise hyperparameters
        alpha = 1.0
        beta = 1.0 / np.var(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        for iteration in range(self.max_iter):
            # Train network with current alpha and beta
            optimizer = optim.Adam(self.model_.parameters(), lr=0.01)
            for _ in range(10):  # Inner training loop
                optimizer.zero_grad()
                outputs = self.model_(X_tensor)
                self.error_ = nn.MSELoss()(outputs, y_tensor).item()
                self.regularisation_ = sum(
                    p.pow(2.0).sum() for p in self.model_.parameters()
                ).item()
                loss = 0.5 * (beta * self.error_ + alpha * self.regularisation_)
                loss.backward()
                optimizer.step()

            # Update alpha and beta using evidence framework
            gamma = sum(alpha / (alpha + beta * eigenvalue) for eigenvalue in self._get_eigenvalues())
            alpha = gamma / self.regularisation_
            beta = (self.n_samples_ - gamma) / self.error_

            # Check for convergence
            if iteration > 0:
                new_evidence = self._calculate_evidence(
                    alpha, beta, self._get_eigenvalues()
                )
                if abs(self.evidence_ - new_evidence) < self.tolerance:
                    break
                self.evidence_ = new_evidence
            else:
                self.evidence_ = self._calculate_evidence(
                    alpha, beta, self._get_eigenvalues()
                )

        return self

    def _get_eigenvalues(self):
        """Computes the eigenvalues of the Hessian matrix.

        Returns:
            list: Eigenvalues of the Hessian matrix.
        """
        # Compute eigenvalues of the Hessian (approximation)
        params = [p.detach().numpy().flatten() for p in self.model_.parameters()]
        hessian = np.cov(np.array(params))
        return np.linalg.eigvalsh(hessian)

    def predict(self, X):
        """Makes predictions using the fitted model.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Predictions.
        """
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model_(X_tensor).numpy()
        return predictions

    def score(self, X, y):
        """Computes the mean squared error.

        Args:
            X (array-like): Input data.
            y (array-like): Target data.

        Returns:
            float: Mean squared error.
        """
        predictions = self.predict(X)
        return -np.mean((predictions - y) ** 2)
