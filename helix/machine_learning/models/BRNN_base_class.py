import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from helix.options.enums import ActivationFunctions, ProblemTypes


class BRNN_base(BaseEstimator, RegressorMixin, nn.Module):
    def __init__(
        self,
        hidden_units=25,
        activation="relu",
        max_iter=200,
        random_state=None,
        tolerance=1e-4,
        learning_rate=0.01,
        num_hidden_layers=2,
    ):
        """
        Base class for Bayesian Regularized Neural Network (BRNN).

        Args:
            hidden_units (int): Number of units in the hidden layers.
            activation (str): Activation function to use.
            max_iter (int): Maximum number of iterations.
            random_state (int): Random seed.
            tolerance (float): Tolerance for stopping criteria.
            learning_rate (float): Learning rate for the optimizer.
            num_hidden_layers (int): Number of hidden layers.
        """

        super().__init__()

        self.hidden_units = hidden_units
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers

    def _init_model(self, n_features, n_outputs):

        layers = []

        for i in range(self.num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(n_features, self.hidden_units))
            elif i == self.num_hidden_layers - 1:
                layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(self._get_activation(self.activation))

        layers.append(nn.Linear(self.hidden_units, n_outputs))
        self.model_ = nn.Sequential(*layers)

        for layer in self.model_:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.num_params_ = sum(p.numel() for p in self.model_.parameters())

    def _get_activation(self, activation):
        activations = {
            ActivationFunctions.ReLU: nn.ReLU(),
            ActivationFunctions.Tanh: nn.Tanh(),
            ActivationFunctions.Sigmoid: nn.Sigmoid(),
            ActivationFunctions.LeakyReLU: nn.LeakyReLU(),
        }
        if activation not in activations:
            raise ValueError("Unsupported activation function")
        return activations[activation]

    def _make_loss(self, problem_type):
        if problem_type == ProblemTypes.Regression:
            self.loss = nn.MSELoss()
        elif problem_type == ProblemTypes.Classification:
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    def _prepare_data(self, X, y):

        X, y = check_X_y(X, y, multi_output=False)
        self.n_features_in_, self.n_samples_ = X.shape[1], X.shape[0]
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X = self.x_scaler_.fit_transform(X)
        y = self.y_scaler_.fit_transform(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

    def fit(self):
        raise NotImplementedError

    def _get_eigenvalues(self, X_tensor, y_tensor):
        """Computes eigenvalues using Fisher Information Matrix (FIM) instead of Hessian."""
        outputs = self.model_(X_tensor)
        loss = self.loss(outputs, y_tensor)

        params = list(self.model_.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=True)
        fisher_matrix = torch.zeros(
            sum(p.numel() for p in params), sum(p.numel() for p in params)
        )

        grad_flat = torch.cat([g.view(-1) for g in grads])
        for i in range(len(grad_flat)):
            second_grads = torch.autograd.grad(grad_flat[i], params, retain_graph=True)
            fisher_matrix[i] = torch.cat([sg.view(-1) for sg in second_grads])

        eigenvalues = torch.linalg.eigvalsh(fisher_matrix).detach().numpy()
        return eigenvalues

    def _calculate_evidence(self, alpha, beta, eigenvalues, error, reg_term):
        N = len(eigenvalues)
        return (
            -0.5 * (beta * error + alpha * reg_term)
            - 0.5 * np.sum(np.log(alpha + beta * eigenvalues))
            + 0.5 * (N * np.log(alpha) + self.n_samples_ * np.log(beta))
        )

    def predict(self):
        raise NotImplementedError

    def predict_proba(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError
