import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    check_array,
    check_is_fitted,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from helix.options.enums import ActivationFunctions, ProblemTypes


class BRNNBase(nn.Module):
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


class BRNNRegressor(BaseEstimator, RegressorMixin, BRNNBase):
    def __init__(
        self,
        hidden_units=10,
        activation="relu",
        max_iter=200,
        random_state=None,
        tolerance=1e-4,
        learning_rate=0.01,
        num_hidden_layers=2,
    ):

        super().__init__(
            hidden_units=hidden_units,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
            tolerance=tolerance,
            learning_rate=learning_rate,
            num_hidden_layers=num_hidden_layers,
        )

    def fit(self, X, y):

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_tensor, y_tensor = self._prepare_data(X, y)

        self._init_model(self.n_features_in_, 1)

        alpha, beta = 0.01, 100.0

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self._make_loss(problem_type=ProblemTypes.Regression)

        self.train()

        for iteration in range(self.max_iter):

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model_(X_tensor)

            # Compute loss
            loss_mse = self.loss(outputs, y_tensor)
            loss_mse = torch.sqrt(loss_mse)

            # Regularisation term. Normalise by number of parameters to avoid overestimation
            reg_term = (
                sum(p.pow(2.0).sum() for p in self.model_.parameters())
                / self.num_params_
            )

            # this is the cost function from equation 17 of the paper
            loss = alpha * loss_mse + beta * reg_term

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Update error and regularisation terms
            self.error_ = loss_mse.item()
            self.regularisation_ = reg_term.item()

            # Compute eigenvalues of the Fisher Information Matrix
            eigenvalues = self._get_eigenvalues(
                X_tensor,
                y_tensor,
            )

            # Update of alpha, beta, and gamma is given by equation 28 in the paper
            gamma = sum(eig / (eig + alpha) for eig in eigenvalues)
            alpha = max(gamma / (2 * self.regularisation_), 1e-6)
            beta = max((self.n_samples_ - gamma) / (2 * self.error_), 1e-6)

            new_evidence = self._calculate_evidence(
                alpha, beta, eigenvalues, self.error_, self.regularisation_
            )
            if iteration > 0 and abs(self.evidence_ - new_evidence) < self.tolerance:
                break
            self.evidence_ = new_evidence

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self.x_scaler_.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            self.eval()
            y = self.model_(X_tensor).numpy()
            return self.y_scaler_.inverse_transform(y)

    def score(self, X, y):
        return -np.mean((self.predict(X) - y) ** 2)


class BRNNClassifier(BaseEstimator, ClassifierMixin, BRNNBase):
    def __init__(
        self,
        hidden_units=10,
        activation="sigmoid",
        max_iter=200,
        random_state=None,
        tolerance=1e-4,
        learning_rate=0.01,
        num_hidden_layers=2,
    ):

        super().__init__(
            hidden_units=hidden_units,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
            tolerance=tolerance,
            learning_rate=learning_rate,
            num_hidden_layers=num_hidden_layers,
        )

    def fit(self, X, y):

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.n_outputs_ = self._infere_num_categories(y)

        X_tensor, _ = self._prepare_data(X, y)
        y_tensor = torch.tensor(y, dtype=torch.long)

        self._init_model(self.n_features_in_, self.n_outputs_)

        alpha, beta = 0.01, 100.0

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self._make_loss(problem_type=ProblemTypes.Classification)

        self.train()

        for iteration in range(self.max_iter):

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model_(X_tensor)

            # Compute loss
            loss = self.loss(outputs, y_tensor)

            # Regularisation term. Normalise by number of parameters to avoid overestimation
            reg_term = (
                sum(p.pow(2.0).sum() for p in self.model_.parameters())
                / self.num_params_
            )

            # this is the cost function from equation 17 of the paper
            loss = alpha * loss + beta * reg_term

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Update error and regularisation terms
            self.error_ = loss.item()
            self.regularisation_ = reg_term.item()

            # Compute eigenvalues of the Fisher Information Matrix
            eigenvalues = self._get_eigenvalues(
                X_tensor,
                y_tensor,
            )

            # Update of alpha, beta, and gamma is given by equation 28 in the paper
            gamma = sum(eig / (eig + alpha) for eig in eigenvalues)
            alpha = max(gamma / (2 * self.regularisation_), 1e-6)
            beta = max((self.n_samples_ - gamma) / (2 * self.error_), 1)

            new_evidence = self._calculate_evidence(
                alpha, beta, eigenvalues, self.error_, self.regularisation_
            )
            if iteration > 0 and abs(self.evidence_ - new_evidence) < self.tolerance:
                break
            self.evidence_ = new_evidence

        return self

    def _infere_num_categories(self, y):
        return len(np.unique(y))

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self.x_scaler_.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            self.eval()
            y = self.model_(X_tensor).numpy()
            y = np.argmax(y, axis=1)
            return y

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self.x_scaler_.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            self.eval()
            y = self.model_(X_tensor).numpy()
            return y

    def score(self, X, y):
        return -np.mean((self.predict(X) - y) ** 2)
