import numpy as np
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from helix.machine_learning.models.BRNN_base_class import BRNNBase
from helix.options.enums import ProblemTypes


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
