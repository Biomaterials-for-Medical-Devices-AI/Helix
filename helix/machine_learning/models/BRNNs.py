from itertools import chain

import numpy as np
from scipy.special import xlogy
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.extmath import safe_sparse_dot


class BaseBRNNMixin:
    """Mixin class containing shared methods for BRNNRegressor and BRNNClassifier."""

    def _compute_reg_term(self):
        """Compute L2 regularization term."""
        total_params = sum(w.size for w in self.coefs_)
        return sum(np.sum(w**2) for w in self.coefs_) / total_params

    def _activation_derivative_func(self):
        """Return the derivative function for the activation."""
        from sklearn.neural_network._base import DERIVATIVES

        return DERIVATIVES[self.activation]

    def _compute_loss_grad(
        self, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        """Compute gradients for coefficients and intercepts with L2 penalty."""
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        total_params = sum(w.size for w in self.coefs_)
        coef_grads[layer] += self.beta_coef * self.coefs_[layer] / total_params
        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], 0)

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Backpropagation with adaptive regularization."""
        n_samples = X.shape[0]

        # Forward pass
        activations = self._forward_pass(activations)

        # Compute loss components
        data_loss = self._compute_data_loss(y, activations)
        reg_term = self._compute_reg_term()
        loss = self.alpha_loss * data_loss + self.beta_coef * reg_term

        # Compute output layer deltas
        self._compute_deltas(y, activations, deltas, n_samples)

        # Last layer gradients
        self._compute_loss_grad(
            self.n_layers_ - 2,
            n_samples,
            activations,
            deltas,
            coef_grads,
            intercept_grads,
        )

        # Backpropagate through hidden layers
        inplace_derivative = self._activation_derivative_func()
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        # Update alpha and beta adaptively
        Np = sum(p.size for p in chain(self.coefs_, self.intercepts_))
        g_diag = np.array([np.sum(p**2) for p in chain(self.coefs_, self.intercepts_)])
        trace_G_inv = np.sum(1.0 / (self.alpha_loss + g_diag + 1e-8))
        gamma = Np - self.alpha_loss * trace_G_inv
        self.alpha_loss = gamma / (2 * reg_term)
        self.beta_coef = (n_samples - gamma) / (2 * data_loss)

        return loss, coef_grads, intercept_grads

    def _compute_data_loss(self, y, activations):
        """Compute data-specific loss (to be implemented by subclasses)."""
        raise NotImplementedError

    def _compute_deltas(self, y, activations, deltas, n_samples):
        """Compute output layer deltas (to be implemented by subclasses)."""
        raise NotImplementedError


def log_loss(y_true, y_prob, sample_weight=None):
    """Log loss function."""
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)
    return -np.average(xlogy(y_true, y_prob), weights=sample_weight, axis=0).sum()


class BRNNRegressor(BaseBRNNMixin, MLPRegressor):
    """BRNN Regressor with RMSE + L2 loss."""

    def __init__(
        self,
        alpha_loss=0.01,
        beta_coef=100,
        hidden_layer_sizes=(100,),
        activation="relu",
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=None,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.alpha_loss = alpha_loss
        self.beta_coef = beta_coef
        self.model_name = "BRNNRegressor"

    def _compute_data_loss(self, y, activations):
        errors = activations[-1] - y
        mse = np.mean(errors**2)
        return np.sqrt(mse + 1e-8)

    def _compute_deltas(self, y, activations, deltas, n_samples):
        errors = activations[-1] - y
        rmse = self._compute_data_loss(y, activations)
        deltas[-1] = errors / (n_samples * rmse)


class BRNNClassifier(BaseBRNNMixin, MLPClassifier):
    """BRNN Classifier with log loss + L2 loss."""

    def __init__(
        self,
        alpha_loss=0.01,
        beta_coef=100,
        hidden_layer_sizes=(100,),
        activation="relu",
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=None,
        class_weight=None,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.alpha_loss = alpha_loss
        self.beta_coef = beta_coef
        self.model_name = "BRNNClassifier"
        self.class_weight = class_weight

    def _compute_data_loss(self, y, activations):
        y_pred = activations[-1]
        return log_loss(y, y_pred)

    def _compute_deltas(self, y, activations, deltas, n_samples):
        y_pred = activations[-1]
        deltas[-1] = (y_pred - y) / n_samples
