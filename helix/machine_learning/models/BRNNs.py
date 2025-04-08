from itertools import chain

import numpy as np
from scipy.special import xlogy
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.extmath import safe_sparse_dot


class BRNNRegressor(MLPRegressor):
    """A wrapper for the MLPRegressor class to create a BRNN regressor."""

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
        """
        alpha_loss: float
            Weighting factor for RMSE in the custom loss function.

        beta_coef: float
            Weighting factor for L2 regularization (penalizes large weights).

        All other parameters are passed to sklearn's MLPRegressor.
        """

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.alpha_loss = alpha_loss  # Custom coefficient for RMSE contribution
        self.beta_coef = beta_coef  # Custom coefficient for L2 penalty
        self.model_name = "BRNNRegressor"

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        n_samples = X.shape[0]

        # Same as original: forward pass to get activations
        activations = self._forward_pass(activations)

        # === Custom Loss Function: RMSE + L2 Regularization ===

        # Difference from original:
        # The original code uses MSE (mean squared error) as the loss.
        # Here we compute RMSE (root of MSE) for better training stability
        # and combine it with an L2 regularization term.

        errors = activations[-1] - y
        mse_loss = np.mean(errors**2)
        rmse_loss = np.sqrt(mse_loss + 1e-8)  # Add epsilon to avoid division by zero

        # Difference from original:
        # Instead of averaging over samples directly,
        # we compute the average squared weights (L2 norm) normalized by total parameter count
        total_params = sum(w.size for w in self.coefs_)
        reg_term = sum(np.sum(w**2) for w in self.coefs_) / total_params

        # === Custom Loss Function: RMSE + L2 Regularization ===
        # We modify the original sklearn loss (MSE) to be:
        #   loss = alpha_loss * RMSE + beta_coef * L2
        # This encourages both accuracy (low RMSE) and weight sparsity (small weights).
        loss = self.alpha_loss * rmse_loss + self.beta_coef * reg_term

        # === Custom Gradient for RMSE Loss ===

        # Difference from original:
        # In the original, delta for last layer is just (activation - y),
        # i.e., the derivative of MSE. Here, we adjust it for RMSE:
        # d(RMSE)/dz = (1 / RMSE) * (prediction - y)
        deltas[-1] = errors / (n_samples * rmse_loss)

        # Same as original: compute gradient for the last layer
        self._compute_loss_grad(
            self.n_layers_ - 2,
            n_samples,
            activations,
            deltas,
            coef_grads,
            intercept_grads,
        )

        # Same as original: backprop through hidden layers
        inplace_derivative = self._activation_derivative_func()
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        # === Dynamic Update of Alpha and Beta (Optional Regularization Strategy) ===

        # Difference from original:
        # This line introduces sparsity-awareness:
        # gamma counts the number of near-zero parameters (approximating sparsity)
        gamma = sum(
            (np.abs(p) < 1e-6).sum() for p in chain(self.coefs_, self.intercepts_)
        )

        # These formulas encourage adaptive regularization.
        self.alpha_loss = gamma / (2 * reg_term)
        self.beta_coef = (n_samples - gamma) / (2 * rmse_loss)

        return loss, coef_grads, intercept_grads

    def _activation_derivative_func(self):
        """Returns the inplace derivative function for the selected activation"""
        from sklearn.neural_network._base import DERIVATIVES

        return DERIVATIVES[self.activation]

    def _compute_loss_grad(
        self, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        """Compute the gradient of loss with respect to coefs and intercept for specified layer."""
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])

        # Add the gradient of the regularization term (scaled by beta_coef)
        total_params = sum(w.size for w in self.coefs_)
        coef_grads[layer] += self.beta_coef * self.coefs_[layer] / total_params

        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], 0)


def log_loss(y_true, y_prob, sample_weight=None):
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)
    return -np.average(xlogy(y_true, y_prob), weights=sample_weight, axis=0).sum()


class BRNNClassifier(MLPClassifier):
    """A wrapper for the MLPClassifier class to create a BRNN classifier."""

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
        self.class_weight = class_weight
        self.model_name = "BRNNClassifier"

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        n_samples = X.shape[0]

        # Forward pass
        activations = self._forward_pass(activations)
        y_pred = activations[-1]

        # === Custom Loss Function: Log Loss + L2 Regularization ===
        total_params = sum(w.size for w in self.coefs_)
        reg_term = sum(np.sum(w**2) for w in self.coefs_) / total_params
        logloss = log_loss(y, y_pred)
        loss = self.alpha_loss * logloss + self.beta_coef * reg_term

        # === Gradient for Log Loss ===
        deltas[-1] = (y_pred - y) / n_samples

        # Output layer gradient
        self._compute_loss_grad(
            self.n_layers_ - 2,
            n_samples,
            activations,
            deltas,
            coef_grads,
            intercept_grads,
        )

        # Hidden layers gradient
        inplace_derivative = self._activation_derivative_func()
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        # Adaptive weighting of loss components
        gamma = sum(
            (np.abs(p) < 1e-6).sum() for p in chain(self.coefs_, self.intercepts_)
        )
        self.alpha_loss = gamma / (2 * reg_term)
        self.beta_coef = (n_samples - gamma) / (2 * logloss)

        return loss, coef_grads, intercept_grads

    def _activation_derivative_func(self):
        from sklearn.neural_network._base import DERIVATIVES

        return DERIVATIVES[self.activation]

    def _compute_loss_grad(
        self, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])

        # Add L2 regularization gradient
        total_params = sum(w.size for w in self.coefs_)
        coef_grads[layer] += self.beta_coef * self.coefs_[layer] / total_params

        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], 0)
