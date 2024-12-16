import torch

from biofefi.options.ml import BrnnOptions


def bayesian_regularization_loss(
    model, prior_mu: float = None, prior_sigma: float = None
) -> torch.Tensor:
    """
    Compute the Bayesian Regularization loss.

    The loss is computed as the sum of squared differences
    between model parameters and their prior mean,
    scaled by the prior standard deviation.

    Args:
        prior_mu (float, optional): The prior mean. Defaults
        to `self._opt.prior_mu` if not provided.

        prior_sigma (float, optional): The prior standard deviation.
        Defaults to `self._opt.prior_sigma` if not provided.

    Returns:
        torch.Tensor: The computed regularization loss.

    Raises:
        ValueError: If both `prior_mu` and `prior_sigma` are not provided.
    """
    # Calculate regularization loss
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum((param - prior_mu) ** 2) / (2 * prior_sigma**2)
    return reg_loss


def compute_brnn_loss(
    model, outputs: torch.Tensor, targets: torch.Tensor, brnn_options: BrnnOptions
) -> torch.Tensor:
    """
    Compute the total loss based on the problem type
    and include regularization loss.

    Args:
        model (nn.Module): The neural network model.
        outputs (torch.Tensor): The predicted outputs from the model.
        targets (torch.Tensor): The true target values.

    Returns:
        torch.Tensor: The total computed loss, including both
        predictive and regularization loss.
    """
    # Compute predictive loss
    predictive_loss = model._make_loss(model.problem_type, outputs, targets)

    # Compute regularization loss
    reg_loss = bayesian_regularization_loss(
        model, prior_mu=brnn_options.prior_mu, prior_sigma=brnn_options.prior_sigma
    )

    # Combine both losses
    total_loss = predictive_loss + brnn_options.lambda_reg * reg_loss
    return total_loss
