from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from helix.options.enums import OptimiserTypes, ProblemTypes
from helix.options.ml import BrnnOptions
from helix.services.weights_init import kaiming_init, normal_init, xavier_init


class BaseNetwork(nn.Module):
    """
    This class is an abstract class for networks
    """

    def __init__(self, brnn_options: BrnnOptions) -> None:
        """
        Initializes the BaseNetwork class
        """
        super().__init__()
        self._name = "BaseNetwork"
        self._brnn_options = brnn_options
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._name

    def _make_loss(
        self, problem_type: ProblemTypes, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the total loss based on the problem type.

        Args:
            problem_type (ProblemTypes): The type of problem (
                Regression or Classification
            ).
            outputs (torch.Tensor): The predicted outputs from the model.
            targets (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The computed loss.

        Raises:
            ValueError: If an unsupported problem type is specified.
        """
        if problem_type == ProblemTypes.Classification:

            # Binary classification
            # Ensure targets are float for BCE loss
            if outputs.size(1) == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                targets = targets.float()
                predictive_loss = loss_fn(outputs.squeeze(), targets)

            # Multi-class classification
            else:
                loss_fn = nn.CrossEntropyLoss()
                targets = targets.squeeze().long()
                predictive_loss = loss_fn(outputs, targets)

        # Regression
        # Ensure targets are float for MSE loss
        elif problem_type == ProblemTypes.Regression:
            loss_fn = nn.MSELoss()
            targets = targets.unsqueeze(-1).float()
            predictive_loss = loss_fn(outputs, targets)

        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        return predictive_loss

    def _initialise_weights(self, init_type: str = "normal") -> None:
        """
        Initializes the weights of the network based on the
        specified initialization type.

        Args:
            init_type (str): The type of weight initialization. Options are:
                - "normal": Uses normal distribution initialization.
                - "xavier_normal": Uses Xavier normal initialization.
                - "kaiming_normal": Uses Kaiming normal initialization.

        Raises:
            NotImplementedError: If an unsupported `init_type` is provided.
        """
        if init_type == "normal":
            self.apply(normal_init)
        elif init_type == "xavier_normal":
            self.apply(xavier_init)
        elif init_type == "kaiming_normal":
            self.apply(kaiming_init)
        else:
            raise NotImplementedError(f"Invalid init type: {init_type}")

    def _make_optimizer(self, optimizer_type, lr):
        """
        Creates and initializes the optimizer for the network.

        Args:
            optimizer_type (str): The type of optimizer to use. Options are:
                - "Adam": Uses the Adam optimizer.
                - "SGD": Uses Stochastic Gradient Descent.
                - "RMSprop": Uses RMSprop optimizer.

            lr (float): The learning rate for the optimizer.

        Raises:
            NotImplementedError: If an unsupported `optimizer_type` is provided.
        """
        if OptimiserTypes.Adam:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif OptimiserTypes.SGD:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif OptimiserTypes.RMSprop:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(
                f"Optimizer type {optimizer_type} not implemented"
            )

    def train_brnn(
        self, X: torch.Tensor, y: torch.Tensor, problem_type: ProblemTypes
    ) -> None:
        """
        Trains the Bayesian Regularized Neural Network.

        Args:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The target data.
            problem_type (ProblemTypes): The problem type.
        """

        self.train()

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._brnn_options.batch_size, shuffle=True
        )

        for _ in range(self._brnn_options.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_X)

                # Compute total loss
                loss = compute_brnn_loss(
                    self, outputs, batch_y, self._brnn_options, problem_type
                )
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        return self

    def __str__(self) -> str:
        """
        Returns the string representation of the network
        """
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Raises:
            NotImplementedError: If the forward pass method is
            not implemented in a subclass.
        """
        raise NotImplementedError

    def _get_num_params(self) -> Tuple[int, int]:
        """
        Returns the total number of parameters and the
        number of trainable parameters in the network.

        Returns:
            Tuple[int, int]: A tuple containing:
                - all_params (int): The total number of
                parameters in the network.
                - trainable_params (int): The total number
                of trainable parameters in the network.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return all_params, trainable_params

    def save_model(self, destination: Path):
        """
        Saves the model's state dictionary to a file.
        """
        torch.save(self.state_dict(), destination)

    # define purely for help from the IDE
    def parameters(self, recurse=True):
        return super().parameters(recurse)


def bayesian_regularization_loss(
    model: BaseNetwork,
    prior_mu: float = None,
    prior_sigma: float = None,
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
    model: BaseNetwork,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    brnn_options: BrnnOptions,
    problem_type: ProblemTypes,
) -> torch.Tensor:
    """
    Compute the total loss based on the problem type
    and include regularization loss.

    Args:
        model (nn.Module): The neural network model.
        outputs (torch.Tensor): The predicted outputs from the model.
        targets (torch.Tensor): The true target values.
        brnn_options (BrnnOptions): The options for the neural network.
        problem_type (ProblemTypes): The problem type.

    Returns:
        torch.Tensor: The total computed loss, including both
        predictive and regularization loss.
    """
    # Compute predictive loss
    predictive_loss = model._make_loss(problem_type, outputs, targets)

    # Compute regularization loss
    reg_loss = bayesian_regularization_loss(
        model, prior_mu=brnn_options.prior_mu, prior_sigma=brnn_options.prior_sigma
    )

    # Combine both losses
    total_loss = predictive_loss + brnn_options.lambda_reg * reg_loss
    return total_loss
