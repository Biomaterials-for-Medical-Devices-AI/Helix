from pathlib import Path
from typing import Self, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

from helix.options.enums import ModelNames, OptimiserTypes, ProblemTypes
from helix.services.weights_init import kaiming_init, normal_init, xavier_init


class BaseBRNN(nn.Module):
    """
    This class is an abstract class for networks
    """

    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 1,
        lr: float = 0.0003,
        prior_mu: int = 0,
        prior_sigma: int = 1,
        lambda_reg: float = 0.01,
    ) -> None:
        """
        Initializes the BaseNetwork class
        """
        super().__init__()
        self._name = "BaseNetwork"
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.lambda_reg = lambda_reg

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

    def _make_optimizer(self, optimizer_type: OptimiserTypes, lr):
        """
        Creates and initializes the optimizer for the network.

        Args:
            optimizer_type (OptimiserTypes): The type of optimizer to use. Options are:
                - "Adam": Uses the Adam optimizer.
                - "SGD": Uses Stochastic Gradient Descent.
                - "RMSprop": Uses RMSprop optimizer.

            lr (float): The learning rate for the optimizer.

        Raises:
            NotImplementedError: If an unsupported `optimizer_type` is provided.
        """
        if optimizer_type == OptimiserTypes.Adam:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == OptimiserTypes.SGD:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer_type == OptimiserTypes.RMSprop:
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
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_X)

                # Compute total loss
                loss = self.compute_brnn_loss(
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

    def _bayesian_regularization_loss(self) -> torch.Tensor:
        """
        Compute the Bayesian Regularization loss.

        The loss is computed as the sum of squared differences
        between model parameters and their prior mean,
        scaled by the prior standard deviation.

        Returns:
            torch.Tensor: The computed regularization loss.
        """
        # Calculate regularization loss
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.sum((param - self.prior_mu) ** 2) / (
                2 * self.prior_sigma**2
            )
        return reg_loss

    def compute_brnn_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        problem_type: ProblemTypes,
    ) -> torch.Tensor:
        """
        Compute the total loss based on the problem type
        and include regularization loss.

        Args:
            outputs (torch.Tensor): The predicted outputs from the model.
            targets (torch.Tensor): The true target values.
            problem_type (ProblemTypes): The problem type.

        Returns:
            torch.Tensor: The total computed loss, including both
            predictive and regularization loss.
        """
        # Compute predictive loss
        predictive_loss = self._make_loss(problem_type, outputs, targets)

        # Compute regularization loss
        reg_loss = self._bayesian_regularization_loss(
            prior_mu=self.prior_mu, prior_sigma=self.prior_sigma
        )

        # Combine both losses
        total_loss = predictive_loss + self.lambda_reg * reg_loss
        return total_loss

    def save_model(self, destination: Path):
        """
        Saves the model's state dictionary to a file.
        """
        torch.save(self.state_dict(), destination)

    # define purely for help from the IDE
    def parameters(self, recurse=True):
        return super().parameters(recurse)


class BayesianRegularisedNNClassifier(ClassifierMixin, BaseEstimator, BaseBRNN):
    """
    This class defines a Bayesian Regularised Neural
    Network for classification tasks.

    Args:
        brnn_options (BrnnOptions): The Bayesian Regularised
        Neural Network options.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        batch_size=32,
        epochs=10,
        hidden_dim=64,
        output_dim=1,
        lr=0.0003,
        prior_mu=0,
        prior_sigma=1,
        lambda_reg=0.01,
        classification_cutoff=0.5,
    ):
        super().__init__(
            batch_size,
            epochs,
            hidden_dim,
            output_dim,
            lr,
            prior_mu,
            prior_sigma,
            lambda_reg,
        )
        self._name = ModelNames.BRNNClassifier
        self.classification_cutoff = classification_cutoff

    def _initialize_network(self, input_dim, output_dim):
        """
        Initializes the network layers based on the input
        and output dimensions.

        Args:
            input_dim (int): The input dimension of the data.
            output_dim (int): The output dimension of the
            data, determined dynamically.
        """
        # Define hidden layers and output layer
        self.layer1 = nn.Linear(input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)

        # Initialize weights and optimizer
        self._initialise_weights()
        self._get_num_params()
        self._make_optimizer(OptimiserTypes.Adam, self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output after applying the forward
            pass through the network.

        Raises:
            ValueError: If an error occurs during the forward pass.
        """
        try:
            x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
            x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
            x = self.output_layer(x)
            return torch.sigmoid(x) if x.size(1) == 1 else torch.softmax(x, dim=1)
        except Exception as e:
            raise ValueError(
                f"Error occured during forward pass of BRNN Classifier: {e}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Raises:
            ValueError: If an error occurs during training.

        Returns:
            Self: The trained model
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).squeeze().long().to(self.device)
        input_dim = X.shape[1]
        output_dim = len(torch.unique(y_tensor))

        try:
            self._initialize_network(input_dim, output_dim)
            self.train()  # set the underlying model to training mode
            self.train_brnn(X_tensor, y_tensor, ProblemTypes.Classification)
            return self
        except Exception as e:
            raise ValueError(f"Error occured during fitting of BRNN Classifier: {e}")

    def predict(self, X, return_probs=False) -> np.ndarray:
        """
        Predict the target values using the trained BRNN Regressor.

        Args:
            X (np.ndarray): The input data.
            return_probs (bool): Whether to return the predicted

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If an error occurs during prediction.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        try:
            self.eval()
            with torch.no_grad():
                outputs = self(X)

                if outputs.size(1) == 1:  # Binary classification
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    return (
                        probabilities
                        if return_probs
                        else (probabilities > self.classification_cutoff).astype(int)
                    )

                else:  # Multi-class classification
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    return (
                        probabilities
                        if return_probs
                        else np.argmax(probabilities, axis=1)
                    )

        except Exception as e:
            raise ValueError(f"Error occured during prediction of BRNN Classifier: {e}")


class BayesianRegularisedNNRegressor(RegressorMixin, BaseEstimator, BaseBRNN):
    """
    This class defines a Bayesian Regularised Neural
    Network for regression tasks.

    Args:
        brnn_options (BrnnOptions): The Bayesian Regularised
        Neural Network options.
    """

    def __init__(
        self,
        batch_size=32,
        epochs=10,
        hidden_dim=64,
        output_dim=1,
        lr=0.0003,
        prior_mu=0,
        prior_sigma=1,
        lambda_reg=0.01,
    ):
        super().__init__(
            batch_size,
            epochs,
            hidden_dim,
            output_dim,
            lr,
            prior_mu,
            prior_sigma,
            lambda_reg,
        )
        self._name = ModelNames.BRNNRegressor

    def _initialize_network(self, input_dim, output_dim):
        """
        Initializes the network layers for BRNN regression.

        Args:
            input_dim (int): The input dimension of the data.
            output_dim (int): The output dimension of the
            data, determined dynamically.
        """
        # Define hidden layers and output layer
        self.layer1 = nn.Linear(input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)

        # Initialize weights and optimizer
        self._initialise_weights()
        self._get_num_params()
        self._make_optimizer(OptimiserTypes.Adam, self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output after applying the forward
            pass through the network.

        Raises:
            ValueError: If an error occurs during the forward pass.
        """
        try:
            x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
            x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
            x = self.output_layer(x)
            return x
        except Exception as e:
            raise ValueError(
                f"Error occured during forward pass of BRNN Regressor: {e}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Raises:
            ValueError: If an error occurs during training.

        Returns:
            Self: The trained model
        """

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).squeeze().long()
        input_dim = X.shape[1]
        output_dim = 1

        try:
            self._initialize_network(input_dim, output_dim)
            self.train()
            self.train_brnn(X_tensor, y_tensor, ProblemTypes.Regression)
            return self
        except Exception as e:
            raise ValueError(f"Error occured during fitting of BRNN Regressor: {e}")

    def predict(self, X) -> np.ndarray:
        """
        Predict the target values using the trained BRNN Regressor.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If an error occurs during prediction.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        try:
            self.eval()
            with torch.no_grad():
                outputs = self(X)
                return outputs.cpu().numpy()
        except Exception as e:
            raise ValueError(f"Error occured during prediction of BRNN Regressor: {e}")
