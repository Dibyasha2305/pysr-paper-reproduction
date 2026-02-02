"""
PyTorch symbolic regression network.

Implements a simple linear symbolic model:

    y = a * x + b

Examples
--------
>>> import torch
>>> from torch_symbolic_net import SymbolicNet, train_symbolic_model
>>> model = SymbolicNet()
>>> X = torch.randn(100, 1)
>>> y = 3.0 * X + 0.5
>>> train_symbolic_model(model, X, y)
>>> model.get_equation()
"""

from typing import Tuple
import torch
import torch.nn as nn


class SymbolicNet(nn.Module):
    """
    Simple symbolic regression network.

    Learns equation:

        y = a * x + b

    Attributes
    ----------
    linear : nn.Linear
        Linear layer with 1 input and 1 output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, 1).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n_samples, 1).
        """
        return self.linear(x)

    def get_equation(self) -> Tuple[float, float]:
        """
        Return learned symbolic equation parameters.

        Returns
        -------
        tuple of float
            (a, b) where equation is y = a * x + b
        """
        a = self.linear.weight.item()
        b = self.linear.bias.item()
        return a, b


def train_symbolic_model(
    model: SymbolicNet,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.05
) -> None:
    """
    Train symbolic regression model.

    Parameters
    ----------
    model : SymbolicNet
        Model to train.
    X : torch.Tensor
        Input tensor of shape (n_samples, 1).
    y : torch.Tensor
        Target tensor of shape (n_samples, 1).
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model = SymbolicNet()

    X = torch.randn(100, 1)
    y = 3.0 * X + 0.5

    train_symbolic_model(model, X, y)

    a, b = model.get_equation()
    print(f"Learned equation: y = {a:.3f} * x + {b:.3f}")

