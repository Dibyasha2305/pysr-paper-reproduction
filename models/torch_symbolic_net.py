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


class SymbolicNet(torch.nn.Module):
    """
    Multivariate linear symbolic network:

        y = w1*x1 + w2*x2 + ... + wn*xn + b
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

    def get_equation(self):
        weights = self.linear.weight.detach().numpy().flatten()
        bias = self.linear.bias.item()

        return weights, bias


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

