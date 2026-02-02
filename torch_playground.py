"""
Simple PyTorch model example.

Demonstrates:
- nn.Module
- forward pass
- tensor shapes

Examples
--------
>>> import torch
>>> from torch_playground import SimpleNet
>>> model = SimpleNet()
>>> X = torch.randn(5, 3)
>>> y = model(X)
>>> y.shape
torch.Size([5, 1])
"""

from typing import Any
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    A minimal fully-connected neural network.

    Parameters
    ----------
    None

    Attributes
    ----------
    fc : nn.Linear
        Linear layer mapping 3 input features to 1 output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=True)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, 3).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n_samples, 1).
        """
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleNet()

    X = torch.randn(100, 1)
    y = 3.0 * X + 0.5


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(200):
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Final loss:", loss.item())
    weight = model.fc.weight.item()
    bias = model.fc.bias.item()

    print(f"Learned equation: y = {weight:.3f} * x + {bias:.3f}")


