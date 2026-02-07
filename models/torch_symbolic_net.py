import torch
import torch.nn as nn


class SymbolicNet(nn.Module):
    """
    Symbolic regression network with polynomial feature expansion.

    φ(x) = [x, x^2, x^3]
    """

    def __init__(self, n_features: int):
        super().__init__()

        self.n_features = n_features
        self.expanded_features = 3 * n_features

        self.linear = nn.Linear(self.expanded_features, 1)

    def expand_features(self, x):
        """
        Polynomial feature expansion.
        """
        x1 = x
        x2 = x ** 2
        x3 = x ** 3
        return torch.cat([x1, x2, x3], dim=1)

    def forward(self, x):
        x_expanded = self.expand_features(x)
        return self.linear(x_expanded)

    def get_equation(self):
        """
        Return weights grouped by polynomial order.
        """
        W = self.linear.weight.detach().cpu().numpy().flatten()
        b = self.linear.bias.item()

        w_x = W[:self.n_features]
        w_x2 = W[self.n_features:2 * self.n_features]
        w_x3 = W[2 * self.n_features:3 * self.n_features]

        return w_x, w_x2, w_x3, b

