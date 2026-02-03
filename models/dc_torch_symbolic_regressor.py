"""
DeepChem TorchModel wrapper for multivariate SymbolicNet.
"""

import numpy as np
import torch
import deepchem as dc

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from deepchem.models.optimizers import Adam

from models.torch_symbolic_net import SymbolicNet


class DCTorchSymbolicRegressor(TorchModel):
    """
    DeepChem TorchModel wrapper for SymbolicNet.

    Learns equation:

        y = w1*x1 + ... + wn*xn + b
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 256
    ) -> None:

        # dummy model (will be replaced later)
        dummy = torch.nn.Linear(1, 1)

        super().__init__(
            model=dummy,
            loss=L2Loss(),
            optimizer=Adam(),
            learning_rate=learning_rate,
            batch_size=batch_size,
            output_types=["prediction"]
        )

    def build(self, dataset):
        """
        Build real model once number of features is known.
        """
        n_features = dataset.X.shape[1]
        self.model = SymbolicNet(n_features)

        # Let DeepChem set up optimizer & internal state
        self._built = False
        self._ensure_built()

    def fit(self, dataset, **kwargs):
        """
        Fit symbolic model.
        """
        self.build(dataset)
        return super().fit(dataset, **kwargs)

    def get_equation(self):
        """
        Return learned symbolic equation.

        Returns
        -------
        weights : np.ndarray
        bias : float
        """
        weights, bias = self.model.get_equation()
        return weights, bias
