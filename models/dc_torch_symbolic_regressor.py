"""
DeepChem TorchModel wrapper for multivariate SymbolicNet
with polynomial (x, x^2, x^3) expansions.
"""

from typing import Dict
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

    Learns symbolic equation:

        y = sum_i (a_i x_i + b_i x_i^2 + c_i x_i^3) + d
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 256
    ) -> None:

        # Dummy model (DeepChem requires model at init)
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
        Build real symbolic model once feature dimension is known.
        """
        n_features = dataset.X.shape[1]

        # Replace dummy with real symbolic model
        self.model = SymbolicNet(n_features)

        # Reset DeepChem internals
        self._built = False
        self._ensure_built()

    def fit(self, dataset, **kwargs):
        """
        Fit symbolic model.
        """
        self.build(dataset)
        return super().fit(dataset, **kwargs)

    def get_equation(self) -> Dict[str, np.ndarray]:
        """
        Return learned symbolic equation components.

        Returns
        -------
        dict with keys:
            linear, quadratic, cubic, bias
        """
        w_x, w_x2, w_x3, b = self.model.get_equation()

        return {
            "linear": np.array(w_x),
            "quadratic": np.array(w_x2),
            "cubic": np.array(w_x3),
            "bias": b
        }


