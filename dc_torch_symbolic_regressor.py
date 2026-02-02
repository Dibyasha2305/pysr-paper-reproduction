"""
DeepChem TorchModel wrapper for SymbolicNet.

Provides DeepChem-compatible symbolic regression using PyTorch backend.

Examples
--------
>>> import numpy as np
>>> import deepchem as dc
>>> from dc_torch_symbolic_regressor import DCTorchSymbolicRegressor
>>>
>>> X = np.random.randn(100, 1)
>>> y = 3.0 * X + 0.5
>>> dataset = dc.data.NumpyDataset(X, y)
>>>
>>> model = DCTorchSymbolicRegressor()
>>> model.fit(dataset)
>>> preds = model.predict(dataset)
>>> model.get_equation()
"""

from typing import Tuple

import torch
import deepchem as dc
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss

from deepchem.models.optimizers import Adam



from torch_symbolic_net import SymbolicNet


class DCTorchSymbolicRegressor(TorchModel):
    """
    DeepChem TorchModel wrapper for SymbolicNet.

    Learns symbolic equation:

        y = a * x + b
    """

    def __init__(self, learning_rate: float = 0.05, batch_size: int = 200) -> None:

        net = SymbolicNet()

        super().__init__(
    model=net,
    loss=L2Loss(),
    optimizer=Adam(),


    learning_rate=learning_rate,
    batch_size=batch_size,
    output_types=["prediction"]
)



    def get_equation(self) -> Tuple[float, float]:
        """
        Return learned symbolic equation.

        Returns
        -------
        tuple of float
            (a, b) where equation is y = a * x + b
        """
        net: SymbolicNet = self.model
        return net.get_equation()
