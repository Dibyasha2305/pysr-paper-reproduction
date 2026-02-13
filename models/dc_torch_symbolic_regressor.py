import torch
import deepchem as dc
import numpy as np

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from deepchem.models.optimizers import Adam

from models.torch_symbolic_net import SymbolicNet


class DCTorchSymbolicRegressor(TorchModel):
    """
    Stable symbolic regressor for molecular benchmarks.

    y = w1*x + w2*x^2 + w3*x^3 + b
    """

    def __init__(self, learning_rate=0.0003, batch_size=256):
        dummy = torch.nn.Linear(1, 1)

        super().__init__(
            model=dummy,
            loss=L2Loss(),
            optimizer=Adam(
                learning_rate=learning_rate,
                weight_decay=1e-3   # 🔴 stronger regularization
            ),
            batch_size=batch_size,
            output_types=["prediction"]
        )

        self.learning_rate = learning_rate

    # ----------------------------
    # build symbolic net
    # ----------------------------
    def build(self, dataset):
        n_features = dataset.X.shape[1]

        self.model = SymbolicNet(n_features)

        # rebuild optimizer
        self._built = False
        self._ensure_built()

    # ----------------------------
    # stable fit with clipping
    # ----------------------------
    def fit(self, dataset, **kwargs):
        self.build(dataset)

        # enable gradient clipping
        for p in self.model.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -5.0, 5.0)
            )

        return super().fit(dataset, **kwargs)

    # ----------------------------
    # equation extraction
    # ----------------------------
    def get_equation(self):
        eq = self.model.get_equation()

        if isinstance(eq, tuple):
            w, b = eq
            return {
                "linear": w,
                "quadratic": np.zeros_like(w),
                "cubic": np.zeros_like(w),
                "bias": b,
            }

        eq.setdefault("linear", None)
        eq.setdefault("quadratic", None)
        eq.setdefault("cubic", None)
        eq.setdefault("bias", 0.0)

        return eq

