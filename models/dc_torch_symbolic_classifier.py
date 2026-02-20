import torch
import numpy as np
import deepchem as dc

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import SigmoidCrossEntropy
from deepchem.models.optimizers import Adam

from models.torch_symbolic_net import SymbolicNet


class DCTorchSymbolicClassifier(TorchModel):
    """
    Symbolic classifier:
        p = sigmoid( w1*x + w2*x^2 + interactions + b )
    """

    def __init__(self, learning_rate=0.001, batch_size=128):

        dummy = torch.nn.Linear(1, 1)

        super().__init__(
            model=dummy,
            loss=SigmoidCrossEntropy(),
            optimizer=Adam(learning_rate=learning_rate, weight_decay=1e-4),
            batch_size=batch_size,
            output_types=["prediction"]
        )

    def build(self, dataset):
        n_features = dataset.X.shape[1]
        self.model = SymbolicNet(n_features)

        self._built = False
        self._ensure_built()

    def fit(self, dataset, **kwargs):
        self.build(dataset)
        return super().fit(dataset, **kwargs)

    def get_equation(self):
        eq = self.model.get_equation()

        if isinstance(eq, tuple):
            w, b = eq
            return {"linear": w, "bias": b}

        eq.setdefault("linear", None)
        eq.setdefault("quadratic", None)
        eq.setdefault("bias", 0.0)
        return eq
