from typing import Optional

import numpy as np
import deepchem as dc


class DCSymbolicRegressor(dc.models.Model):
    """
    Minimal DeepChem-native symbolic regressor.

    Implements a single-term symbolic model::

        y = a * x_i

    where ``a`` is a scalar coefficient and ``x_i`` is a selected feature.

    This class serves as a lightweight example of integrating symbolic-style
    models with DeepChem's ``Model`` API.
    """

    def __init__(self, n_iters: int = 1000, seed: int = 0, **kwargs) -> None:
        """
        Parameters
        ----------
        n_iters : int, default=1000
            Number of random search iterations.
        seed : int, default=0
            Random seed.
        **kwargs
            Extra keyword arguments passed to ``dc.models.Model``.
        """
        super().__init__(model_dir=None, **kwargs)
        self.n_iters: int = n_iters
        self.seed: int = seed

        self.best_coef: Optional[float] = None
        self.best_idx: Optional[int] = None
        self.best_loss: float = np.inf

    def fit(self, dataset: dc.data.Dataset, **kwargs) -> None:
        """
        Fit the symbolic regressor.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset with features ``X`` and labels ``y``.
        **kwargs
            Unused additional arguments.
        """
        X: np.ndarray = dataset.X
        y: np.ndarray = dataset.y.flatten()

        np.random.seed(self.seed)
        n_features: int = X.shape[1]

        for _ in range(self.n_iters):
            idx: int = np.random.randint(n_features)
            coef: float = float(np.random.randn())

            preds: np.ndarray = coef * X[:, idx]
            loss: float = float(np.mean((preds - y) ** 2))

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_coef = coef
                self.best_idx = idx

    def predict(self, dataset: dc.data.Dataset, **kwargs) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset containing input features.

        Returns
        -------
        np.ndarray
            Predictions with shape ``(n_samples, 1)``.
        """
        if self.best_coef is None or self.best_idx is None:
            raise ValueError("Model has not been fitted yet.")

        X: np.ndarray = dataset.X
        preds: np.ndarray = self.best_coef * X[:, self.best_idx]
        return preds.reshape(-1, 1)

    def get_expression(self) -> str:
        """
        Return the discovered symbolic expression.

        Returns
        -------
        str
            Symbolic expression string.
        """
        if self.best_coef is None or self.best_idx is None:
            raise ValueError("Model has not been fitted yet.")

        return f"{self.best_coef:.4f} * x{self.best_idx}"
