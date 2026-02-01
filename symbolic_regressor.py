from typing import Dict, Optional

import numpy as np
import deepchem as dc


class DCSymbolicRegressor(dc.models.Model):
    """
    DeepChem-native symbolic regression model.

    This model implements a very simple single-term symbolic regressor of
    the form::

        y = a * x_i

    where ``a`` is a scalar coefficient and ``x_i`` is one selected input
    feature. The model searches over random coefficients and feature indices
    to minimize mean squared error.

    Notes
    -----
    This class is intended as a minimal prototype to demonstrate how
    symbolic-style models can be integrated into DeepChem's ``Model`` API.

        Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from symbolic_regressor import DCSymbolicRegressor
    >>>
    >>> X = np.random.randn(50, 3)
    >>> y = 2.0 * X[:, 0]
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>>
    >>> model = DCSymbolicRegressor(n_iters=500, seed=0)
    >>> model.fit(dataset)
    >>> preds = model.predict(dataset)
    >>> summary = model.summarize()
    >>> print(summary["expression"])

    """

    def __init__(self, n_iters: int = 1000, seed: int = 0, **kwargs) -> None:
        """
        Parameters
        ----------
        n_iters : int, default=1000
            Number of random search iterations.
        seed : int, default=0
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments passed to ``dc.models.Model``.
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
            DeepChem dataset containing features ``X`` and targets ``y``.
        **kwargs
            Unused additional arguments.

        Returns
        -------
        None
        """
        X: np.ndarray = dataset.X
        y: np.ndarray = dataset.y.flatten()

        np.random.seed(self.seed)
        n_features: int = X.shape[1]

        for _ in range(self.n_iters):
            idx: int = np.random.randint(n_features)
            coef: float = float(np.random.randn())

            preds: np.ndarray = coef * X[:, idx]
            loss: float = float(((preds - y) ** 2).mean())

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_coef = coef
                self.best_idx = idx

    def predict(
        self, dataset: dc.data.Dataset, transformers=None, **kwargs
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset containing input features.
        transformers : list, optional
            Unused (for API compatibility).
        **kwargs
            Unused additional arguments.

        Returns
        -------
        np.ndarray
            Predicted values with shape ``(n_samples, 1)``.
        """
        if self.best_coef is None or self.best_idx is None:
            raise ValueError("Model has not been fitted yet.")

        X: np.ndarray = dataset.X
        preds: np.ndarray = self.best_coef * X[:, self.best_idx]
        return preds.reshape(-1, 1)

    def summarize(self) -> Dict[str, float]:
        """
        Return a summary of the discovered symbolic expression.

        Returns
        -------
        dict
            Dictionary containing expression string and MSE.
        """
        if self.best_coef is None or self.best_idx is None:
            raise ValueError("Model has not been fitted yet.")

        return {
            "expression": f"{self.best_coef:.4f} * x{self.best_idx}",
            "mse": self.best_loss,
        }
