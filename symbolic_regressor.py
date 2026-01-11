import numpy as np
import deepchem as dc


class DCSymbolicRegressor(dc.models.Model):
    """
    DeepChem-native symbolic regression model.
    Single-term symbolic model: y = a * x_i
    """

    def __init__(self, n_iters=1000, seed=0, **kwargs):
        super().__init__(model_dir=None, **kwargs)
        self.n_iters = n_iters
        self.seed = seed
        self.best_coef = None
        self.best_idx = None
        self.best_loss = np.inf

    def fit(self, dataset, **kwargs):
        X = dataset.X
        y = dataset.y.flatten()

        np.random.seed(self.seed)
        n_features = X.shape[1]

        for _ in range(self.n_iters):
            idx = np.random.randint(n_features)
            coef = np.random.randn()

            preds = coef * X[:, idx]
            loss = ((preds - y) ** 2).mean()

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_coef = coef
                self.best_idx = idx

    def predict(self, dataset, transformers=None, **kwargs):
        X = dataset.X
        preds = self.best_coef * X[:, self.best_idx]
        return preds.reshape(-1, 1)

    def summarize(self):
        return {
            "expression": f"{self.best_coef:.4f} * x{self.best_idx}",
            "mse": self.best_loss
        }

