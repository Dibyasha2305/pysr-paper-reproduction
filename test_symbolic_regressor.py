import numpy as np
import deepchem as dc

from symbolic_regressor import DCSymbolicRegressor


def test_dc_symbolic_regressor_basic():
    """
    Basic integration test for DCSymbolicRegressor.
    """

    # Create simple linear dataset: y = 2 * x0
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    y = 2.0 * X[:, 0]

    dataset = dc.data.NumpyDataset(X, y)

    model = DCSymbolicRegressor(n_iters=200, seed=0)
    model.fit(dataset)

    preds = model.predict(dataset)

    # Shape check
    assert preds.shape == (100, 1)

    summary = model.summarize()

    # Key existence check
    assert "expression" in summary
    assert "mse" in summary

    # MSE should be finite
    assert summary["mse"] < 10.0
