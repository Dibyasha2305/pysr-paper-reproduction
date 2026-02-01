"""
Example: Symbolic surrogate modeling on ESOL using DeepChem.

This script demonstrates how to train the DeepChem-native
DCSymbolicRegressor on a reduced-feature version of the ESOL dataset.

Steps
-----
1. Load ESOL dataset
2. Select top-k features by variance
3. Fit symbolic regressor
4. Evaluate MSE

Run
---
python run_esol.py
"""

import numpy as np
import deepchem as dc

from symbolic_regressor import DCSymbolicRegressor


def main() -> None:
    """Run symbolic surrogate experiment on ESOL."""
    print("--- Script started ---")

    # Load ESOL (Delaney) dataset
    tasks, datasets, transformers = dc.molnet.load_delaney()
    train_dataset, valid_dataset, test_dataset = datasets

    print("Tasks:", tasks)
    print("Train X shape:", train_dataset.X.shape)
    print("Valid X shape:", valid_dataset.X.shape)

    # Feature reduction (variance-based)
    X = train_dataset.X
    variances = X.var(axis=0)

    top_k = 10
    top_idx = np.argsort(variances)[-top_k:]

    train_dataset = dc.data.NumpyDataset(
        train_dataset.X[:, top_idx], train_dataset.y, train_dataset.ids
    )

    valid_dataset = dc.data.NumpyDataset(
        valid_dataset.X[:, top_idx], valid_dataset.y, valid_dataset.ids
    )

    print("Reduced feature shape:", train_dataset.X.shape)

    # Train DeepChem-native symbolic model
    model = DCSymbolicRegressor(n_iters=2000, seed=0)
    model.fit(train_dataset)

    summary = model.summarize()
    print("Learned expression:", summary["expression"])

    # Evaluate using DeepChem API
    metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

    train_scores = model.evaluate(train_dataset, [metric])
    valid_scores = model.evaluate(valid_dataset, [metric])

    print("Train MSE:", train_scores["mean_squared_error"])
    print("Valid MSE:", valid_scores["mean_squared_error"])

    print("--- Done ---")

    summary = model.summarize()
    print("Summary:", summary)


if __name__ == "__main__":
    main()
