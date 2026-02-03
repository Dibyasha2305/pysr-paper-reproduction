"""
Symbolic surrogate modeling on ESOL dataset.

Pipeline:
1. Load ESOL (MoleculeNet)
2. Featurize with RDKit descriptors
3. Train baseline ML model
4. Fit symbolic regression surrogate
5. Compare performance
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor


# -------------------------
# Dataset Loading
# -------------------------

def load_esol_rdkit():
    """
    Load ESOL dataset with RDKit descriptor featurizer.

    Returns
    -------
    train, valid, test : dc.data.Dataset
    """
    featurizer = dc.feat.RDKitDescriptors()

    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer=featurizer,
        splitter="random",
        reload=False
    )

    train, valid, test = datasets
    return train, valid, test


# -------------------------
# Baseline Model
# -------------------------

def train_baseline_model(train, test):
    """
    Train RandomForest baseline on RDKit descriptors.

    Returns
    -------
    model : RandomForestRegressor
    rmse : float
    preds : np.ndarray
    y_test : np.ndarray
    """
    X_train, y_train = train.X, train.y.flatten()
    X_test, y_test = test.X, test.y.flatten()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=0
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)



    return model, rmse, preds, y_test


# -------------------------
# Symbolic Surrogate
# -------------------------

def train_symbolic_surrogate(train, test, baseline_model):
    """
    Train symbolic regressor to mimic baseline predictions.

    Returns
    -------
    sym_model : DCTorchSymbolicRegressor
    rmse : float
    """
    X_train = train.X
    y_train = baseline_model.predict(train.X)

    X_test = test.X
    y_test = baseline_model.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym_model = DCTorchSymbolicRegressor(
        learning_rate=0.01,
        batch_size=256
    )

    sym_model.fit(train_ds, nb_epoch=2000)

    preds = sym_model.predict(test_ds).flatten()
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return sym_model, rmse
def noise_experiment(train, test, baseline_model):
    """
    Train symbolic surrogate under noisy targets.
    """

    noise_levels = [0.0, 0.1, 0.2, 0.3]

    print("\nNoise robustness experiment:")

    for sigma in noise_levels:
        noisy_preds = baseline_model.predict(train.X)
        noisy_preds = noisy_preds + sigma * np.random.randn(len(noisy_preds))

        train_ds = dc.data.NumpyDataset(train.X, noisy_preds)

        sym_model = DCTorchSymbolicRegressor(
            learning_rate=0.01,
            batch_size=256
        )

        sym_model.fit(train_ds, nb_epoch=2000)

        test_preds = sym_model.predict(
            dc.data.NumpyDataset(test.X, baseline_model.predict(test.X))
        ).flatten()

        mse = mean_squared_error(
            baseline_model.predict(test.X),
            test_preds
        )

        rmse = np.sqrt(mse)

        print(f"  Noise σ={sigma}: RMSE = {rmse:.3f}")


# -------------------------
# Main
# -------------------------

def main():
    print("Loading ESOL with RDKit descriptors...")
    train, valid, test = load_esol_rdkit()

    print("Training baseline RandomForest...")
    baseline_model, baseline_rmse, baseline_preds, y_test = train_baseline_model(
        train, test
    )
    np.save("outputs/baseline_preds.npy", baseline_preds)
    np.save("outputs/y_test.npy", y_test)

    print(f"Baseline RMSE (true): {baseline_rmse:.3f}")

    print("Training symbolic surrogate...")
    sym_model, sym_rmse = train_symbolic_surrogate(
        train, test, baseline_model
    )
    sym_preds = sym_model.predict(
    dc.data.NumpyDataset(test.X, baseline_model.predict(test.X))
).flatten()

    np.save("outputs/symbolic_preds.npy", sym_preds)

    print(f"Symbolic surrogate RMSE (vs baseline predictions): {sym_rmse:.3f}")

    weights, b = sym_model.get_equation()
    top_idx = np.argsort(np.abs(weights))[::-1][:5]

    print("Top contributing features:")
    for i in top_idx:
        print(f"  w{i}: {weights[i]:.4f}")

    print(f"Bias: {b:.4f}")
    noise_experiment(train, test, baseline_model)



if __name__ == "__main__":
    main()
