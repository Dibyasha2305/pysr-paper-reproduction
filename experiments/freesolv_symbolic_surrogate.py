"""
Symbolic surrogate modeling on FreeSolv dataset.

Pipeline
--------
1. Load FreeSolv (MoleculeNet)
2. Featurize with RDKit descriptors
3. Train RandomForest baseline
4. Fit symbolic regression surrogate
5. Evaluate RMSE and inspect weights
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor


# -------------------------------------------------------
# Data Loader
# -------------------------------------------------------

def load_freesolv_rdkit():
    """
    Load FreeSolv with RDKit descriptors.

    Returns
    -------
    train, valid, test : dc.data.Dataset
    """
    featurizer = dc.feat.RDKitDescriptors()

    tasks, datasets, transformers = dc.molnet.load_freesolv(
        featurizer=featurizer,
        splitter="random"
    )

    train, valid, test = datasets
    return train, valid, test


# -------------------------------------------------------
# Baseline Model
# -------------------------------------------------------

def train_baseline(train, test):
    """
    Train RandomForest on true FreeSolv labels.
    """

    X_train, y_train = train.X, train.y.flatten()
    X_test, y_test = test.X, test.y.flatten()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=0
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, rmse


# -------------------------------------------------------
# Symbolic Surrogate
# -------------------------------------------------------

def train_symbolic_surrogate(train, test, baseline_model):
    """
    Train symbolic regressor to mimic baseline predictions.
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

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym_model, rmse


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    print("Loading FreeSolv with RDKit descriptors...")
    train, valid, test = load_freesolv_rdkit()

    print("Training baseline RandomForest...")
    baseline_model, baseline_rmse = train_baseline(train, test)
    print(f"Baseline RMSE (true): {baseline_rmse:.3f}")

    print("Training symbolic surrogate...")
    sym_model, sym_rmse = train_symbolic_surrogate(
        train, test, baseline_model
    )
    print(f"Symbolic surrogate RMSE (vs baseline predictions): {sym_rmse:.3f}")

    weights, bias = sym_model.get_equation()

    print("Top contributing features:")
    for i in np.argsort(np.abs(weights))[-5:][::-1]:
        print(f"  w{i}: {weights[i]:.4f}")

    print(f"Bias: {bias:.4f}")


if __name__ == "__main__":
    main()