"""
Symbolic surrogate modeling on Lipophilicity (Lipo) dataset.

Pipeline:
1. Load Lipo (MoleculeNet)
2. Featurize with RDKit descriptors
3. Train RandomForest baseline
4. Train symbolic surrogate to mimic baseline
5. Report RMSE + top symbolic features
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor

# --------------------------------------------------

def load_lipo_rdkit():
    """
    Load Lipophilicity dataset with RDKit descriptors.
    """
    featurizer = dc.feat.RDKitDescriptors()

    tasks, datasets, transformers = dc.molnet.load_lipo(
        featurizer=featurizer,
        splitter="random"
    )

    train, valid, test = datasets
    return train, valid, test

# --------------------------------------------------

def train_baseline_model(train, test):
    """
    Train RandomForest baseline.
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

# --------------------------------------------------

def train_symbolic_surrogate(train, test, baseline_model):
    """
    Train symbolic model to mimic baseline predictions.
    """
    X_train = train.X
    y_train = baseline_model.predict(train.X)

    X_test = test.X
    y_test = baseline_model.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym_model = DCTorchSymbolicRegressor(
        learning_rate=0.001,   # lower LR for stability
        batch_size=256
    )

    sym_model.fit(train_ds, nb_epoch=2000)

    preds = sym_model.predict(test_ds).flatten()

    # ---- Robust NaN handling ----
    mask = ~np.isnan(preds)

    if mask.sum() == 0:
        print("All symbolic predictions are NaN (training unstable).")
        return sym_model, np.inf

    preds = preds[mask]
    y_test = y_test[mask]

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym_model, rmse

# --------------------------------------------------

def main():

    print("Loading Lipophilicity with RDKit descriptors...")
    train, valid, test = load_lipo_rdkit()

    print("Training baseline RandomForest...")
    baseline_model, baseline_rmse = train_baseline_model(train, test)
    print(f"Baseline RMSE (true): {baseline_rmse:.3f}")

    print("Training symbolic surrogate...")
    sym_model, sym_rmse = train_symbolic_surrogate(
        train, test, baseline_model
    )

    if np.isinf(sym_rmse):
        print("Symbolic surrogate failed to converge.")
        return

    print(f"Symbolic surrogate RMSE (vs baseline predictions): {sym_rmse:.3f}")

    weights, bias = sym_model.get_weights()

    print("Top contributing features:")
    top_idx = np.argsort(np.abs(weights))[::-1][:5]

    for i in top_idx:
        print(f"  w{i}: {weights[i]:.4f}")

    print(f"Bias: {bias:.4f}")

# --------------------------------------------------

if __name__ == "__main__":
    main()
