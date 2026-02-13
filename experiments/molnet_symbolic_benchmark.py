"""
Stable MoleculeNet symbolic benchmark
with RF feature selection + polynomial expansion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor


# --------------------------------------------------
# Loaders
# --------------------------------------------------

def load_esol():
    _, ds, _ = dc.molnet.load_delaney(featurizer=dc.feat.RDKitDescriptors(), splitter="random", reload=False)
    return ds

def load_freesolv():
    _, ds, _ = dc.molnet.load_freesolv(featurizer=dc.feat.RDKitDescriptors(), splitter="random", reload=False)
    return ds

def load_lipo():
    _, ds, _ = dc.molnet.load_lipo(featurizer=dc.feat.RDKitDescriptors(), splitter="random", reload=False)
    return ds


# --------------------------------------------------
# RF + feature selection
# --------------------------------------------------

def train_rf(train, test, dataset_name):

    # dataset-specific feature count
    if dataset_name == "ESOL":
        top_k = 30
    elif dataset_name == "FreeSolv":
        top_k = 15
    else:  # Lipophilicity
        top_k = 10

    X_train = train.X
    y_train = train.y.flatten()

    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    X_train_sel = X_train[:, idx]
    X_test_sel = test.X[:, idx]

    preds = rf.predict(test.X)
    rmse = np.sqrt(mean_squared_error(test.y.flatten(), preds))

    return rf, rmse, idx, X_train_sel, X_test_sel

# --------------------------------------------------
# Polynomial expansion
# --------------------------------------------------

def expand(X):
    return np.concatenate([X, X**2, X**3], axis=1)


# --------------------------------------------------
# Symbolic training (stable)
# --------------------------------------------------

def train_symbolic(rf, idx, X_train_sel, X_test_sel, train, test):

    X_train = expand(X_train_sel)
    X_test = expand(X_test_sel)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = rf.predict(train.X)
    y_test = rf.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym = DCTorchSymbolicRegressor(
        learning_rate=0.001,
        batch_size=128
    )

    sym.fit(train_ds, nb_epoch=2000)

    preds = sym.predict(test_ds).flatten()

    mask = ~np.isnan(preds)
    if mask.sum() == 0:
        return np.nan

    preds = preds[mask]
    y_test = y_test[mask]

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


# --------------------------------------------------
# Run dataset
# --------------------------------------------------

def run_dataset(name, loader):

    train, valid, test = loader()

    rf, rf_rmse, idx, X_train_sel, X_test_sel = train_rf(train, test, name)


    sym_rmse = train_symbolic(rf, idx, X_train_sel, X_test_sel, train, test)

    print(f"\n------{name}------")
    print(f"RF RMSE: {rf_rmse:.3f}")
    print(f"Symbolic RMSE: {sym_rmse:.3f}")
    print(f"Gap: {sym_rmse - rf_rmse:.3f}")

    return rf_rmse, sym_rmse


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    results = {}

    results["ESOL"] = run_dataset("ESOL", load_esol)
    results["FreeSolv"] = run_dataset("FreeSolv", load_freesolv)
    results["Lipophilicity"] = run_dataset("Lipophilicity", load_lipo)

    print("\n------ FINAL TABLE ---------")
    print("Dataset | RF | Symbolic | Gap")

    for k, (rf, sym) in results.items():
        print(f"{k:12s} {rf:.3f} {sym:.3f} {sym-rf:.3f}")


if __name__ == "__main__":
    main()
