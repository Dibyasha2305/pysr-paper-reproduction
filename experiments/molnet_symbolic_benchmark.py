"""
Stable MoleculeNet symbolic benchmark
with RF feature selection + polynomial expansion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from models.dc_torch_symbolic_model import DCTorchSymbolicModel


# --------------------------------------------------
# Loaders (scaffold)
# --------------------------------------------------

def load_esol():
    _, ds, _ = dc.molnet.load_delaney(
        featurizer=dc.feat.RDKitDescriptors(),
        splitter="scaffold",
        reload=False
    )
    return ds


def load_freesolv():
    _, ds, _ = dc.molnet.load_freesolv(
        featurizer=dc.feat.RDKitDescriptors(),
        splitter="scaffold",
        reload=False
    )
    return ds


def load_lipo():
    _, ds, _ = dc.molnet.load_lipo(
        featurizer=dc.feat.RDKitDescriptors(),
        splitter="scaffold",
        reload=False
    )
    return ds


# --------------------------------------------------
# RF + feature selection
# --------------------------------------------------

def train_rf(train, test, dataset_name):

    if dataset_name == "ESOL":
        top_k = 12
    elif dataset_name == "FreeSolv":
        top_k = 8
    else:
        top_k = 6

    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(train.X, train.y.flatten())

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    X_train_sel = train.X[:, idx]
    X_test_sel = test.X[:, idx]

    preds = rf.predict(test.X)
    rmse = np.sqrt(mean_squared_error(test.y.flatten(), preds))

    return rf, rmse, idx, X_train_sel, X_test_sel


# --------------------------------------------------
# Nonlinear expansion
# --------------------------------------------------

def expand(X):
    X = np.asarray(X)
    X2 = X**2
    n = X.shape[1]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((X[:, i] * X[:, j])[:, None])

    parts = [X, X2]
    if pairs:
        parts.append(np.hstack(pairs))

    return np.concatenate(parts, axis=1)


# --------------------------------------------------
# Plot
# --------------------------------------------------

def save_prediction_plot(name, rf_preds, sym_preds):
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(rf_preds, sym_preds, alpha=0.6)

    mn = min(rf_preds.min(), sym_preds.min())
    mx = max(rf_preds.max(), sym_preds.max())
    plt.plot([mn, mx], [mn, mx], "--")

    plt.xlabel("RandomForest predictions")
    plt.ylabel("Symbolic predictions")
    plt.title(f"{name}: RF vs Symbolic")

    plt.tight_layout()
    plt.savefig(f"outputs/{name}_pred.png", dpi=150)
    plt.close()


# --------------------------------------------------
# Symbolic training
# --------------------------------------------------

def train_symbolic(name, rf, idx, X_train_sel, X_test_sel, train, test):

    X_train = expand(X_train_sel)
    X_test = expand(X_test_sel)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = rf.predict(train.X)
    y_test = rf.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym = DCTorchSymbolicModel(
        mode="regression",
        learning_rate=0.001,
        batch_size=128
    )

    sym.fit(train_ds, nb_epoch=2000)

    preds = sym.predict(test_ds).flatten()

    save_prediction_plot(name, y_test, preds)

    mask = ~np.isnan(preds)
    preds = preds[mask]
    y_test = y_test[mask]

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym, rmse


# --------------------------------------------------
# Run dataset
# --------------------------------------------------

def run_dataset(name, loader):

    train, valid, test = loader()

    rf, rf_rmse, idx, X_train_sel, X_test_sel = train_rf(train, test, name)

    sym, sym_rmse = train_symbolic(
        name, rf, idx, X_train_sel, X_test_sel, train, test
    )

    # save equation summary
    eq = sym.get_equation()
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/molnet_equations.txt", "a") as f:
        f.write(f"\n----- {name} -----\n")
        for k, v in eq.items():
            if k != "bias" and v is not None:
                f.write(f"{k}: {np.mean(np.abs(v)):.4f}\n")
        f.write(f"bias: {eq['bias']:.4f}\n")

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