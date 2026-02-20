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

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor


# --------------------------------------------------
# Loaders
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

def expand(X):
    X = np.asarray(X)
    X2 = X ** 2
    n = X.shape[1]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((X[:, i] * X[:, j])[:, None])

    X_pair = np.hstack(pairs) if pairs else None
    X_sqrt = np.sqrt(np.abs(X) + 1e-8)

    parts = [X, X2, X_sqrt]
    if X_pair is not None:
        parts.append(X_pair)

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
# Symbolic training (FIXED)
# --------------------------------------------------

def train_symbolic(name, rf, idx, X_train_sel, X_test_sel, train, test):

    # dataset-specific cubic strength
    if name == "ESOL":
        cubic_k = 10
    elif name == "FreeSolv":
        cubic_k = 3
    else:  # Lipophilicity
        cubic_k = 8

    X_train = expand(X_train_sel)
    X_test = expand(X_test_sel)



    scaler = StandardScaler(with_mean=True, with_std=True)

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

    # save plot
    save_prediction_plot(name, y_test, preds)

    # NaN safety
    mask = ~np.isnan(preds)
    if mask.sum() == 0:
        return sym, np.nan

    preds = preds[mask]
    y_test = y_test[mask]

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym, rmse


# --------------------------------------------------
# Run dataset (FIXED)
# --------------------------------------------------

def run_dataset(name, loader):

    train, valid, test = loader()

    rf, rf_rmse, idx, X_train_sel, X_test_sel = train_rf(train, test, name)

    sym, sym_rmse = train_symbolic(
        name, rf, idx, X_train_sel, X_test_sel, train, test
    )

    # Save equation
    try:
        eq = sym.get_equation()

        with open("outputs/molnet_equations.txt", "a") as f:
            f.write(f"\n----- {name} -----\n")
            for k, v in eq.items():
                if k != "bias" and v is not None:
                    f.write(f"{k}: {np.mean(np.abs(v)):.4f}\n")
            f.write(f"bias: {eq['bias']:.4f}\n")
    except Exception as e:
        print("Equation save failed:", e)

    print(f"\n------{name}------")
    print(f"RF RMSE: {rf_rmse:.3f}")
    print(f"Symbolic RMSE: {sym_rmse:.3f}")
    print(f"Gap: {sym_rmse - rf_rmse:.3f}")

    return rf_rmse, sym_rmse
def save_bar_plot(names, rf_vals, sym_vals):
    os.makedirs("outputs", exist_ok=True)

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, rf_vals, width, label="RandomForest")
    plt.bar(x + width/2, sym_vals, width, label="Symbolic")

    plt.xticks(x, names)
    plt.ylabel("RMSE")
    plt.title("MoleculeNet: RF vs Symbolic")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/molnet_bar.png", dpi=150)
    plt.close()


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

    rf_vals = []
    sym_vals = []
    names = []

    for k, (rf, sym) in results.items():
        print(f"{k:12s} {rf:.3f} {sym:.3f} {sym-rf:.3f}")
        rf_vals.append(rf)
        sym_vals.append(sym)
        names.append(k)

    # Save bar plot
    save_bar_plot(names, rf_vals, sym_vals)



if __name__ == "__main__":
    main()
