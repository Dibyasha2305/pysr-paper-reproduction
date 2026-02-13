"""
ESOL symbolic surrogate with stabilized nonlinear expansion
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
# Dataset
# --------------------------------------------------

def load_esol():
    featurizer = dc.feat.RDKitDescriptors()

    _, datasets, _ = dc.molnet.load_delaney(
        featurizer=featurizer,
        splitter="random",
        reload=False
    )

    return datasets


# --------------------------------------------------
# Polynomial expansion
# --------------------------------------------------

def expand_features(X):
    return np.concatenate([X, X**2, X**3], axis=1)


# --------------------------------------------------
# Train RF
# --------------------------------------------------

def train_rf(train, test):
    X_train, y_train = train.X, train.y.flatten()
    X_test, y_test = test.X, test.y.flatten()

    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return rf, rmse


# --------------------------------------------------
# Train symbolic
# --------------------------------------------------

def train_symbolic(train, test, rf):

    X_train = expand_features(train.X)
    X_test = expand_features(test.X)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = rf.predict(train.X)
    y_test = rf.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym = DCTorchSymbolicRegressor(
        learning_rate=0.001,
        batch_size=256
    )

    sym.fit(train_ds, nb_epoch=3000)

    preds = sym.predict(test_ds).flatten()

    mask = ~np.isnan(preds)
    preds = preds[mask]
    y_test = y_test[mask]

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym, rmse


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading ESOL...")
    train, valid, test = load_esol()

    print("Training RF...")
    rf, rf_rmse = train_rf(train, test)
    print(f"RF RMSE: {rf_rmse:.3f}")

    print("Training symbolic...")
    sym, sym_rmse = train_symbolic(train, test, rf)
    print(f"Symbolic RMSE: {sym_rmse:.3f}")
    print(f"Gap: {sym_rmse - rf_rmse:.3f}")

    eq = sym.get_equation()

    linear = eq["linear"]
    quadratic = eq["quadratic"]
    cubic = eq["cubic"]

    all_terms = np.concatenate([linear, quadratic, cubic])
    labels = (
        [f"x{i}" for i in range(len(linear))] +
        [f"x{i}^2" for i in range(len(quadratic))] +
        [f"x{i}^3" for i in range(len(cubic))]
    )

    top = np.argsort(np.abs(all_terms))[::-1][:5]

    print("\nTop symbolic terms:")
    for i in top:
        print(f"{labels[i]}: {all_terms[i]:.4f}")

    print(f"Bias: {eq['bias']:.4f}")


if __name__ == "__main__":
    main()

