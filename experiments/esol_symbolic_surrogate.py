"""
Symbolic surrogate modeling on ESOL dataset
using RDKit descriptors and polynomial symbolic regression.
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

def load_esol_rdkit():
    featurizer = dc.feat.RDKitDescriptors()

    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer=featurizer,
        splitter="random",
        reload=False
    )

    train, valid, test = datasets
    return train, valid, test


# --------------------------------------------------
# Feature expansion
# --------------------------------------------------

def add_polynomial_features(X):
    """
    Add quadratic (x^2) features.
    """
    return np.concatenate([X, X ** 2], axis=1)


# --------------------------------------------------
# Baseline model
# --------------------------------------------------

def train_baseline_model(train, test):
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
# Symbolic surrogate
# --------------------------------------------------

def train_symbolic_surrogate(train, test, baseline_model):

    # Polynomial expansion
    X_train = add_polynomial_features(train.X)
    X_test = add_polynomial_features(test.X)

    # Scale features (CRUCIAL for symbolic stability)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Surrogate learns baseline predictions
    y_train = baseline_model.predict(train.X)
    y_test = baseline_model.predict(test.X)

    train_ds = dc.data.NumpyDataset(X_train, y_train)
    test_ds = dc.data.NumpyDataset(X_test, y_test)

    sym_model = DCTorchSymbolicRegressor(
        learning_rate=0.005,   # lower LR for stability
        batch_size=256
    )

    sym_model.fit(train_ds, nb_epoch=1500)

    preds = sym_model.predict(test_ds).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return sym_model, rmse, scaler


# --------------------------------------------------
# Noise robustness
# --------------------------------------------------

def noise_experiment(train, test, baseline_model, scaler):

    print("\nNoise robustness experiment:")

    X_train = scaler.transform(add_polynomial_features(train.X))
    X_test = scaler.transform(add_polynomial_features(test.X))

    for sigma in [0.0, 0.1, 0.2, 0.3]:

        noisy_targets = baseline_model.predict(train.X)
        noisy_targets += sigma * np.random.randn(len(noisy_targets))

        train_ds = dc.data.NumpyDataset(X_train, noisy_targets)

        sym_model = DCTorchSymbolicRegressor(
            learning_rate=0.005,
            batch_size=256
        )

        sym_model.fit(train_ds, nb_epoch=1500)

        preds = sym_model.predict(
            dc.data.NumpyDataset(X_test, baseline_model.predict(test.X))
        ).flatten()

        rmse = np.sqrt(
            mean_squared_error(
                baseline_model.predict(test.X),
                preds
            )
        )

        print(f"  Noise σ={sigma}: RMSE = {rmse:.3f}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading ESOL with RDKit descriptors...")
    train, valid, test = load_esol_rdkit()

    print("Training baseline RandomForest...")
    baseline_model, baseline_rmse = train_baseline_model(train, test)
    print(f"Baseline RMSE (true): {baseline_rmse:.3f}")

    print("Training symbolic surrogate...")
    sym_model, sym_rmse, scaler = train_symbolic_surrogate(
        train, test, baseline_model
    )
    print(f"Symbolic surrogate RMSE (vs baseline): {sym_rmse:.3f}")

    # ----------------------------------------------
    # Inspect symbolic equation
    # ----------------------------------------------

    eq = sym_model.get_equation()

    lin = eq["linear"]
    quad = eq["quadratic"]
    cub = eq["cubic"]

    all_terms = np.concatenate([lin, quad, cub])

    labels = (
        [f"x{i}" for i in range(len(lin))] +
        [f"x{i}^2" for i in range(len(quad))] +
        [f"x{i}^3" for i in range(len(cub))]
    )

    print("Top contributing symbolic terms:")
    top_idx = np.argsort(np.abs(all_terms))[::-1][:5]

    for i in top_idx:
        print(f"  {labels[i]}: {all_terms[i]:.4f}")

    print(f"Bias: {eq['bias']:.4f}")

    # ----------------------------------------------
    # Noise robustness
    # ----------------------------------------------

    noise_experiment(train, test, baseline_model, scaler)


if __name__ == "__main__":
    main()
