import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import deepchem as dc
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from rdkit.Chem import Descriptors

from models.dc_torch_symbolic_regressor import DCTorchSymbolicRegressor


# --------------------------------------------------
# Descriptor names from RDKit
# --------------------------------------------------

RDKit_NAMES = [d[0] for d in Descriptors.descList]


# --------------------------------------------------
# Save ROC plot
# --------------------------------------------------

def save_roc_plot(name, y_true, rf_probs, sym_probs):
    os.makedirs("outputs", exist_ok=True)

    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_probs)
    fpr_sym, tpr_sym, _ = roc_curve(y_true, sym_probs)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr_rf, tpr_rf, label="RandomForest")
    plt.plot(fpr_sym, tpr_sym, label="Symbolic")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"outputs/{name}_roc.png", dpi=150)
    plt.close()


# --------------------------------------------------
# Save symbolic equation summary
# --------------------------------------------------

def save_equation(name, sym):
    eq = sym.get_equation()

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/molnet_classification_equations.txt", "a") as f:
        f.write(f"\n----- {name} -----\n")
        for k, v in eq.items():
            if k != "bias" and v is not None:
                f.write(f"{k}: {np.mean(np.abs(v)):.4f}\n")
        f.write(f"bias: {eq['bias']:.4f}\n")


# --------------------------------------------------
# Clean NaNs
# --------------------------------------------------

def clean_dataset(ds):
    mask = ~np.isnan(ds.X).any(axis=1)
    return dc.data.NumpyDataset(ds.X[mask], ds.y[mask])


# --------------------------------------------------
# Train models
# --------------------------------------------------

def train_models(name, train, test):

    train = clean_dataset(train)
    test = clean_dataset(test)

    if len(test.y) == 0:
        print(f"{name}: no valid samples after cleaning")
        return np.nan, np.nan

    # ---------------- RF ----------------
    rf = RandomForestClassifier(n_estimators=300, random_state=0)
    rf.fit(train.X, train.y.flatten())

    rf_probs = rf.predict_proba(test.X)[:, 1]
    y_test_rf = test.y.flatten()

    rf_auc = roc_auc_score(y_test_rf, rf_probs)

    # ---------------- Feature selection ----------------
    importances = rf.feature_importances_
    top_k = 12 if name == "BBBP" else 10
    idx = np.argsort(importances)[::-1][:top_k]

    X_train_sel = train.X[:, idx]
    X_test_sel = test.X[:, idx]

    # nonlinear expansion
    X_train_exp = np.concatenate([X_train_sel, X_train_sel**2], axis=1)
    X_test_exp = np.concatenate([X_test_sel, X_test_sel**2], axis=1)

    scaler = StandardScaler()
    X_train_exp = scaler.fit_transform(X_train_exp)
    X_test_exp = scaler.transform(X_test_exp)

    train_ds = dc.data.NumpyDataset(X_train_exp, train.y.flatten())
    test_ds = dc.data.NumpyDataset(X_test_exp, test.y.flatten())

    sym = DCTorchSymbolicRegressor(
        learning_rate=0.001,
        batch_size=128
    )

    sym.fit(train_ds, nb_epoch=2000)

    # ---------------- Human-readable formula ----------------
    eq = sym.get_equation()
    linear = eq.get("linear")
    quad = eq.get("quadratic")
    bias = eq.get("bias")

    selected_names = [RDKit_NAMES[i] for i in idx]

    terms = []

    for w, name_i in zip(linear, selected_names):
        if abs(w) > 0.05:
            sign = "+" if w > 0 else "−"
            terms.append(f"{sign} {abs(w):.2f}·{name_i}")

    for w, name_i in zip(quad, selected_names):
        if abs(w) > 0.05:
            sign = "+" if w > 0 else "−"
            terms.append(f"{sign} {abs(w):.2f}·{name_i}²")

    formula = f"logit({name}) = {bias:.2f} " + " ".join(terms)

    print("\nRecovered symbolic formula:")
    print(formula)

    # ---------------- Predictions ----------------
    logits = sym.predict(test_ds).flatten()
    probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))

    sym_auc = roc_auc_score(y_test_rf, probs)

    save_roc_plot(name, y_test_rf, rf_probs, probs)
    save_equation(name, sym)

    print(f"{name} RF AUC: {rf_auc:.3f}")
    print(f"{name} Symbolic AUC: {sym_auc:.3f}")

    return rf_auc, sym_auc


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    results = {}

    # BBBP
    _, ds, _ = dc.molnet.load_bbbp(
        featurizer=dc.feat.RDKitDescriptors(),
        splitter="scaffold",
        reload=False
    )
    train, valid, test = ds
    results["BBBP"] = train_models("BBBP", train, test)

    # BACE
    _, ds, _ = dc.molnet.load_bace_classification(
        featurizer=dc.feat.RDKitDescriptors(),
        splitter="scaffold",
        reload=False
    )
    train, valid, test = ds
    results["BACE"] = train_models("BACE", train, test)

    print("\n--- FINAL CLASSIFICATION ---")
    print("Dataset | RF AUC | Symbolic AUC")

    for k, (rf, sym) in results.items():
        print(f"{k:6s} {rf:.3f} {sym:.3f}")


if __name__ == "__main__":
    main()