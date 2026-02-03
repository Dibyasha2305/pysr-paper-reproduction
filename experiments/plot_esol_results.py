import numpy as np
import matplotlib.pyplot as plt

# Load saved arrays
y_true = np.load("outputs/y_test.npy")
baseline_preds = np.load("outputs/baseline_preds.npy")
symbolic_preds = np.load("outputs/symbolic_preds.npy")

# --------------------------
# True vs Baseline
# --------------------------

plt.figure()
plt.scatter(y_true, baseline_preds, alpha=0.6)
plt.xlabel("True Solubility")
plt.ylabel("Baseline Predictions")
plt.title("ESOL: True vs Baseline")
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])
plt.tight_layout()
plt.savefig("outputs/true_vs_baseline.png")
plt.close()

# --------------------------
# Baseline vs Symbolic
# --------------------------

plt.figure()
plt.scatter(baseline_preds, symbolic_preds, alpha=0.6)
plt.xlabel("Baseline Predictions")
plt.ylabel("Symbolic Predictions")
plt.title("ESOL: Baseline vs Symbolic Surrogate")
plt.plot([baseline_preds.min(), baseline_preds.max()],
         [baseline_preds.min(), baseline_preds.max()])
plt.tight_layout()
plt.savefig("outputs/baseline_vs_symbolic.png")
plt.close()

print("Saved plots to outputs/")
