import deepchem as dc
import numpy as np

print("Loading ESOL dataset with same featurizer...")

tasks, datasets, _ = dc.molnet.load_delaney(
    featurizer="ECFP",
    splitter="random",
    seed=42
)

train_dataset, _, _ = datasets

X = train_dataset.X

important_features = [119, 356, 561]

print("Important feature indices:", important_features)

for idx in important_features:
    active = np.mean(X[:, idx])
    print(f"Feature x{idx} average activation: {active:.4f}")

print("STEP 4 COMPLETE")
