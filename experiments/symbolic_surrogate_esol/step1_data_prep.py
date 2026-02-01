# step1_data_prep.py
import deepchem as dc
import numpy as np
from sklearn.preprocessing import StandardScaler

# Fix seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("Loading ESOL dataset...")

# Load ESOL dataset
tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True),
    splitter="random",
    seed=SEED,
)

train_dataset, valid_dataset, test_dataset = datasets


def dc_to_numpy(dataset):
    X = np.array([x.to_numpy() for x in dataset.X])
    y = dataset.y.reshape(-1)
    return X, y


# Convert to numpy
X_train, y_train = dc_to_numpy(train_dataset)
X_test, y_test = dc_to_numpy(test_dataset)

print("Raw shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save for next steps
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("STEP 1 COMPLETE")
