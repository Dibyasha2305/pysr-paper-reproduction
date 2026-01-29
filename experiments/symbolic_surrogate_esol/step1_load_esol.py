# step1_load_esol.py

import deepchem as dc
import numpy as np
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

print("Loading ESOL dataset")

tasks, datasets, transformers = dc.molnet.load_delaney()

train_dataset, valid_dataset, test_dataset = datasets

def to_numpy(dataset):
    X = np.array(dataset.X)
    y = dataset.y.reshape(-1)
    return X, y

X_train, y_train = to_numpy(train_dataset)
X_valid, y_valid = to_numpy(valid_dataset)
X_test, y_test = to_numpy(test_dataset)

print("Train size:", X_train.shape[0])
print("Valid size:", X_valid.shape[0])
print("Test size :", X_test.shape[0])
print("Feature type:", type(X_train))
print("Target shape:", y_train.shape)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Save arrays
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_valid.npy", X_valid)
np.save("y_valid.npy", y_valid)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saved numpy arrays.")
print("STEP 1 COMPLETE")
