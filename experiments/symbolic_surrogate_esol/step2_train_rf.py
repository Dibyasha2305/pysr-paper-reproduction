# step2_train_rf.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("Loading saved data")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Training Random Forest model")

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

preds = rf.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("Random Forest Test MSE:", mse)
joblib.dump(rf, "rf_model.pkl")
np.save("rf_test_preds.npy", preds)

print("STEP 2 COMPLETE")

import joblib
joblib.dump(rf, "rf_model.pkl")
