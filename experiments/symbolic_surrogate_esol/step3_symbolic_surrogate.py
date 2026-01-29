import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("Loading saved arrays...")

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

print("Loading trained RF model...")
rf = joblib.load("rf_model.pkl")

print("Generating RF predictions...")
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)

print("Training symbolic surrogate (PySR)...")

model = PySRRegressor(
    niterations=300,
    population_size=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log"],
    maxsize=15,
    verbosity=1,
)

model.fit(X_train, y_train_rf)

print("\nDiscovered symbolic equations (Pareto front):")
equations = model.sympy()

if isinstance(equations, list):
    for eq in equations:
        print(eq)
else:
    print(equations)

print("\nEvaluating surrogate...")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test_rf, y_pred)

print("Surrogate fidelity MSE:", mse)
print("STEP 3 COMPLETE")
