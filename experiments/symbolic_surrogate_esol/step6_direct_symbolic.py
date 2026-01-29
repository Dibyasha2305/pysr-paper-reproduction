import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error

print("Loading arrays...")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Training direct symbolic regression model...")

model = PySRRegressor(
    niterations=200,
    population_size=200,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "cos"],
    maxsize=15,
    verbosity=1,
)

model.fit(X_train, y_train)

eq = model.sympy()

print("\nDiscovered equation:")
print(eq)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("\nDirect symbolic Test MSE:", mse)
print("STEP 6 COMPLETE")
