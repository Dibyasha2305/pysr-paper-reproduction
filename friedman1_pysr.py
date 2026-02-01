import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Friedman-1 dataset generation

np.random.seed(0)

n_samples = 1000
X = np.random.rand(n_samples, 5)

y = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# PySR model (paper-style)
model = PySRRegressor(
    niterations=100,
    population_size=100,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin"],
    complexity_of_constants=2,
    parsimony=1e-3,
    verbosity=0,
)


model.fit(X_train, y_train)


# Results

print("\nDiscovered equations (Pareto front):")
equations = model.sympy()

if isinstance(equations, (list, tuple)):
    for eq in equations:
        print(eq)
else:
    print(equations)


# Evaluate best model

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("\nTest MSE:", mse)
