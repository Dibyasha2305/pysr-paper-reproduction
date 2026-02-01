import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# Friedman-2 dataset
# y = sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2)
# -----------------------------
np.random.seed(0)

n_samples = 1200
X = np.random.rand(n_samples, 4)

# Avoid division by zero
eps = 1e-3
x1 = X[:, 0]
x2 = X[:, 1] + eps
x3 = X[:, 2]
x4 = X[:, 3] + eps

y = np.sqrt(x1**2 + (x2 * x3 - 1.0 / (x2 * x4)) ** 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------
# PySR model (fast + interpretable)
# -----------------------------
model = PySRRegressor(
    niterations=150,
    population_size=150,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    complexity_of_constants=2,
    parsimony=1e-3,
    verbosity=0,
)

model.fit(X_train, y_train)

# -----------------------------
# Results
# -----------------------------
print("\nDiscovered equations (Pareto front):")
equations = model.sympy()

if isinstance(equations, (list, tuple)):
    for eq in equations:
        print(eq)
else:
    print(equations)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("\nTest MSE:", mse)
