from pysr import PySRRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=3)

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "cos"]
)

model.fit(X, y)
print(model)
