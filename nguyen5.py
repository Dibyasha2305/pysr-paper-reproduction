import numpy as np
from pysr import PySRRegressor
import sympy as sp

# Nguyen-5 dataset
# y = sin(x^2) * cos(x) - 1
np.random.seed(0)

x = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(x[:, 0] ** 2) * np.cos(x[:, 0]) - 1.0

model = PySRRegressor(
    niterations=500,
    population_size=300,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "cos"],
    complexity_of_constants=2,
    parsimony=1e-3,
    verbosity=1
)

model.fit(x, y)

print("\nDiscovered equations (Pareto front):")
equations = model.sympy()

if isinstance(equations, (list, tuple)):
    for eq in equations:
        print(eq)
else:
    print(equations)

# Exact recovery check (symbolic equivalence)
print("\nExact recovery check:")
x_sym = sp.symbols("x0")
true_expr = sp.sin(x_sym**2) * sp.cos(x_sym) - 1

if isinstance(equations, (list, tuple)):
    for eq in equations:
        simplified = sp.simplify(eq - true_expr)
        print(eq, "->", simplified == 0)
else:
    simplified = sp.simplify(equations - true_expr)
    print(equations, "->", simplified == 0)
