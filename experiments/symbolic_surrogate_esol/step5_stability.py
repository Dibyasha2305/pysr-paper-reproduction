import numpy as np
from pysr import PySRRegressor
import joblib

print("Loading data...")

X_train = np.load("X_train.npy")
rf = joblib.load("rf_model.pkl")
y_train_rf = rf.predict(X_train)

seeds = [0, 1, 2]

equations = []

for seed in seeds:
    print(f"\nRunning PySR with seed {seed}")
    model = PySRRegressor(
        niterations=150,
        population_size=150,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        random_state=seed,
        verbosity=0,
    )

    model.fit(X_train, y_train_rf)
    eq = model.sympy()
    print(eq)
    equations.append(str(eq))

print("\nCollected equations:")
for e in equations:
    print(e)

print("\nSTEP 5 COMPLETE")
