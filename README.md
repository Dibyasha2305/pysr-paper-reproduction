PySR Paper Reproduction & Symbolic ML Experiments with DeepChem

This repository contains:

A reproduction of key benchmark results from the PySR (Symbolic Regression) paper

Extensions exploring symbolic regression for interpretability of machine learning models, including experiments on DeepChem molecular datasets

The goal is to study how symbolic regression can recover known equations, discover interpretable structure in real datasets, and act as an interpretable surrogate for black-box models.

Part 1 : PySR Paper Reproduction
Benchmark Functions

Exact or near-exact symbolic recovery was achieved for standard Nguyen benchmarks:

Nguyen-1 → Nguyen-5

Example recovered form (Nguyen-5):

sin(x^2) * cos(x) - 1

Real-Valued Benchmarks

Friedman-1

Interpretable symbolic expressions capturing nonlinear feature interactions

Friedman-2

Harder rational/nonlinear dataset

Symbolic expressions recover meaningful structure even when exact recovery is difficult

These experiments demonstrate PySR’s ability to balance accuracy and symbolic simplicity.

Part 2 : Symbolic Surrogate Models for DeepChem (ESOL)

We explore whether symbolic regression can explain machine learning models trained on molecular data.

Pipeline

Load ESOL dataset from DeepChem

Train Random Forest regressor

Fit symbolic regression model to approximate RF predictions (surrogate)

Analyze discovered equations

Evaluate stability across random seeds

Compare against direct symbolic regression on labels

Key Findings

Symbolic surrogate achieves reasonable fidelity to the Random Forest model

Discovered expressions rely on a small subset of fingerprint features

The same feature indices appear across multiple random seeds

Direct symbolic regression on labels performs significantly worse than surrogate modeling

This suggests symbolic regression is more effective as an interpretable approximation layer than as a direct replacement for ML models on complex chemical data.

Repository Structure
.
├── nguyen5.py
├── friedman1_pysr.py
├── friedman2_pysr.py
├── experiments/
│   └── symbolic_surrogate_esol/
│       ├── step1_load_esol.py
│       ├── step2_train_rf.py
│       ├── step3_symbolic_surrogate.py
│       ├── step4_feature_analysis.py
│       ├── step5_stability.py
│       └── step6_direct_symbolic.py
└── outputs/

Tools Used

PySR (Python interface to SymbolicRegression.jl)

Julia backend via juliacall

DeepChem

NumPy, SciPy, scikit-learn

SymPy

Key Pints

Symbolic regression can:

Recover known analytical equations

Discover interpretable structure in synthetic benchmarks

Serve as a practical interpretability tool for chemical ML models

These results motivate deeper integration of symbolic machine learning inside DeepChem, ideally via native PyTorch-based symbolic modules.

