\# PySR Paper Reproduction



This repository contains a reproduction of results from the PySR (Symbolic Regression) paper.



\## Reproduced Results



\### Benchmark Reproduction

\- Nguyen-1 → Nguyen-5: Exact symbolic recovery achieved using PySR



\### Real Dataset Reproduction

\- Friedman-1: Interpretable symbolic expressions recovered, capturing nonlinear interactions

\- Friedman-2: Symbolic regression on a harder nonlinear and rational dataset, demonstrating interpretable structure discovery



\## Tools Used

\- PySR (Python interface to SymbolicRegression.jl)

\- Julia backend via `juliacall`

\- NumPy, SymPy, scikit-learn



\## Notes

\- Experiments were run with limited compute budgets for reproducibility.

\- Exact recovery is not expected for all real datasets; structural interpretability is the primary evaluation criterion.



