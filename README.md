_**DeepChem-Compatible Symbolic Models for MoleculeNet**_

This repository implements symbolic regression and classification models in PyTorch, integrated with DeepChem’s TorchModel API and evaluated on MoleculeNet benchmarks using scaffold splits.

The goal is to explore whether interpretable symbolic models can approach the performance of standard machine learning baselines (Random Forest) on molecular property prediction tasks while producing explicit mathematical formulas.


_**Motivation**_


DeepChem is transitioning toward PyTorch as its primary deep learning backend.
While neural architectures are well supported, symbolic / equation-based models remain largely external.

Symbolic models are valuable in chemistry because they:
- produce interpretable structure–property relationships
- reveal descriptor contributions
- enable scientific insight beyond black-box prediction

This project demonstrates how symbolic models can be:
- implemented in PyTorch
- wrapped as DeepChem TorchModel
- trained on MoleculeNet datasets
- evaluated against ML baselines
- used to recover human-readable formulas


_**Symbolic Model**_

The symbolic model learns an explicit nonlinear function over molecular descriptors:

logit(y) = b + Σ wi·xi + Σ qi·xi²

where:
xi = RDKit molecular descriptor
wi = linear coefficient
qi = quadratic coefficient
b = bias

For regression tasks:

y = b + Σ wi·xi + Σ qi·xi²

For classification:

p = sigmoid(logit)


_**DeepChem Integration**_

The model is implemented as:

_torch_symbolic_net.py_ → PyTorch symbolic network

_dc_torch_symbolic_regressor.py_ → DeepChem TorchModel wrapper

_dc_torch_symbolic_classifier.py _→ logistic symbolic model


This allows symbolic models to use:
- DeepChem datasets
- scaffold splitting
- NumpyDataset pipelines
- training loops
- evaluation metrics


_**MoleculeNet Regression Benchmarks**_

Datasets:

- ESOL (solubility)
- FreeSolv (hydration free energy)
- Lipophilicity (logD)

All experiments use:

- RDKit descriptors
- RandomForest feature selection
- quadratic expansion
- standardization
- scaffold split

**Results (RMSE)**

| Dataset       | RF    | Symbolic | Gap    |
| ------------- | ----- | -------- | ------ |
| ESOL          | 0.317 | 0.324    | +0.007 |
| FreeSolv      | 0.240 | 0.230    | −0.010 |
| Lipophilicity | 0.546 | 0.427    | −0.119 |

Symbolic regression matches or exceeds RF on FreeSolv and Lipophilicity and is nearly identical on ESOL.


_**MoleculeNet Classification Benchmarks**_

Datasets:

- BBBP (blood–brain barrier)
- BACE (binding affinity)

Model:

- symbolic logistic regression
- RDKit descriptors
- quadratic expansion
- scaffold split

Results (AUC)
| Dataset | RF    | Symbolic |
| ------- | ----- | -------- |
| BBBP    | 0.747 | 0.674    |
| BACE    | 0.796 | 0.702    |

Symbolic classifiers are reasonably close to RF, especially on BACE.



_**Recovered Symbolic Formulas**_

Example recovered symbolic classifiers:

**BBBP**
logit(BBBP) =
 0.68
 + 0.08·SlogP_VSA9
 + 0.06·SlogP_VSA1
 − 0.05·PEOE_VSA8
 + 0.19·fr_urea
 + 0.14·NumAmideBonds
 + 0.07·NumSaturatedHeterocycles²
 + 0.07·NumHeteroatoms²
 + 0.06·SlogP_VSA8²

**BACE**
logit(BACE) =
 0.33
 + 0.13·PEOE_VSA1
 + 0.11·PEOE_VSA11
 − 0.16·MolWt
 + 0.10·NumSaturatedCarbocycles
 + 0.09·MaxEStateIndex
 − 0.15·PEOE_VSA2²
 + 0.60·MaxEStateIndex²

These formulas show interpretable descriptor contributions with linear and nonlinear structure.


_**Repository Structure**_

src/
│
├── models/
│   ├── torch_symbolic_net.py
│   ├── dc_torch_symbolic_regressor.py
│   └── dc_torch_symbolic_classifier.py
│
├── experiments/
│   ├── molnet_symbolic_benchmark.py
│   └── molnet_symbolic_classification.py
│
└── outputs/
    ├── *_pred.png
    ├── *_roc.png
    ├── molnet_equations.txt
    └── molnet_classification_equations.txt


_**Installation**_

conda create -n sr_env python=3.10
conda activate sr_env
pip install torch deepchem scikit-learn matplotlib


_**Running Benchmarks**_

**Regression:**
python experiments/molnet_symbolic_benchmark.py

**Classification:**
python experiments/molnet_symbolic_classification.py

**Outputs are saved in:**
outputs/


_**Key Contributions**_

This project shows that:

- symbolic models integrate cleanly with DeepChem TorchModel
- interpretable formulas can approach RF performance
- nonlinear descriptor terms improve accuracy
- scaffold-split MoleculeNet benchmarks are feasible
- symbolic classifiers achieve competitive AUC


_**Future Directions**_

- richer operator libraries (exp, log, interactions)
- sparse symbolic selection
- hybrid symbolic-neural models
- uncertainty estimation
- larger MoleculeNet tasks
- integration into DeepChem core


**Status**

Draft research prototype for DeepChem symbolic modeling.

Prepared as part of symbolic modeling exploration under DeepChem PyTorch transition.
