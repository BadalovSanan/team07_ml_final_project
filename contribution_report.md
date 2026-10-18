## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Badalov, Sanan         | BadalovSanan  | Team lead, Data cleaning pipeline (`src/data_prep.py`), evaluation metrics (`src/evaluate.py`), exploratory data analysis (EDA)  (`src/eda.py`), initial pipeline setup (`src/run_all.py`) & Ridge regression baseline (`src/ridge.py`),  & IEEE report writing (`report/report.tex`), tagged v1.0-final |
| Hasanli, Oghuz         | Oghuz20 | Decision Tree implementation (`src/decision_tree.py`) & tree experiment sweeps |
| Sadigov, Sadigov       | sadigsadikhov | Pegasos SVM implementation (`src/svm.py`) & kernelization / Random Fourier Features |
| Hasanli, Ali           | ali-hasanli | Evaluation & integration lead: sklearn baselines, cross-validation (mean ± std), timing, results tables, error analysis, finalized pipeline (`src/run_all.py`), README, requirements. |

## Per-member summary
**Badalov, Sanan —** Led project coordination and built the data cleaning pipeline for Problem 1 (`src/data_prep.py`), including leakage removal, the stratified split and the train-median tier label, and wrote the from-scratch evaluation metrics (`src/evaluate.py`). Generated the EDA figures (`src/eda.py`), created the initial `src/run_all.py`, and implemented the Ridge regression baseline (`src/ridge.py`) and wrote the IEEE report. Tagged v1.0-final.

**Hasanli, Oghuz —** Implemented the core Decision Tree algorithm from scratch in NumPy (`src/decision_tree.py`), supporting Gini impurity, Entropy with Information Gain, and MSE reduction for regression. Developed the tree hyperparameter sweep modules and verified split correctness against scikit-learn reference baselines.

**Sadigov, Sadigov —** Implemented the soft-margin linear Support Vector Machine from scratch using the Pegasos sub-gradient descent method (`src/svm.py`). Developed the Random Fourier Features (RFF) non-linear kernel approximation module, tracked weight norms and margin geometry, and tuned regularization parameters.

**Hasanli, Ali —** Extended and finalized the end-to-end pipeline script (`src/run_all.py`), ensuring clean logging and cross-validation (mean ± std) execution without data leakage. Implemented the scikit-learn baselines, fit/predict timing and the error analysis, and integrated all model outputs into structured CSV/markdown results tables under `report/`. Maintained the README and pinned requirements.

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Badalov, Sanan | 25% |
| Hasanli, Oghuz | 25% |
| Sadigov, Sadigov | 25% |
| Hasanli, Ali | 25% |

*Note: Workload was distributed evenly across data exploration, custom algorithm design, pipeline orchestration, and experimental evaluation.*
