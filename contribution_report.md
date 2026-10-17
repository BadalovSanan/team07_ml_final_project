## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Badalov, Sanan         | BadalovSanan  | Team lead, exploratory data analysis (EDA), and core report coordination |
| Hasanli, Oghuz         | Oghuz20 | Decision Tree implementation (`src/decision_tree.py`) & tree experiment sweeps |
| Sadigov, Sadigov       | sadigsadikhov | Pegasos SVM implementation (`src/svm.py`) & kernelization / Random Fourier Features |
| Hasanli, Ali           | ali-hasanli | Pipeline orchestration (`src/run_all.py`), data cleaning pipeline, and evaluation metrics |

## Per-member summary
**Badalov, Sanan —** Led project coordination and exploratory data analysis (EDA) for Problem 1, generating pricing and geographic distribution plots. Assisted in structuring the IEEE report, compiling final benchmark outcomes, and organizing team presentation milestones.

**Hasanli, Oghuz —** Implemented the core Decision Tree algorithm from scratch in NumPy (`src/decision_tree.py`), supporting Gini impurity, Entropy with Information Gain, and MSE reduction for regression. Developed the tree hyperparameter sweep modules and verified split correctness against scikit-learn reference baselines.

**Sadigov, Sadigov —** Implemented the soft-margin linear Support Vector Machine from scratch using the Pegasos sub-gradient descent method (`src/svm.py`). Developed the Random Fourier Features (RFF) non-linear kernel approximation module, tracked weight norms and margin geometry, and tuned regularization parameters.

**Hasanli, Ali —** Designed and finalized the end-to-end pipeline orchestration script (`src/run_all.py`), ensuring clean logging and cross-validation execution without data leakage. Handled raw data cleaning, outlier removal, duplication checks, and integrated all model outputs into structured CSV/markdown tables under `report/`. Managed repository version control and tagging (`v1.0-final`).

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
