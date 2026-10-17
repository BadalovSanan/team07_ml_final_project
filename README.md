# MLE-AI-201 Machine Learning — Final Project (Team 07)

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** predict `price` (using `log(price)`).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines and Ridge regression only. For the SVM, we use the **Pegasos** sub-gradient method (see `src/svm.py`).

## Repo layout
.
├── README.md               # this file
├── requirements.txt        # pinned dependencies
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient) & RFF
│   ├── ridge.py            #   Custom Ridge regression
│   ├── evaluate.py         #   metrics + comparison helpers
│   ├── tree_experiments.py #   hyperparameter search for trees
│   ├── svm_experiments.py  #   hyperparameter search for SVMs
│   └── run_all.py          #   ONE command reproduces every number/figure
├── report/                 # IEEE report (self-contained)
│   ├── report.tex
│   └── figures/
└── presentation/           # your slides PDF


## Reproducibility (graded)
- Pinned dependencies in `requirements.txt`; all random seeds are fixed (`SEED = 42`).
- `python -m src.run_all` reproduces **every** headline number and figure from a clean checkout.
- Do **not** commit the raw data archive (see `.gitignore`).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put the dataset in data/ (see data/README.md), then:
python -m src.run_all
(Tip: Use python -m src.run_all --quick for a fast smoke test with a smaller data subset and fewer folds).