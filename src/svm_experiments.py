"""
svm_experiments.py -- Problem 4b: all SVM sweeps, figures and weight analysis.

Run from the repo root AFTER person A's data_prep.py works:
    python -m src.svm_experiments

What it does (all with a fixed seed, using TRAIN for fitting and VALIDATION
for choosing hyper-parameters -- the TEST set is never touched here):
  1. lambda sweep for the LINEAR Pegasos SVM
       -> margin width, #support vectors, train/val F1
  2. (gamma, lambda) sweep for the Random-Fourier-Features SVM
  3. convergence plot (objective vs epoch) for the best models
  4. top positive / negative weights of the best linear SVM
Results are saved to report/figures/ (png) and report/svm_results.csv.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_prep
from .svm import PegasosSVM, RandomFourierFeatures

FIG_DIR = "report/figures"
SEED = 42
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_EPOCHS = 20
RFF_COMPONENTS = 300


# ---------------------------------------------------------------- tiny metrics
def f1_score(y_true, y_pred):
    """F1 of the positive class (1 = premium), computed by hand."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return float(2 * p * r / (p + r))


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


# ------------------------------------------------------------------ load data
def load_task_b_data():
    """Use A's functions to build standardized splits + tier labels (Task B)."""
    df = data_prep.clean(data_prep.load_raw())
    X, y_price, names = data_prep.make_features(df)
    X_tr, p_tr, X_val, p_val, X_te, p_te = data_prep.train_val_test_split(X, y_price)
    y_tr, thr = data_prep.make_tier_label(p_tr)            # threshold from TRAIN only
    y_val, _ = data_prep.make_tier_label(p_val, thr)
    y_te, _ = data_prep.make_tier_label(p_te, thr)
    X_tr, X_val, X_te = data_prep.standardize(X_tr, X_val, X_te)
    return X_tr, y_tr, X_val, y_val, X_te, y_te, list(names)


# ------------------------------------------------------------------ sweeps
def sweep_linear(X_tr, y_tr, X_val, y_val):
    rows = []
    for lam in LAMBDAS:
        t = time.time()
        m = PegasosSVM(lambda_=lam, n_epochs=N_EPOCHS, random_state=SEED).fit(X_tr, y_tr)
        rows.append(dict(
            model="linear", lambda_=lam, gamma=np.nan,
            weight_norm=m.weight_norm(), margin=m.margin_width(),
            n_sv=int(m.support_vector_mask(X_tr, y_tr).sum()),
            train_f1=f1_score(y_tr, m.predict(X_tr)),
            val_f1=f1_score(y_val, m.predict(X_val)),
            val_acc=accuracy(y_val, m.predict(X_val)),
            final_obj=m.history_[-1], seconds=time.time() - t, _model=m))
        print(f"[linear] lambda={lam:<7g} margin={rows[-1]['margin']:8.3f} "
              f"#SV={rows[-1]['n_sv']:6d} trainF1={rows[-1]['train_f1']:.3f} "
              f"valF1={rows[-1]['val_f1']:.3f}")
    return rows


def sweep_rff(X_tr, y_tr, X_val, y_val):
    d = X_tr.shape[1]
    gammas = [g / d for g in (1.0, 2.0, 4.0, 8.0)]   # gamma relative to 1/#features
    rows = []
    for g in gammas:
        rff = RandomFourierFeatures(gamma=g, n_components=RFF_COMPONENTS, random_state=SEED).fit(X_tr)
        Z_tr, Z_val = rff.transform(X_tr), rff.transform(X_val)
        for lam in LAMBDAS:
            t = time.time()
            m = PegasosSVM(lambda_=lam, n_epochs=N_EPOCHS, random_state=SEED).fit(Z_tr, y_tr)
            rows.append(dict(
                model="rff", lambda_=lam, gamma=g,
                weight_norm=m.weight_norm(), margin=m.margin_width(),
                n_sv=int(m.support_vector_mask(Z_tr, y_tr).sum()),
                train_f1=f1_score(y_tr, m.predict(Z_tr)),
                val_f1=f1_score(y_val, m.predict(Z_val)),
                val_acc=accuracy(y_val, m.predict(Z_val)),
                final_obj=m.history_[-1], seconds=time.time() - t, _model=m))
            print(f"[rff] gamma={g:.4f} lambda={lam:<7g} trainF1={rows[-1]['train_f1']:.3f} "
                  f"valF1={rows[-1]['val_f1']:.3f}")
    return rows


# ------------------------------------------------------------------ outputs
def save_csv(rows, path="report/svm_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("Saved", path)


def plot_lambda_sweep(lin_rows):
    lam = [r["lambda_"] for r in lin_rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogx(lam, [r["margin"] for r in lin_rows], marker="o")
    ax[0].set(title="Margin width 2/||w|| vs lambda", xlabel="lambda", ylabel="margin width")
    ax[1].semilogx(lam, [r["n_sv"] for r in lin_rows], marker="o", color="tab:green")
    ax[1].set(title="Number of support vectors vs lambda", xlabel="lambda", ylabel="# SV")
    ax[2].semilogx(lam, [r["train_f1"] for r in lin_rows], marker="o", label="train F1")
    ax[2].semilogx(lam, [r["val_f1"] for r in lin_rows], marker="s", label="validation F1")
    ax[2].set(title="F1 vs lambda (linear SVM)", xlabel="lambda", ylabel="F1")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svm_lambda_sweep.png", dpi=130)
    plt.close()


def plot_rff_heatmap(rff_rows):
    gammas = sorted({r["gamma"] for r in rff_rows})
    grid = np.array([[next(r["val_f1"] for r in rff_rows
                           if r["gamma"] == g and r["lambda_"] == l) for l in LAMBDAS]
                     for g in gammas])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS])
    ax.set_yticks(range(len(gammas)), [f"{g:.3f}" for g in gammas])
    ax.set(title="RFF SVM: validation F1", xlabel="lambda", ylabel="gamma")
    for i in range(len(gammas)):
        for j in range(len(LAMBDAS)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svm_rff_heatmap.png", dpi=130)
    plt.close()


def plot_convergence(best_lin, best_rff):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(1, len(best_lin["_model"].history_) + 1), best_lin["_model"].history_,
            marker="o", label=f"linear (lambda={best_lin['lambda_']:g})")
    ax.plot(range(1, len(best_rff["_model"].history_) + 1), best_rff["_model"].history_,
            marker="s", label=f"RFF (lambda={best_rff['lambda_']:g})")
    ax.set(title="Pegasos convergence: objective J per epoch", xlabel="epoch", ylabel="J(w, b)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svm_convergence.png", dpi=130)
    plt.close()


def top_weights(model, names, k=10):
    """Print the features pushing hardest toward premium (+) or standard (-)."""
    order = np.argsort(model.w)
    print("\nTop features pushing toward PREMIUM (largest positive weights):")
    for i in order[::-1][:k]:
        print(f"  {names[i]:<30s} {model.w[i]:+.3f}")
    print("Top features pushing toward STANDARD (largest negative weights):")
    for i in order[:k]:
        print(f"  {names[i]:<30s} {model.w[i]:+.3f}")


# ------------------------------------------------------------------ main
def run(X_tr, y_tr, X_val, y_val, names):
    os.makedirs(FIG_DIR, exist_ok=True)
    lin_rows = sweep_linear(X_tr, y_tr, X_val, y_val)
    rff_rows = sweep_rff(X_tr, y_tr, X_val, y_val)

    best_lin = max(lin_rows, key=lambda r: r["val_f1"])   # choose on VALIDATION only
    best_rff = max(rff_rows, key=lambda r: r["val_f1"])

    plot_lambda_sweep(lin_rows)
    plot_rff_heatmap(rff_rows)
    plot_convergence(best_lin, best_rff)
    save_csv(lin_rows + rff_rows)
    top_weights(best_lin["_model"], names)

    print("\n=== BEST (chosen on validation F1) ===")
    print(f"linear: lambda={best_lin['lambda_']:g}  valF1={best_lin['val_f1']:.4f}")
    print(f"rff   : lambda={best_rff['lambda_']:g}  gamma={best_rff['gamma']:.4f}  "
          f"valF1={best_rff['val_f1']:.4f}")
    return best_lin, best_rff


def main():
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_task_b_data()
    print(f"train={X_tr.shape}, val={X_val.shape}, features={len(names)}, "
          f"premium share (train)={y_tr.mean():.2f}")
    run(X_tr, y_tr, X_val, y_val, names)


if __name__ == "__main__":
    main()