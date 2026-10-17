"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator should:
  1. load + clean the data,
  2. build features, make the splits, standardize (for the SVM),
  3. train YOUR decision tree (Task A regression, Task B classification),
  4. train YOUR Pegasos SVM (Task B classification),
  5. train the scikit-learn baselines with matched settings,
  6. compute metrics and SAVE all figures to report/figures/,
  7. print a final comparison table.

Everything must be deterministic (fixed seeds). Keep print output tidy so a
grader can read the headline numbers straight from the console.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep
from . import eda
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures
from .ridge import RidgeRegression
from . import evaluate
from . import svm_experiments
from . import tree_experiments

SEED = data_prep.SEED
FIG_DIR = eda.FIG_DIR
REPORT_DIR = "report"
RIDGE_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
SVC_MAX_ROWS = 10000
BLUE, ORANGE, GREEN, RED, GREY = "#3266AD", "#D9822B", "#3E8E5B", "#B5443B", "#8A8A8A"


class RFFSVM:
    def __init__(self, gamma, lambda_, n_components, n_epochs, random_state=SEED):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.rff = RandomFourierFeatures(gamma=self.gamma, n_components=self.n_components,
                                         random_state=self.random_state).fit(X)
        self.svm = PegasosSVM(lambda_=self.lambda_, n_epochs=self.n_epochs,
                              random_state=self.random_state).fit(self.rff.transform(X), y)
        return self

    def decision_function(self, X):
        return self.svm.decision_function(self.rff.transform(X))

    def predict(self, X):
        return self.svm.predict(self.rff.transform(X))


class Subsample:
    def __init__(self, build, max_rows, seed=SEED):
        self.build = build
        self.max_rows = max_rows
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if len(X) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.model = self.build(len(X)).fit(X, y)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        return self.model.predict(X)


def spec(name, make, std=False, scratch=False):
    return {"name": name, "make": make, "std": std, "scratch": scratch}


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.decision_function(X))


def fit_and_predict(sp, X_a, y_a, evals, want_score=False):
    keys = list(evals)
    mats = [evals[k] for k in keys]
    if sp["std"]:
        scaled = data_prep.standardize(X_a, *mats)
        X_a, mats = scaled[0], list(scaled[1:])
    model = sp["make"]()
    t0 = time.perf_counter()
    model.fit(X_a, y_a)
    out = {"model": model, "fit_s": time.perf_counter() - t0, "pred": {}, "score": {}, "predict_s": 0.0}
    for k, M in zip(keys, mats):
        t0 = time.perf_counter()
        out["pred"][k] = np.asarray(model.predict(M))
        if k == "test":
            out["predict_s"] = time.perf_counter() - t0
        if want_score:
            out["score"][k] = get_score(model, M)
    return out


def kfold_indices(y, folds, seed, stratified):
    rng = np.random.default_rng(seed)
    n = len(y)
    fold_of = np.empty(n, dtype=int)
    if stratified:
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            fold_of[idx] = np.arange(len(idx)) % folds
    else:
        fold_of[rng.permutation(n)] = np.arange(n) % folds
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(folds)]


def cls_metrics(y, pred, score=None):
    p, r, f = evaluate.precision_recall_f1(y, pred)
    out = {"acc": float(np.mean(np.asarray(y) == np.asarray(pred))), "prec": p, "rec": r, "f1": f}
    if score is not None:
        out["auc"] = evaluate.roc_auc(y, score)
    return out


def reg_metrics(ylog, pred_log):
    price, pred_price = np.exp(ylog), np.exp(pred_log)
    return {"r2_log": evaluate.r2(ylog, pred_log), "rmse_log": evaluate.rmse(ylog, pred_log),
            "mae_azn": evaluate.mae(price, pred_price), "rmse_azn": evaluate.rmse(price, pred_price),
            "r2_azn": evaluate.r2(price, pred_price)}


def cross_validate(sp, X, y, folds, task):
    stratified = task == "classification"
    scores = {"acc": [], "f1": []} if stratified else {"r2_log": [], "rmse_log": []}
    for tr_idx, te_idx in kfold_indices(y, folds, SEED, stratified):
        res = fit_and_predict(sp, X[tr_idx], y[tr_idx], {"fold": X[te_idx]})
        m = cls_metrics(y[te_idx], res["pred"]["fold"]) if stratified else reg_metrics(y[te_idx], res["pred"]["fold"])
        for k in scores:
            scores[k].append(m[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}


def evaluate_specs(specs, task, data, X_cv, y_cv, folds):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    rows, fitted, preds = {}, {}, {}
    for sp in specs:
        print(f"-> {sp['name']}", flush=True)
        try:
            cls = task == "classification"
            res = fit_and_predict(sp, X_tr, y_tr, {"val": X_val, "test": X_te}, want_score=cls)
            cv = cross_validate(sp, X_cv, y_cv, folds, task)
            if cls:
                val_m = cls_metrics(y_val, res["pred"]["val"])
                te = cls_metrics(y_te, res["pred"]["test"], res["score"]["test"])
                row = {"val_f1": val_m["f1"], "test_acc": te["acc"], "test_prec": te["prec"],
                       "test_rec": te["rec"], "test_f1": te["f1"], "test_auc": te["auc"],
                       "cv_f1": cv["f1"], "cv_acc": cv["acc"]}
            else:
                val_m = reg_metrics(y_val, res["pred"]["val"])
                te = reg_metrics(y_te, res["pred"]["test"])
                row = {"val_rmse_log": val_m["rmse_log"], "test_r2_log": te["r2_log"],
                       "test_rmse_log": te["rmse_log"], "test_mae_azn": te["mae_azn"],
                       "test_rmse_azn": te["rmse_azn"], "test_r2_azn": te["r2_azn"],
                       "cv_r2_log": cv["r2_log"], "cv_rmse_log": cv["rmse_log"]}
            row["fit_s"], row["predict_s"] = res["fit_s"], res["predict_s"]
            rows[sp["name"]] = row
            fitted[sp["name"]] = res["model"]
            preds[sp["name"]] = res["pred"]["test"]
        except Exception:
            print(f"[warn] {sp['name']} failed and is left out of the table:")
            traceback.print_exc()
    return rows, fitted, preds


def best_svm_from_csv(path=os.path.join(REPORT_DIR, "svm_results.csv")):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    def best(kind):
        r = max((r for r in rows if r["model"] == kind), key=lambda r: float(r["val_f1"]))
        return {"lambda_": float(r["lambda_"]), "gamma": float(r["gamma"]), "val_f1": float(r["val_f1"])}

    return best("linear"), best("rff")


def pick_ridge_lambda(X_tr, ylog_tr, X_val, ylog_val):
    Xs_tr, Xs_val = data_prep.standardize(X_tr, X_val)
    errs = [evaluate.rmse(ylog_val, RidgeRegression(lam).fit(Xs_tr, ylog_tr).predict(Xs_val))
            for lam in RIDGE_LAMBDAS]
    return RIDGE_LAMBDAS[int(np.argmin(errs))]


def save_table(rows, name):
    os.makedirs(REPORT_DIR, exist_ok=True)
    cols = []
    for m in rows.values():
        cols += [k for k in m if k not in cols]
    pairs = {k for k in cols if any(isinstance(m.get(k), tuple) for m in rows.values())}

    def cell(v):
        if isinstance(v, tuple):
            return f"{v[0]:.4f} ± {v[1]:.4f}"
        return "-" if v is None else f"{v:.4f}"

    with open(os.path.join(REPORT_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("| model | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for n, m in rows.items():
            f.write(f"| {n} | " + " | ".join(cell(m.get(c)) for c in cols) + " |\n")
    with open(os.path.join(REPORT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [h for k in cols for h in ((f"{k}_mean", f"{k}_std") if k in pairs else (k,))])
        for n, m in rows.items():
            line = [n]
            for k in cols:
                v = m.get(k)
                line += list(v) if k in pairs else [v]
            w.writerow(line)
    print(f"Saved {REPORT_DIR}/{name}.md and .csv")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)
    print(f"saved {FIG_DIR}/{name}")


def plot_importances(models, feat_names, k=10):
    fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8))
    ax = np.atleast_1d(ax)
    for a, (title, model, color) in zip(ax, models):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1][:k][::-1]
        a.barh([feat_names[i] for i in order], imp[order], color=color)
        a.set_xlabel("impurity-decrease importance (sums to 1)")
        a.set_title(title, fontsize=10)
        a.grid(True, axis="x", color="#E3E3E3", linewidth=0.6)
        print(f"\n{title} - top features:")
        for i in order[::-1][:5]:
            print(f"  {feat_names[i]:<24s} {imp[i]:.3f}")
    _save(fig, "feature_importances_trees.png")


def plot_svm_weights(model, feat_names, k=8):
    w = model.w
    order = np.argsort(w)
    pick = np.concatenate([order[:k], order[-k:]])
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.barh([feat_names[i] for i in pick], w[pick], color=[RED if w[i] < 0 else BLUE for i in pick])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("weight on standardized feature (+ premium, - standard)")
    ax.set_title("Linear Pegasos SVM: largest weights", fontsize=10)
    ax.grid(True, axis="x", color="#E3E3E3", linewidth=0.6)
    _save(fig, "svm_weights.png")
    print("\nSVM weights - toward PREMIUM:  " + ", ".join(f"{feat_names[i]} ({w[i]:+.2f})" for i in order[::-1][:5]))
    print("SVM weights - toward STANDARD: " + ", ".join(f"{feat_names[i]} ({w[i]:+.2f})" for i in order[:5]))


def category_labels(X, feat_names):
    cols = [i for i, n in enumerate(feat_names) if n.startswith("cat_")]
    if not cols:
        return np.array(["all"] * len(X))
    names = np.array([feat_names[i][4:] for i in cols])
    sub = X[:, cols]
    return np.where(sub.max(axis=1) > 0, names[sub.argmax(axis=1)], "other")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {path}")


def error_analysis_cls(name, pred, y_te, p_te, thr, X_te, feat_names):
    cm = evaluate.confusion_matrix(y_te, pred)
    wrong = np.asarray(pred) != np.asarray(y_te)
    gap = np.abs(np.log(p_te / thr))
    edges = np.quantile(gap, np.linspace(0, 1, 6))
    which = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0, 4)
    bin_err = [float(wrong[which == b].mean()) for b in range(5)]
    cats = category_labels(X_te, feat_names)
    cat_rows = sorted(((c, int((cats == c).sum()), float(wrong[cats == c].mean())) for c in np.unique(cats)),
                      key=lambda r: -r[2])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_xticks([0, 1], ["standard", "premium"])
    ax[0].set_yticks([0, 1], ["standard", "premium"])
    ax[0].set_xlabel("predicted")
    ax[0].set_ylabel("actual")
    ax[0].set_title(f"Confusion matrix: {name}", fontsize=9)
    ax[1].bar([f"{b + 1}" for b in range(5)], bin_err, color=BLUE)
    ax[1].set_xlabel("distance from tier threshold (1 = closest 20%)")
    ax[1].set_ylabel("error rate")
    ax[1].set_title("Errors concentrate near the threshold?", fontsize=9)
    ax[2].barh([f"{c} (n={n})" for c, n, _ in cat_rows][::-1], [e for _, _, e in cat_rows][::-1], color=ORANGE)
    ax[2].set_xlabel("error rate")
    ax[2].set_title("Error rate by property category", fontsize=9)
    for a in ax[1:]:
        a.grid(True, color="#E3E3E3", linewidth=0.6)
    _save(fig, "error_analysis_task_b.png")
    write_csv(os.path.join(REPORT_DIR, "error_by_category_task_b.csv"), ["category", "n_test", "error_rate"], cat_rows)
    print(f"\nTask B error analysis ({name}): overall error {wrong.mean():.3f}; "
          f"error rate by distance-to-threshold bin (closest to farthest): " + ", ".join(f"{e:.3f}" for e in bin_err))
    print("Highest-error categories: " + ", ".join(f"{c} ({e:.3f}, n={n})" for c, n, e in cat_rows[:3]))


def error_analysis_reg(name, pred_log, ylog_te, X_te, feat_names):
    resid = np.asarray(pred_log) - np.asarray(ylog_te)
    cats = category_labels(X_te, feat_names)
    cat_rows = sorted(((c, int((cats == c).sum()), float(np.abs(resid[cats == c]).mean())) for c in np.unique(cats)),
                      key=lambda r: -r[2])
    area_i = feat_names.index("area_m2") if "area_m2" in feat_names else None
    worst = np.argsort(-np.abs(resid))[:20]
    sub = np.random.default_rng(SEED).choice(len(resid), min(len(resid), 5000), replace=False)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(np.asarray(ylog_te)[sub], np.asarray(pred_log)[sub], s=3, alpha=0.3, color=BLUE, linewidths=0)
    lim = [min(np.min(ylog_te), np.min(pred_log)), max(np.max(ylog_te), np.max(pred_log))]
    ax[0].plot(lim, lim, color=RED, linewidth=1)
    ax[0].set_xlabel("actual log(price)")
    ax[0].set_ylabel("predicted log(price)")
    ax[0].set_title(f"Predicted vs actual: {name}", fontsize=9)
    ax[1].hist(resid, bins=60, color=BLUE)
    ax[1].set_xlabel("residual (predicted - actual log price)")
    ax[1].set_ylabel("listings")
    ax[1].set_title("Residuals", fontsize=9)
    ax[2].barh([f"{c} (n={n})" for c, n, _ in cat_rows][::-1], [e for _, _, e in cat_rows][::-1], color=ORANGE)
    ax[2].set_xlabel("mean |residual| in log(price)")
    ax[2].set_title("Error by property category", fontsize=9)
    for a in ax:
        a.grid(True, color="#E3E3E3", linewidth=0.6)
    _save(fig, "error_analysis_task_a.png")
    write_csv(os.path.join(REPORT_DIR, "error_by_category_task_a.csv"),
              ["category", "n_test", "mean_abs_log_residual"], cat_rows)
    write_csv(os.path.join(REPORT_DIR, "worst_predictions_task_a.csv"),
              ["actual_price", "predicted_price", "area_m2", "category"],
              [(round(float(np.exp(ylog_te[i]))), round(float(np.exp(pred_log[i]))),
                round(float(X_te[i, area_i]), 1) if area_i is not None else "", cats[i]) for i in worst])
    print(f"\nTask A error analysis ({name}): mean |residual| {np.abs(resid).mean():.3f} (log scale)")
    print("Highest-error categories: " + ", ".join(f"{c} ({e:.3f}, n={n})" for c, n, e in cat_rows[:3]))
    print("Worst 5 test listings (actual -> predicted AZN): " + "; ".join(
        f"{np.exp(ylog_te[i]):,.0f} -> {np.exp(pred_log[i]):,.0f} ({cats[i]})" for i in worst[:5]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reproduce every number and figure of the project.")
    p.add_argument("--quick", action="store_true", help="smoke test: small data subset, tiny sweeps, 3 folds")
    p.add_argument("--skip-sweeps", action="store_true",
                   help="reuse report/tree_best.json and report/svm_results.csv from an earlier run")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    folds = 3 if args.quick else args.folds
    svc_rows = 2000 if args.quick else SVC_MAX_ROWS
    if args.quick:
        svm_experiments.LAMBDAS = [1e-4, 1e-2]
        svm_experiments.N_EPOCHS = 3

    # 1. data -------------------------------------------------------------
    df = data_prep.load_raw()
    df = data_prep.clean(df)
    if args.quick:
        df = df.sample(n=min(len(df), 6000), random_state=SEED).reset_index(drop=True)
    X, y, feat_names = data_prep.make_features(df)
    splits = data_prep.train_val_test_split(X, y)
    # EDA figures -> report/figures/
    eda.price_distribution(df)
    eda.area_vs_price(df)
    eda.rooms_vs_price(df)
    eda.location_effect(df)
    eda.tier_balance(df)

    X_tr, p_tr, X_val, p_val, X_te, p_te = splits
    ylog_tr, ylog_val, ylog_te = np.log(p_tr), np.log(p_val), np.log(p_te)
    t_tr, thr = data_prep.make_tier_label(p_tr)
    t_val, _ = data_prep.make_tier_label(p_val, thr)
    t_te, _ = data_prep.make_tier_label(p_te, thr)
    X_cv = np.vstack([X_tr, X_val])
    print(f"\nrows train/val/test = {len(X_tr):,}/{len(X_val):,}/{len(X_te):,}, features = {len(feat_names)}, "
          f"tier threshold (TRAIN median) = {thr:,.0f} AZN, premium share test = {t_te.mean():.3f}")

    print("\n=== hyper-parameters (chosen on VALIDATION only) ===")
    if args.skip_sweeps:
        best_cls, best_reg = tree_experiments.load_best()
        best_lin, best_rff = best_svm_from_csv()
    else:
        best_cls, best_reg = tree_experiments.run(X_tr, t_tr, ylog_tr, X_val, t_val, ylog_val, quick=args.quick)
        Xs_tr, Xs_val, _ = data_prep.standardize(X_tr, X_val, X_te)
        best_lin, best_rff = svm_experiments.run(Xs_tr, t_tr, Xs_val, t_val, feat_names)
    ridge_lambda = pick_ridge_lambda(X_tr, ylog_tr, X_val, ylog_val)
    n_epochs, n_comp = svm_experiments.N_EPOCHS, svm_experiments.RFF_COMPONENTS
    print(f"tree Task B: {best_cls}\ntree Task A: {best_reg}\nlinear SVM: lambda={best_lin['lambda_']:g}\n"
          f"RFF SVM: lambda={best_rff['lambda_']:g}, gamma={best_rff['gamma']:.4f}\nridge: lambda={ridge_lambda:g}")

    # 2. Task A: regression (price) --------------------------------------
    tree_a = dict(max_depth=best_reg["max_depth"], min_samples_leaf=best_reg["min_samples_leaf"])
    specs_a = [
        spec("Mean baseline", lambda: DummyRegressor(strategy="mean")),
        spec("Scratch tree (mse)", lambda: DecisionTree(task="regression", criterion="mse",
                                                        random_state=SEED, **tree_a), scratch=True),
        spec("sklearn DecisionTreeRegressor", lambda: DecisionTreeRegressor(random_state=SEED, **tree_a)),
        spec(f"Scratch Ridge (lambda={ridge_lambda:g})", lambda: RidgeRegression(ridge_lambda), std=True, scratch=True),
        spec("sklearn Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=ridge_lambda))),
    ]
    print("\n=== Task A (price): fitting models ===")
    rows_a, fitted_a, preds_a = evaluate_specs(
        specs_a, "regression", (X_tr, ylog_tr, X_val, ylog_val, X_te, ylog_te), X_cv,
        np.concatenate([ylog_tr, ylog_val]), folds)

    # 3. Task B: classification (price tier) -----------------------------
    tree_b = dict(max_depth=best_cls["max_depth"], min_samples_leaf=best_cls["min_samples_leaf"])
    crit = best_cls["criterion"]
    lam, lam_r, gam = best_lin["lambda_"], best_rff["lambda_"], best_rff["gamma"]
    scratch_tree_b = f"Scratch tree ({crit})"
    scratch_svm_b = "Scratch Pegasos SVM (linear)"
    specs_b = [
        spec("Majority-class baseline", lambda: DummyClassifier(strategy="most_frequent")),
        spec(scratch_tree_b, lambda: DecisionTree(task="classification", criterion=crit,
                                                  random_state=SEED, **tree_b), scratch=True),
        spec(f"sklearn DecisionTreeClassifier ({crit})", lambda: DecisionTreeClassifier(
            criterion=crit, random_state=SEED, **tree_b)),
        spec(scratch_svm_b, lambda: PegasosSVM(lambda_=lam, n_epochs=n_epochs, random_state=SEED),
             std=True, scratch=True),
        spec("sklearn SGDClassifier (hinge, Pipeline)", lambda: make_pipeline(
            StandardScaler(), SGDClassifier(loss="hinge", penalty="l2", alpha=lam, max_iter=n_epochs,
                                             tol=None, random_state=SEED))),
        spec("Scratch RFF-SVM (RBF approx.)", lambda: RFFSVM(gam, lam_r, n_comp, n_epochs),
             std=True, scratch=True),
        spec(f"sklearn SVC (rbf, Pipeline, <= {svc_rows:,} rows)", lambda: Subsample(
            lambda n: make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gam, C=1.0 / (lam_r * n))),
            svc_rows)),
    ]
    print("\n=== Task B (price tier): fitting models ===")
    rows_b, fitted_b, preds_b = evaluate_specs(
        specs_b, "classification", (X_tr, t_tr, X_val, t_val, X_te, t_te), X_cv,
        np.concatenate([t_tr, t_val]), folds)

    # 4. metrics + figures + comparison ----------------------------------
    print("\n" + "=" * 100 + "\nTASK A - regression of log(price). Test-set metrics; CV = mean ± std over "
          f"{folds} folds on train+val.\n" + "=" * 100)
    evaluate.compare(rows_a)
    save_table(rows_a, "results_task_a")
    print("\n" + "=" * 100 + "\nTASK B - price tier (premium = above the TRAIN median). Test-set metrics; "
          f"CV = mean ± std over {folds} folds on train+val.\n" + "=" * 100)
    evaluate.compare(rows_b)
    save_table(rows_b, "results_task_b")
    print("\nNotes: fit_s / predict_s are seconds on the full training split / the test set (single run). "
          "Scratch SVM and Ridge are standardized with TRAIN statistics (fold-train statistics inside CV); "
          "the sklearn SVMs scale inside a Pipeline. The sklearn SVC is trained on a subsample (O(n^2)) with "
          "C = 1/(lambda * n) so its objective matches Pegasos.")

    trees = [(t, m, c) for t, m, c in (("Task B tree (price tier)", fitted_b.get(scratch_tree_b), BLUE),
                                       ("Task A tree (log price)", fitted_a.get("Scratch tree (mse)"), GREEN))
             if m is not None]
    if trees:
        plot_importances(trees, feat_names)
    if scratch_svm_b in fitted_b:
        plot_svm_weights(fitted_b[scratch_svm_b], feat_names)

    scratch_b = [n for n in rows_b if any(s["name"] == n and s["scratch"] for s in specs_b)]
    scratch_a = [n for n in rows_a if any(s["name"] == n and s["scratch"] for s in specs_a)]
    if scratch_b:
        best_b = max(scratch_b, key=lambda n: rows_b[n]["val_f1"])
        error_analysis_cls(best_b, preds_b[best_b], t_te, p_te, thr, X_te, feat_names)
    if scratch_a:
        best_a = min(scratch_a, key=lambda n: rows_a[n]["val_rmse_log"])
        error_analysis_reg(best_a, preds_a[best_a], ylog_te, X_te, feat_names)

    print("\nDone. Tables are in report/, figures in report/figures/.")


if __name__ == "__main__":
    main()