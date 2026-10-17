"""
tree_experiments.py -- Problem 4a: tree sweeps, figures and the chosen configs.

Run from the repo root AFTER person A's data_prep.py works:
    python -m src.tree_experiments

What it does (fixed seed, TRAIN for fitting and VALIDATION for choosing; the
TEST set is never touched here):
  1. max_depth sweep for Gini vs entropy (Task B) and for MSE (Task A)
       -> train vs validation score = the overfitting / bias-variance story
  2. second hyper-parameter: min_samples_leaf sweep at the best depth
  3. learning curves (score vs training-set size) at the chosen configuration
Results are saved to report/figures/ (png), report/tree_results.csv and
report/tree_best.json (the chosen configs, read back by run_all.py).

Task B score = F1 of the premium class (higher is better).
Task A score = RMSE of log(price) (lower is better).
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_prep, evaluate
from .decision_tree import DecisionTree

FIG_DIR = "report/figures"
RESULTS_PATH = "report/tree_results.csv"
BEST_PATH = "report/tree_best.json"
SEED = 42
CRITERIA = ["gini", "entropy"]
DEPTHS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
MIN_LEAVES = [1, 5, 10, 25, 50, 100, 250]
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
BLUE, ORANGE, GREEN, RED = "#3266AD", "#D9822B", "#3E8E5B", "#B5443B"


def _fit(task, criterion, X, y, depth, leaf):
    t = time.time()
    model = DecisionTree(task=task, criterion=criterion, max_depth=depth,
                         min_samples_leaf=leaf, random_state=SEED).fit(X, y)
    return model, time.time() - t


def _score(task, model, X, y):
    pred = model.predict(X)
    if task == "classification":
        return evaluate.precision_recall_f1(y, pred)[2]
    return evaluate.rmse(y, pred)


def _n_leaves(node):
    return 1 if node.is_leaf() else _n_leaves(node.left) + _n_leaves(node.right)


def _row(task, criterion, sweep, value, model, seconds, tr, va):
    return dict(task=task, criterion=criterion, sweep=sweep, value=value,
                train=tr, val=va, leaves=_n_leaves(model.root), seconds=seconds)


def _better(task, a, b):
    return a > b if task == "classification" else a < b


def _best(task, rows):
    key = (lambda r: r["val"]) if task == "classification" else (lambda r: -r["val"])
    return max(rows, key=key)


def sweep_depth(task, criterion, data, depths):
    X_tr, y_tr, X_val, y_val = data
    rows = []
    for d in depths:
        m, sec = _fit(task, criterion, X_tr, y_tr, d, 1)
        rows.append(_row(task, criterion, "max_depth", d, m, sec,
                         _score(task, m, X_tr, y_tr), _score(task, m, X_val, y_val)))
        print(f"[tree {task[:3]} {criterion:7s}] max_depth={d:<3d} train={rows[-1]['train']:.4f} "
              f"val={rows[-1]['val']:.4f} leaves={rows[-1]['leaves']}")
    return rows


def sweep_min_leaf(task, criterion, depth, data, leaves):
    X_tr, y_tr, X_val, y_val = data
    rows = []
    for k in leaves:
        m, sec = _fit(task, criterion, X_tr, y_tr, depth, k)
        rows.append(_row(task, criterion, "min_samples_leaf", k, m, sec,
                         _score(task, m, X_tr, y_tr), _score(task, m, X_val, y_val)))
        print(f"[tree {task[:3]} {criterion:7s}] depth={depth} min_samples_leaf={k:<4d} "
              f"train={rows[-1]['train']:.4f} val={rows[-1]['val']:.4f} leaves={rows[-1]['leaves']}")
    return rows


def learning_curve(task, cfg, data, fractions):
    X_tr, y_tr, X_val, y_val = data
    perm = np.random.default_rng(SEED).permutation(len(y_tr))
    rows = []
    for f in fractions:
        idx = perm[:max(int(f * len(y_tr)), 20)]
        m, sec = _fit(task, cfg["criterion"], X_tr[idx], y_tr[idx], cfg["max_depth"], cfg["min_samples_leaf"])
        rows.append(_row(task, cfg["criterion"], "train_size", len(idx), m, sec,
                         _score(task, m, X_tr[idx], y_tr[idx]), _score(task, m, X_val, y_val)))
        print(f"[tree {task[:3]} curve] n={len(idx):<7d} train={rows[-1]['train']:.4f} val={rows[-1]['val']:.4f}")
    return rows


def _panel(ax, task, series, xlabel, logx=False):
    for (label, rows, color) in series:
        x = [r["value"] for r in rows]
        ax.plot(x, [r["train"] for r in rows], marker="o", linestyle="--", color=color, label=f"{label} train")
        ax.plot(x, [r["val"] for r in rows], marker="s", color=color, label=f"{label} validation")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 (premium class)" if task == "classification" else "RMSE of log(price)")
    ax.set_title("Task B: price tier" if task == "classification" else "Task A: log(price)", fontsize=10)
    ax.grid(True, color="#E3E3E3", linewidth=0.6)
    ax.legend(fontsize=7)


def plot_depth(cls_rows, reg_rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _panel(ax[0], "classification", [(c, cls_rows[c], col) for c, col in zip(CRITERIA, (BLUE, ORANGE))], "max_depth")
    _panel(ax[1], "regression", [("mse", reg_rows, GREEN)], "max_depth")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tree_depth_sweep.png", dpi=130)
    plt.close()


def plot_min_leaf(cls_rows, reg_rows, cls_cfg):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _panel(ax[0], "classification", [(cls_cfg["criterion"], cls_rows, BLUE)], "min_samples_leaf", logx=True)
    _panel(ax[1], "regression", [("mse", reg_rows, GREEN)], "min_samples_leaf", logx=True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tree_min_leaf_sweep.png", dpi=130)
    plt.close()


def plot_learning(cls_rows, reg_rows, cls_cfg):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _panel(ax[0], "classification", [(cls_cfg["criterion"], cls_rows, BLUE)], "training-set size", logx=True)
    _panel(ax[1], "regression", [("mse", reg_rows, GREEN)], "training-set size", logx=True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tree_learning_curve.png", dpi=130)
    plt.close()


def save_csv(rows, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Saved", path)


def load_best(path=BEST_PATH):
    with open(path) as f:
        best = json.load(f)
    return best["classification"], best["regression"]


def run(X_tr, ytier_tr, ylog_tr, X_val, ytier_val, ylog_val, quick=False):
    os.makedirs(FIG_DIR, exist_ok=True)
    depths = [2, 4, 6] if quick else DEPTHS
    leaves = [1, 10, 50] if quick else MIN_LEAVES
    fractions = [0.25, 0.5, 1.0] if quick else FRACTIONS
    cls_data = (X_tr, ytier_tr, X_val, ytier_val)
    reg_data = (X_tr, ylog_tr, X_val, ylog_val)

    cls_depth = {c: sweep_depth("classification", c, cls_data, depths) for c in CRITERIA}
    reg_depth = sweep_depth("regression", "mse", reg_data, depths)
    plot_depth(cls_depth, reg_depth)

    best_c = max((_best("classification", cls_depth[c]) for c in CRITERIA), key=lambda r: r["val"])
    best_r = _best("regression", reg_depth)

    cls_leaf = sweep_min_leaf("classification", best_c["criterion"], best_c["value"], cls_data, leaves)
    reg_leaf = sweep_min_leaf("regression", "mse", best_r["value"], reg_data, leaves)
    plot_min_leaf(cls_leaf, reg_leaf, best_c)

    leaf_c = _best("classification", cls_leaf)
    leaf_r = _best("regression", reg_leaf)
    best_cls = dict(criterion=best_c["criterion"], max_depth=int(best_c["value"]),
                    min_samples_leaf=int(leaf_c["value"]), val_f1=float(leaf_c["val"]))
    best_reg = dict(criterion="mse", max_depth=int(best_r["value"]),
                    min_samples_leaf=int(leaf_r["value"]), val_rmse_log=float(leaf_r["val"]))

    cls_curve = learning_curve("classification", best_cls, cls_data, fractions)
    reg_curve = learning_curve("regression", best_reg, reg_data, fractions)
    plot_learning(cls_curve, reg_curve, best_cls)

    all_rows = ([r for c in CRITERIA for r in cls_depth[c]] + reg_depth + cls_leaf + reg_leaf
                + cls_curve + reg_curve)
    save_csv(all_rows)
    with open(BEST_PATH, "w") as f:
        json.dump(dict(classification=best_cls, regression=best_reg), f, indent=2)

    print("\n=== BEST (chosen on validation only) ===")
    print(f"Task B tree: {best_cls}")
    print(f"Task A tree: {best_reg}")
    gini = max(r["val"] for r in cls_depth["gini"])
    entropy = max(r["val"] for r in cls_depth["entropy"])
    print(f"Gini vs entropy (best validation F1 over depths): gini={gini:.4f}  entropy={entropy:.4f}")
    return best_cls, best_reg


def load_data():
    df = data_prep.clean(data_prep.load_raw())
    X, p, names = data_prep.make_features(df)
    X_tr, p_tr, X_val, p_val, X_te, p_te = data_prep.train_val_test_split(X, p)
    t_tr, thr = data_prep.make_tier_label(p_tr)
    t_val, _ = data_prep.make_tier_label(p_val, thr)
    return X_tr, t_tr, np.log(p_tr), X_val, t_val, np.log(p_val), list(names)


def main():
    X_tr, t_tr, l_tr, X_val, t_val, l_val, names = load_data()
    print(f"train={X_tr.shape}, val={X_val.shape}, features={len(names)}")
    run(X_tr, t_tr, l_tr, X_val, t_val, l_val)


if __name__ == "__main__":
    main()
