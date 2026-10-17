"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        y = np.asarray(y)
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.var(y))
        p = np.bincount(y, minlength=self.n_classes_) / len(y)
        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _counts_impurity(self, counts, n):
        p = counts / n[:, None]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log2(p), 0.0)
        return -np.sum(plogp, axis=1)

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n, n_features = X.shape
        parent = self._impurity(y)
        min_leaf = self.min_samples_leaf
        classification = self.task == "classification"
        if classification:
            onehot = np.eye(self.n_classes_)[y]
            total = onehot.sum(axis=0)
        else:
            yc = y.astype(float) - y.mean()
            total_sum = yc.sum()
            total_sq = np.sum(yc ** 2)
        n_left = np.arange(1, n, dtype=float)
        n_right = n - n_left

        best = None
        best_gain = -np.inf
        for j in range(n_features):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            if xs[0] == xs[-1]:
                continue
            valid = (xs[1:] > xs[:-1]) & (n_left >= min_leaf) & (n_right >= min_leaf)
            if not valid.any():
                continue
            if classification:
                left = np.cumsum(onehot[order], axis=0)[:-1]
                right = total - left
                imp_l = self._counts_impurity(left, n_left)
                imp_r = self._counts_impurity(right, n_right)
            else:
                cs = np.cumsum(yc[order])[:-1]
                cs2 = np.cumsum(yc[order] ** 2)[:-1]
                imp_l = np.maximum(cs2 / n_left - (cs / n_left) ** 2, 0.0)
                imp_r = np.maximum(
                    (total_sq - cs2) / n_right - ((total_sum - cs) / n_right) ** 2, 0.0
                )
            gain = parent - (n_left * imp_l + n_right * imp_r) / n
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = gain[i]
                threshold = (xs[i] + xs[i + 1]) / 2.0
                if threshold == xs[i + 1]:
                    threshold = xs[i]
                best = (j, float(threshold), float(gain[i]))
        return best

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        n = len(y)
        node = Node()
        if self.task == "classification":
            node.value = np.bincount(y, minlength=self.n_classes_) / n
        else:
            node.value = float(np.mean(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n < self.min_samples_split or n < 2 * self.min_samples_leaf:
            return node
        if self._impurity(y) <= np.finfo(float).eps:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, decrease = split
        weighted_decrease = (n / self.n_samples_) * decrease
        if weighted_decrease + 1e-7 < self.min_impurity_decrease:
            return node

        self.feature_importances_[feature] += weighted_decrease
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification" and self.criterion not in ("gini", "entropy"):
            raise ValueError("classification needs criterion 'gini' or 'entropy'")
        if self.task == "regression" and self.criterion != "mse":
            raise ValueError("regression needs criterion 'mse'")
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.n_samples_, self.n_features_ = X.shape
        self.feature_importances_ = np.zeros(self.n_features_)
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)
        self.root = self._grow(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])