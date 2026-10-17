"""
Unit tests for the from-scratch DecisionTree (src/decision_tree.py).

Run from the repo root:
    python -m pytest tests -v
"""

import os
import sys
from collections import Counter

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decision_tree import DecisionTree  # noqa: E402


def depth(node):
    if node.is_leaf():
        return 0
    return 1 + max(depth(node.left), depth(node.right))


def n_leaves(node):
    if node.is_leaf():
        return 1
    return n_leaves(node.left) + n_leaves(node.right)


def leaf_sizes(tree, X):
    counts = Counter()
    for x in np.asarray(X, dtype=float):
        node = tree.root
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        counts[id(node)] += 1
    return list(counts.values())


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8)).astype(np.float32).astype(float)
    Xt = rng.normal(size=(500, 8)).astype(np.float32).astype(float)
    y_cls = (X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.5, size=2000) > 1).astype(int)
    y_reg = 3 * X[:, 0] - X[:, 2] ** 2 + rng.normal(scale=0.3, size=2000)
    return X, Xt, y_cls, y_reg


# --- impurity ---------------------------------------------------------------
def test_impurity_values():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    gini = DecisionTree("classification", "gini").fit(X, y)
    entropy = DecisionTree("classification", "entropy").fit(X, y)
    assert gini._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert gini._impurity(np.array([0, 0, 0, 0])) == pytest.approx(0.0)
    assert entropy._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert entropy._impurity(np.array([1, 1, 1, 1])) == pytest.approx(0.0)
    reg = DecisionTree("regression", "mse")
    assert reg._impurity(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 3.0)


# --- match sklearn on tiny, fully determined examples -----------------------
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_sklearn_tiny_one_feature(criterion):
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ours = DecisionTree("classification", criterion).fit(X, y)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(X, y)
    assert ours.root.feature == sk.tree_.feature[0] == 0
    assert ours.root.threshold == pytest.approx(sk.tree_.threshold[0])
    assert ours.root.threshold == pytest.approx(6.5)
    assert n_leaves(ours.root) == sk.get_n_leaves() == 2
    assert np.array_equal(ours.predict(X), sk.predict(X))
    assert np.allclose(ours.predict_proba(X), sk.predict_proba(X))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_sklearn_tiny_picks_informative_feature(criterion):
    X = np.array([[5.0, 1.0], [3.0, 2.0], [5.0, 10.0], [3.0, 11.0]])
    y = np.array([0, 0, 1, 1])
    ours = DecisionTree("classification", criterion).fit(X, y)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(X, y)
    assert ours.root.feature == sk.tree_.feature[0] == 1
    assert ours.root.threshold == pytest.approx(sk.tree_.threshold[0])
    assert ours.root.threshold == pytest.approx(6.0)
    assert np.array_equal(ours.predict(X), y)


def test_matches_sklearn_tiny_regression():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([1.0, 1.2, 0.8, 5.0, 5.2, 4.8])
    ours = DecisionTree("regression", "mse", max_depth=1).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    assert ours.root.feature == sk.tree_.feature[0]
    assert ours.root.threshold == pytest.approx(sk.tree_.threshold[0])
    assert np.allclose(ours.predict(X), sk.predict(X))


# --- match sklearn on larger data (no ties, float32-exact inputs) -----------
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_sklearn_classification_depth_limited(data, criterion):
    X, Xt, y_cls, _ = data
    ours = DecisionTree("classification", criterion, max_depth=4).fit(X, y_cls)
    sk = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=0).fit(X, y_cls)
    assert np.array_equal(ours.predict(Xt), sk.predict(Xt))
    assert np.allclose(ours.predict_proba(Xt), sk.predict_proba(Xt))
    assert np.allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-9)


def test_matches_sklearn_regression(data):
    X, Xt, _, y_reg = data
    ours = DecisionTree("regression", "mse", min_samples_leaf=10).fit(X, y_reg)
    sk = DecisionTreeRegressor(min_samples_leaf=10, random_state=0).fit(X, y_reg)
    assert np.allclose(ours.predict(Xt), sk.predict(Xt))
    assert np.allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-9)


def test_matches_sklearn_min_impurity_decrease(data):
    X, Xt, y_cls, y_reg = data
    ours = DecisionTree("classification", "gini", min_impurity_decrease=0.002).fit(X, y_cls)
    sk = DecisionTreeClassifier(min_impurity_decrease=0.002, random_state=0).fit(X, y_cls)
    assert np.array_equal(ours.predict(Xt), sk.predict(Xt))
    ours = DecisionTree("regression", "mse", min_impurity_decrease=0.01).fit(X, y_reg)
    sk = DecisionTreeRegressor(min_impurity_decrease=0.01, random_state=0).fit(X, y_reg)
    assert np.allclose(ours.predict(Xt), sk.predict(Xt))


# --- every hyperparameter must actually change the tree ---------------------
def test_max_depth_changes_tree(data):
    X, _, y_cls, _ = data
    sizes = []
    for d in [1, 2, 4, 8]:
        t = DecisionTree("classification", "gini", max_depth=d).fit(X, y_cls)
        assert depth(t.root) <= d
        sizes.append(n_leaves(t.root))
    assert sizes == sorted(sizes) and len(set(sizes)) > 1


def test_max_depth_zero_gives_single_leaf(data):
    X, _, y_cls, _ = data
    t = DecisionTree("classification", "gini", max_depth=0).fit(X, y_cls)
    assert t.root.is_leaf()


def test_min_samples_leaf_changes_tree(data):
    X, _, y_cls, _ = data
    full = DecisionTree("classification", "gini").fit(X, y_cls)
    limited = DecisionTree("classification", "gini", min_samples_leaf=25).fit(X, y_cls)
    assert n_leaves(limited.root) < n_leaves(full.root)
    assert min(leaf_sizes(limited, X)) >= 25


def test_min_samples_split_changes_tree(data):
    X, _, y_cls, _ = data
    full = DecisionTree("classification", "gini").fit(X, y_cls)
    limited = DecisionTree("classification", "gini", min_samples_split=100).fit(X, y_cls)
    assert n_leaves(limited.root) < n_leaves(full.root)
    too_big = DecisionTree("classification", "gini", min_samples_split=len(X) + 1).fit(X, y_cls)
    assert too_big.root.is_leaf()


def test_min_impurity_decrease_changes_tree(data):
    X, _, y_cls, y_reg = data
    a = DecisionTree("classification", "gini").fit(X, y_cls)
    b = DecisionTree("classification", "gini", min_impurity_decrease=0.005).fit(X, y_cls)
    assert n_leaves(b.root) < n_leaves(a.root)
    a = DecisionTree("regression", "mse", max_depth=8).fit(X, y_reg)
    b = DecisionTree("regression", "mse", max_depth=8, min_impurity_decrease=0.05).fit(X, y_reg)
    assert n_leaves(b.root) < n_leaves(a.root)


def test_gini_and_entropy_can_differ(data):
    X, Xt, y_cls, _ = data
    g = DecisionTree("classification", "gini").fit(X, y_cls)
    e = DecisionTree("classification", "entropy").fit(X, y_cls)
    assert n_leaves(g.root) != n_leaves(e.root) or not np.array_equal(g.predict(Xt), e.predict(Xt))


# --- predictions, probabilities, importances --------------------------------
def test_predict_proba_valid(data):
    X, Xt, y_cls, _ = data
    t = DecisionTree("classification", "gini", max_depth=5).fit(X, y_cls)
    p = t.predict_proba(Xt)
    assert p.shape == (len(Xt), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.array_equal(t.predict(Xt), t.classes_[np.argmax(p, axis=1)])


def test_feature_importances(data):
    X, _, y_cls, y_reg = data
    t = DecisionTree("classification", "gini", max_depth=5).fit(X, y_cls)
    assert t.feature_importances_.sum() == pytest.approx(1.0)
    assert (t.feature_importances_ >= 0).all()
    assert set(np.argsort(t.feature_importances_)[-2:]) == {0, 1}
    r = DecisionTree("regression", "mse", max_depth=5).fit(X, y_reg)
    assert r.feature_importances_.sum() == pytest.approx(1.0)
    assert np.argmax(r.feature_importances_) == 0


def test_fully_grown_tree_memorises_training_data(data):
    X, _, y_cls, y_reg = data
    t = DecisionTree("classification", "gini").fit(X, y_cls)
    assert np.array_equal(t.predict(X), y_cls)
    r = DecisionTree("regression", "mse").fit(X, y_reg)
    assert np.allclose(r.predict(X), y_reg)


def test_pure_node_is_a_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    t = DecisionTree("classification", "gini").fit(X, np.ones(10, dtype=int))
    assert t.root.is_leaf()
    assert np.all(t.predict(X) == 1)


def test_xor_needs_zero_gain_first_split():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    t = DecisionTree("classification", "gini").fit(X, y)
    assert np.array_equal(t.predict(X), y)


def test_string_labels_are_supported():
    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    y = np.array(["standard", "standard", "premium", "premium"])
    t = DecisionTree("classification", "gini").fit(X, y)
    assert list(t.predict(np.array([[1.5], [10.5]]))) == ["standard", "premium"]


def test_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int)
    a = DecisionTree("classification", "gini", max_depth=4).fit(X, y)
    b = DecisionTree("classification", "gini", max_depth=4).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


# --- input validation -------------------------------------------------------
def test_bad_task_criterion_combinations():
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        DecisionTree("classification", "mse").fit(X, np.array([0, 1]))
    with pytest.raises(ValueError):
        DecisionTree("regression", "gini").fit(X, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        DecisionTree("regression", "mse").fit(X, np.array([0.0, 1.0])).predict_proba(X)
