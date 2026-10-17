"""
svm.py -- a soft-margin SVM built FROM SCRATCH with PEGASOS (NumPy only).

THE BIG IDEA (plain English)
----------------------------
An SVM draws a straight line (in many dimensions: a "hyperplane") that splits
two classes, and tries to keep the line as FAR as possible from the closest
points of both classes. The empty gap around the line is called the MARGIN.

The line is   w . x + b = 0   (w = weights, b = bias/offset).
  * if  w . x + b > 0  -> we predict the "+1" class (premium)
  * if  w . x + b < 0  -> we predict the "-1" class (standard)

"Soft-margin" means we allow some points to be on the wrong side or inside the
margin, but we pay a penalty for them (the hinge loss). We minimise:

    J(w, b) = (lambda/2) * ||w||^2  +  (1/n) * sum_i max(0, 1 - y_i (w . x_i + b))
              [ keeps margin wide ]       [ penalty for bad points ]

  * lambda BIG   -> wide margin, but many points are allowed to be wrong
  * lambda SMALL -> narrow margin, fits the training points more strictly

HOW PEGASOS MINIMISES J
-----------------------
Repeat many times: pick one training point, look at it, nudge w a little.
  * If the point is correctly classified with room to spare (y*(w.x+b) >= 1):
        only shrink w a tiny bit (this is the "keep the margin wide" part).
  * If the point is wrong or too close to the line (y*(w.x+b) < 1):
        shrink w AND push w toward classifying this point correctly.
The step size ("learning rate") is  eta_t = 1 / (lambda * (t + t0)).

NOTE ON t0: pure Pegasos uses t0 = 0, but then the very first step sizes are
huge when lambda is tiny (eta = 1/(lambda*1) = 10,000 for lambda=1e-4) and the
weights jump wildly. We start the counter at t0 = 1/lambda by default, which
makes the first step size about 1. This is a common, safe variant.
Set t0=0 to get the textbook version.

Non-linearity: we use RANDOM FOURIER FEATURES (class RandomFourierFeatures
below). It maps X to a bigger space whose dot products approximate the RBF
kernel, then our LINEAR Pegasos runs there. Reason: it is short, fast
(O(n) instead of the O(n^2) kernel matrix) and reuses the same solver.

scikit-learn is only a BASELINE elsewhere -- it is NOT used in this file.
Remember to STANDARDISE features before training (fit scaler on TRAIN only).
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength (wide vs narrow margin)
        n_epochs: int = 20,          # number of full passes over the training data
        batch_size: int = 1,         # 1 = classic Pegasos; bigger = faster, smoother
        lr_schedule: str = "pegasos",  # "pegasos": 1/(lambda*(t+t0)) | "constant" | "sqrt"
        eta0: float = 0.1,           # step size, used only by "constant" and "sqrt"
        t0: float | None = None,     # step-counter offset (None -> 1/lambda)
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.t0 = t0
        self.random_state = random_state

        self.w = None                # weight vector (set by fit)
        self.b = 0.0                 # bias (set by fit)
        self.classes_ = None         # the two original labels, e.g. [0, 1]
        self.history_ = []           # objective value J after every epoch

    # ------------------------------------------------------------------ labels
    def _to_pm1(self, y):
        """Map the two original labels (e.g. 0/1) to -1/+1.
        The larger label becomes +1 ("premium"). We remember the mapping
        in self.classes_ so predict() can return the original labels."""
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Need exactly 2 classes, got {len(classes)}.")
        self.classes_ = classes
        return np.where(y == classes[1], 1.0, -1.0)

    # --------------------------------------------------------------- objective
    def objective(self, X, y_pm):
        """J(w,b) = lambda/2 * ||w||^2 + mean hinge loss. y_pm must be +/-1."""
        margins = y_pm * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins)
        return 0.5 * self.lambda_ * float(self.w @ self.w) + float(hinge.mean())

    # ---------------------------------------------------------- learning rate
    def _eta(self, t):
        """Step size at step t (t starts at 1)."""
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * (t + self._t0))
        if self.lr_schedule == "constant":
            return self.eta0
        if self.lr_schedule == "sqrt":
            return self.eta0 / np.sqrt(t)
        raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

    # --------------------------------------------------------------------- fit
    def fit(self, X, y):
        """Train with (mini-batch) sub-gradient descent on the hinge-loss primal."""
        X = np.asarray(X, dtype=float)
        y_pm = self._to_pm1(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self._t0 = (1.0 / self.lambda_) if self.t0 is None else float(self.t0)
        self.w = np.zeros(d)
        self.b = 0.0
        self.history_ = []

        t = 0
        for _ in range(self.n_epochs):
            order = rng.permutation(n)                 # shuffle each epoch
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb = X[idx], y_pm[idx]
                t += 1
                eta = self._eta(t)

                # which points in this batch are "bad" (inside margin / wrong)?
                viol = yb * (Xb @ self.w + self.b) < 1.0

                # shrink w (regularisation part)
                self.w *= (1.0 - eta * self.lambda_)
                # push w toward the bad points (hinge part), averaged over batch
                if viol.any():
                    k = len(idx)
                    self.w += (eta / k) * (yb[viol][:, None] * Xb[viol]).sum(axis=0)
                    self.b += (eta / k) * yb[viol].sum()   # bias: own step, not shrunk

            self.history_.append(self.objective(X, y_pm))  # for the "loss goes down" plot
        return self

    # ---------------------------------------------------------------- predict
    def decision_function(self, X):
        """Signed distance-like score  w . x + b  (positive -> class +1)."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Return labels in the ORIGINAL label space (e.g. 0/1)."""
        pos = self.decision_function(X) >= 0
        return np.where(pos, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------- margin & SVs (3c)
    def weight_norm(self):
        """||w||."""
        return float(np.linalg.norm(self.w))

    def margin_width(self):
        """Width of the margin = 2 / ||w||."""
        return 2.0 / self.weight_norm()

    def support_vector_mask(self, X, y):
        """True for training points with y_i (w . x_i + b) <= 1
        (on or inside the margin, or misclassified) = the support vectors."""
        y_pm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        return y_pm * self.decision_function(X) <= 1.0 + 1e-12


class RandomFourierFeatures:
    """Non-linear extension (Problem 3d, option ii): random Fourier features.

    Maps X (n x d) to Z (n x D) with
        Z = sqrt(2/D) * cos(X @ W + c),   W ~ Normal(0, 2*gamma),  c ~ Uniform(0, 2*pi)
    so that  Z_i . Z_j  is approximately  exp(-gamma * ||x_i - x_j||^2)  (RBF kernel).
    Train PegasosSVM on Z instead of X to get a non-linear boundary.
        gamma        : RBF bandwidth (bigger = wigglier boundary)
        n_components : D, number of random features (more = better approximation)
    """

    def __init__(self, gamma: float = 0.1, n_components: int = 300, random_state: int = 42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.W = None
        self.c = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(X.shape[1], self.n_components))
        self.c = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W + self.c)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)