from __future__ import annotations

import numpy as np


class RidgeRegression:
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.lambda_ * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, Xc.T @ yc)
        self.b = float(y_mean - x_mean @ self.w)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b