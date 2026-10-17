"""
Sanity tests for the from-scratch SVM on TOY data (no real dataset needed).
Run from the repo root:   python -m tests.test_svm
Saves a figure to report/figures/svm_toy_demo.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.svm import PegasosSVM, RandomFourierFeatures

rng = np.random.default_rng(0)


def make_blobs(n=400):
    """Two overlapping Gaussian clouds (linearly separable-ish)."""
    a = rng.normal([-1.5, -1.5], 1.0, size=(n // 2, 2))
    b = rng.normal([1.5, 1.5], 1.0, size=(n // 2, 2))
    return np.vstack([a, b]), np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)


def make_circles(n=600):
    """Inner disc (class 1) inside a ring (class 0): NOT linearly separable."""
    r = np.r_[rng.uniform(0, 1, n // 2), rng.uniform(2, 3, n // 2)]
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    return X, np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)


def acc(y, p):
    return float((y == p).mean())


# 1. Linear SVM on blobs: objective must go down, accuracy must be high -------
X, y = make_blobs()
svm = PegasosSVM(lambda_=1e-2, n_epochs=15).fit(X, y)
h = svm.history_
print(f"[linear] objective: first epoch {h[0]:.4f} -> last epoch {h[-1]:.4f}")
assert h[-1] < h[0], "objective did not decrease"
a_lin = acc(y, svm.predict(X))
print(f"[linear] train accuracy = {a_lin:.3f}")
assert a_lin > 0.90

print(f"[margin] ||w|| = {svm.weight_norm():.3f}, margin width 2/||w|| = {svm.margin_width():.3f}, "
      f"support vectors = {svm.support_vector_mask(X, y).sum()} / {len(y)}")

# 2. Bigger lambda -> smaller ||w|| -> WIDER margin ----------------------------
widths = {}
for lam in [1e-3, 1e-2, 1e-1, 1.0]:
    m = PegasosSVM(lambda_=lam, n_epochs=15).fit(X, y)
    widths[lam] = m.margin_width()
    print(f"[lambda sweep] lambda={lam:<6g} margin width={m.margin_width():6.3f}  "
          f"#SV={m.support_vector_mask(X, y).sum()}")
ws = list(widths.values())
assert all(ws[i] < ws[i + 1] for i in range(len(ws) - 1)), "margin should grow with lambda"

# 3. Circles: linear fails, random Fourier features succeed --------------------
Xc, yc = make_circles()
lin = PegasosSVM(lambda_=1e-3, n_epochs=15).fit(Xc, yc)
rff = RandomFourierFeatures(gamma=0.5, n_components=200, random_state=1).fit(Xc)
nl = PegasosSVM(lambda_=1e-3, n_epochs=15).fit(rff.transform(Xc), yc)
a_c_lin = acc(yc, lin.predict(Xc))
a_c_rff = acc(yc, nl.predict(rff.transform(Xc)))
print(f"[circles] linear acc = {a_c_lin:.3f}   RFF acc = {a_c_rff:.3f}")
assert a_c_rff > a_c_lin + 0.2

# 4. Optional comparison with scikit-learn (BASELINE only) ----------------------
try:
    from sklearn.linear_model import SGDClassifier
    sk = SGDClassifier(loss="hinge", alpha=1e-2, random_state=0).fit(X, y)
    print(f"[sklearn SGD hinge] train accuracy = {acc(y, sk.predict(X)):.3f} (ours: {a_lin:.3f})")
except ImportError:
    pass

# 5. Figure -----------------------------------------------------------------------
os.makedirs("report/figures", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

ax[0].plot(range(1, len(h) + 1), h, marker="o")
ax[0].set(title="Objective J decreases during training", xlabel="epoch", ylabel="J(w, b)")

gx, gy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid = np.c_[gx.ravel(), gy.ravel()]
z = svm.decision_function(grid).reshape(gx.shape)
ax[1].contourf(gx, gy, z, levels=[-100, 0, 100], alpha=0.15, colors=["tab:blue", "tab:red"])
ax[1].contour(gx, gy, z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
sv = svm.support_vector_mask(X, y)
ax[1].scatter(*X[~sv].T, c=y[~sv], cmap="coolwarm", s=12)
ax[1].scatter(*X[sv].T, c=y[sv], cmap="coolwarm", s=30, edgecolors="k")
ax[1].set(title=f"Linear: margin & support vectors (circled)\nmargin width={svm.margin_width():.2f}",
          xlim=(-5, 5), ylim=(-5, 5))

z2 = nl.decision_function(rff.transform(grid)).reshape(gx.shape)
ax[2].contourf(gx, gy, z2, levels=[-100, 0, 100], alpha=0.15, colors=["tab:blue", "tab:red"])
ax[2].contour(gx, gy, z2, levels=[0], colors="k")
ax[2].scatter(*Xc.T, c=yc, cmap="coolwarm", s=10)
ax[2].set(title=f"Random Fourier features: non-linear boundary\nacc={a_c_rff:.2f} (linear only: {a_c_lin:.2f})",
          xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
plt.tight_layout()
plt.savefig("report/figures/svm_toy_demo.png", dpi=130)
print("Saved report/figures/svm_toy_demo.png")
print("ALL SVM TESTS PASSED")