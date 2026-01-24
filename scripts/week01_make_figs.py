import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name: str):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print("saved:", path)

# ----------------------------
# 1) Supervised learning schematic
# ----------------------------
plt.figure(figsize=(12, 6))

# Boxes
boxes = {
    "Data\n$D=\\{(x_i,y_i)\\}_{i=1}^N$": (0.08, 0.35, 0.20, 0.25),
    "Model\n$f_\\theta: \\mathcal{X}\\to\\mathcal{Y}$": (0.40, 0.35, 0.20, 0.25),
    "Prediction\n$\\hat{y}=f_\\theta(x)$": (0.72, 0.35, 0.20, 0.25),
    "Loss / Metric\n$\\ell(y,\\hat{y})$  or  (Acc, AP) $\\uparrow$": (0.40, 0.05, 0.20, 0.20),
}

ax = plt.gca()
ax.set_axis_off()

for text, (x, y, w, h) in boxes.items():
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=14)

# Arrows
ax.annotate("", xy=(0.40, 0.475), xytext=(0.28, 0.475), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.72, 0.475), xytext=(0.60, 0.475), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.50, 0.35),  xytext=(0.50, 0.25),  arrowprops=dict(arrowstyle="->", lw=2))
ax.text(0.50, 0.30, "training loop", ha="center", va="center", fontsize=13)

plt.title("Supervised learning: learn $f(x)$ from labeled pairs $(x,y)$", fontsize=18)
plt.tight_layout(pad=1.2)
savefig("supervised_schematic.png")

# ----------------------------
# 2) Iris 2D view (PML uses Iris as canonical example)
# ----------------------------
iris = load_iris()
X = iris.data
y = iris.target
feat_names = iris.feature_names
class_names = iris.target_names

# Petal length vs petal width
pl = X[:, 2]
pw = X[:, 3]

plt.figure(figsize=(12, 6))
for c in np.unique(y):
    plt.scatter(pl[y == c], pw[y == c], label=class_names[c], s=40)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Iris dataset: simple 2D view (petal length vs petal width)", fontsize=18)
plt.legend()
plt.tight_layout(pad=1.2)
savefig("iris_2d.png")

# ----------------------------
# 3) Overfitting intuition: polynomial regression fits
# ----------------------------
rng = np.random.default_rng(0)
n = 30
x = np.linspace(-3, 3, n)
true_f = 0.5*x**3 - 0.8*x
y_obs = true_f + rng.normal(0, 1.0, size=n)

idx = rng.permutation(n)
train_idx = idx[:18]
test_idx = idx[18:]

xtr, ytr = x[train_idx], y_obs[train_idx]
xte, yte = x[test_idx], y_obs[test_idx]

def fit_poly(deg, xtr, ytr):
    Xtr = np.vander(xtr, N=deg+1, increasing=True)
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return w

def pred_poly(w, x):
    deg = len(w) - 1
    X = np.vander(x, N=deg+1, increasing=True)
    return X @ w

xgrid = np.linspace(x.min()-0.2, x.max()+0.2, 400)

plt.figure(figsize=(13.5, 6))
plt.scatter(xtr, ytr, label="train", s=55)
plt.scatter(xte, yte, label="test", s=55, marker="x")

for deg in [1, 3, 12]:
    w = fit_poly(deg, xtr, ytr)
    plt.plot(xgrid, pred_poly(w, xgrid), label=f"degree {deg}", linewidth=2)

plt.plot(xgrid, 0.5*xgrid**3 - 0.8*xgrid, linestyle="--", linewidth=2, label="true function")
plt.ylim(-15, 12)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Overfitting intuition: low vs high degree polynomial regression", fontsize=18)
plt.legend(loc="upper left")
plt.tight_layout(pad=1.2)
savefig("overfitting_polyfits.png")

# ----------------------------
# 4) Complexity vs generalization: train/test error vs degree (log scale)
# ----------------------------
degrees = list(range(0, 13))
train_mse, test_mse = [], []

for deg in degrees:
    w = fit_poly(deg, xtr, ytr)
    ytr_hat = pred_poly(w, xtr)
    yte_hat = pred_poly(w, xte)
    train_mse.append(np.mean((ytr - ytr_hat)**2))
    test_mse.append(np.mean((yte - yte_hat)**2))

plt.figure(figsize=(13, 6))
plt.plot(degrees, train_mse, marker="o", label="train MSE")
plt.plot(degrees, test_mse, marker="o", label="test MSE")
plt.yscale("log")
plt.xlabel("polynomial degree")
plt.ylabel("mean squared error (log scale)")
plt.title("Model complexity vs generalization: train error decreases, test error is U-shaped", fontsize=16)
plt.legend()
plt.tight_layout(pad=1.2)
savefig("error_vs_degree.png")

# ----------------------------
# 5) K-fold CV schematic (visual intuition)
# ----------------------------
K = 5
N = 40
fold_sizes = [N // K] * K
for i in range(N % K):
    fold_sizes[i] += 1

folds = []
start = 0
for fs in fold_sizes:
    folds.append((start, start+fs))
    start += fs

plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_axis_off()

y0 = 0.65
for k, (a, b) in enumerate(folds):
    # draw full bar outline
    ax.add_patch(plt.Rectangle((0.05, y0 - 0.05*k), 0.9, 0.03, fill=False, linewidth=1))
    # paint validation fold segment
    left = 0.05 + 0.9*(a/N)
    width = 0.9*((b-a)/N)
    ax.add_patch(plt.Rectangle((left, y0 - 0.05*k), width, 0.03, fill=True, alpha=0.35))
    ax.text(0.01, y0 - 0.05*k + 0.015, f"fold {k+1}", va="center", fontsize=11)

ax.text(0.05, 0.92, "K-fold cross-validation: each row uses a different validation fold", fontsize=14)
ax.text(0.05, 0.87, "(shaded = validation, unshaded = training)", fontsize=11)
plt.tight_layout(pad=1.2)
savefig("kfold_schematic.png")

# ----------------------------
# 6) Leakage demo: scaling on all data vs train-only (conceptual)
# ----------------------------
rng = np.random.default_rng(1)
N = 200
n_train = 150
n_test = N - n_train

# "presente" (train) vs "futuro" (test): split tipo temporal
X_train = rng.normal(0, 1, size=(n_train, 2))
X_test = rng.normal(0, 1, size=(n_test, 2))

# cambio de distribución solo en test (simula drift / futuro)
X_test[:, 0] += 3.5

X_all = np.vstack([X_train, X_test])

# BAD: fit scaler on ALL data
sc_all = StandardScaler().fit(X_all)
X_train_bad = sc_all.transform(X_train)
X_test_bad = sc_all.transform(X_test)

# GOOD: fit scaler ONLY on train
sc_tr = StandardScaler().fit(X_train)
X_train_good = sc_tr.transform(X_train)
X_test_good = sc_tr.transform(X_test)

plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train_bad[:, 0], X_train_bad[:, 1], s=18, label="entrenamiento (escalado con TODO)")
plt.scatter(X_test_bad[:, 0], X_test_bad[:, 1], s=18, marker="x", label="prueba (escalado con TODO)")
plt.title("FUGA: fit del escalador con TODO el dato")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_train_good[:, 0], X_train_good[:, 1], s=18, label="entrenamiento (fit en entrenamiento)")
plt.scatter(X_test_good[:, 0], X_test_good[:, 1], s=18, marker="x", label="prueba (fit en entrenamiento)")
plt.title("Correcto: fit del escalador SOLO en entrenamiento")
plt.legend()

# mismo encuadre para comparar visualmente
for ax in plt.gcf().axes:
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

plt.suptitle("Intuición: fuga por preprocesamiento (por qué el pipeline importa)", fontsize=16)
plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.96])
savefig("leakage_scaling.png")

# ----------------------------
# 7) PR curve + Average Precision (imbalanced classification)
# ----------------------------
rng = np.random.default_rng(2)
n = 600
# 10% positives
y = (rng.random(n) < 0.10).astype(int)

# scores: positives have higher mean, but overlap exists
scores = rng.normal(0, 1, size=n) + 1.2*y

prec, rec, _ = precision_recall_curve(y, scores)
ap = average_precision_score(y, scores)

plt.figure(figsize=(12, 6))
plt.plot(rec, prec, linewidth=2)
plt.xlabel("cobertura (recall)")
plt.ylabel("precisión")
plt.title(f"Curva Precisión–Cobertura (AP = {ap:.3f})", fontsize=18)
plt.tight_layout(pad=1.2)
savefig("pr_curve.png")

# ----------------------------
# 8) Selection bias: "best on test" gets optimistic
# ----------------------------
rng = np.random.default_rng(3)
true_err = 0.30
sigma = 0.06  # noise level of a finite test estimate

ms = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])
trials = 4000

best = []
q10 = []
q90 = []

for m in ms:
    # each trial: evaluate m candidate models on the same finite test set (noisy estimates)
    eps = rng.normal(0.0, sigma, size=(trials, m))
    est = true_err + eps
    b = np.min(est, axis=1)
    best.append(np.mean(b))
    q10.append(np.quantile(b, 0.10))
    q90.append(np.quantile(b, 0.90))

best = np.array(best)
q10 = np.array(q10)
q90 = np.array(q90)

plt.figure(figsize=(12, 6))
plt.plot(ms, best, marker="o", linewidth=2, label=r"$\mathbb{E}[\min_j \widehat{R}_{test}(f_j)]$")
plt.fill_between(ms, q10, q90, alpha=0.20, label="10–90% (por ruido)")
plt.axhline(true_err, linestyle="--", linewidth=2, color="black", label=r"$\min_j R(f_j)$ (referencia)")
plt.xscale("log")
plt.xlabel("número de modelos probados (m)")
plt.ylabel("error estimado en test (del elegido)")
plt.title("Sesgo por selección: elegir el 'mejor en test' produce optimismo", fontsize=16)
plt.legend()
plt.tight_layout(pad=1.2)
savefig("selection_bias.png")

print("\nDone. Figures are in:", OUTDIR)
