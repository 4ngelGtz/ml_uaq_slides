"""
Week 15 - Trees, Bagging and Random Forests (pedagogical Matplotlib).

Generates PNGs in slides/figs/ with prefix week15_*.
Uses only matplotlib + numpy + sklearn (no seaborn).

Note: no se encontraron notebooks locales del libro Murphy ni del repo probml
para los temas de esta semana. Las figuras se generan con datasets sinteticos
reproducibles (sklearn + numpy) inspirados conceptualmente en el Cap. 18 de
"Probabilistic Machine Learning: An Introduction" (Murphy).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification, make_moons  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import (  # noqa: E402
    PartialDependenceDisplay,
    permutation_importance,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)

# Paleta consistente con el resto del curso.
C_FG = "#1f2937"
C_ACCENT = "#2563eb"      # azul (clase 0 / modelo base)
C_RES = "#dc2626"         # rojo (clase 1 / modelo alterno)
C_PROJ = "#059669"        # verde (modelo de ensamble)
C_GRID = "#e5e7eb"
C_WARN = "#d97706"        # naranja
C_PURPLE = "#7c3aed"

# Mapas de color para fronteras: azul claro vs rojo claro.
CMAP_BG = plt.cm.RdBu_r


def _save_fig(fig, fname: str, *, pad_inches: float = 0.22) -> None:
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight",
                pad_inches=pad_inches)
    plt.close(fig)
    print("saved:", out)


def _style_ax(ax) -> None:
    ax.grid(True, linestyle=":", alpha=0.4, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C_FG)


def _plot_decision_boundary(ax, model, X, y, *, title: str,
                            grid_step: float = 0.02) -> None:
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=20, cmap=CMAP_BG, alpha=0.55,
                    vmin=0, vmax=1)
    else:
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                    cmap=CMAP_BG, alpha=0.55)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C_ACCENT, s=14,
               edgecolors="white", linewidths=0.4, label="clase 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C_RES, s=14,
               edgecolors="white", linewidths=0.4, label="clase 1")
    ax.set_title(title, fontsize=10, color=C_FG, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C_FG)


# ------------------------------------------------------------------
# 1) Logistic regression vs Decision tree en un dataset no lineal
# ------------------------------------------------------------------
def fig_logistic_vs_tree() -> None:
    """Inspirado en la motivacion del Cap. 18.1 de Murphy: cuando una
    frontera lineal no basta, un arbol puede capturar la estructura."""
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    lr = LogisticRegression().fit(X, y)
    tree = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    _plot_decision_boundary(axes[0], lr, X, y,
                            title="Logistic regression (frontera lineal)")
    _plot_decision_boundary(axes[1], tree, X, y,
                            title="Decision tree (fronteras axis-aligned)")
    fig.suptitle("Cuando una linea recta no basta",
                 fontsize=11.5, color=C_FG, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "week15_frontera_logistic_vs_tree.png")


# ------------------------------------------------------------------
# 2) Comparacion de profundidad: max_depth = 1, 2, 5
# ------------------------------------------------------------------
def fig_tree_depth_comparison() -> None:
    """Cada split anade una particion rectangular del espacio."""
    X, y = make_moons(n_samples=300, noise=0.25, random_state=7)
    depths = [1, 2, 5]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, d in zip(axes, depths):
        tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        _plot_decision_boundary(ax, tree, X, y,
                                title=f"max_depth = {d}")
    fig.suptitle("Mas profundidad -> particion mas fina del espacio",
                 fontsize=11.5, color=C_FG, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "week15_tree_depth_comparison.png")


# ------------------------------------------------------------------
# 3) Sobreajuste: train/test accuracy vs max_depth
# ------------------------------------------------------------------
def fig_overfitting_tree_depth() -> None:
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=4, n_redundant=2,
        n_repeated=0, n_classes=2, class_sep=1.2, flip_y=0.05,
        random_state=42,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    depths = list(range(1, 21))
    acc_tr, acc_te = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        acc_tr.append(m.score(Xtr, ytr))
        acc_te.append(m.score(Xte, yte))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(depths, acc_tr, marker="o", color=C_ACCENT, linewidth=1.6,
            label="train accuracy")
    ax.plot(depths, acc_te, marker="s", color=C_RES, linewidth=1.6,
            label="test accuracy")
    ax.set_xlabel("max_depth", fontsize=10, color=C_FG)
    ax.set_ylabel("accuracy", fontsize=10, color=C_FG)
    ax.set_title("Un arbol profundo memoriza el train",
                 fontsize=11, color=C_FG, pad=8)
    ax.legend(fontsize=9, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week15_overfitting_tree_depth.png")


# ------------------------------------------------------------------
# 4) Demo conceptual de bagging / bootstrap
# ------------------------------------------------------------------
def fig_bagging_bootstrap_demo() -> None:
    """Tres arboles distintos entrenados sobre tres muestras bootstrap
    distintas + voto promedio. Ilustra reduccion de varianza."""
    X, y = make_moons(n_samples=300, noise=0.30, random_state=1)
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    boots = []
    for i, ax in enumerate(axes[:3]):
        idx = rng.integers(0, len(X), size=len(X))  # bootstrap con reemplazo
        tree = DecisionTreeClassifier(max_depth=None,
                                      random_state=i).fit(X[idx], y[idx])
        boots.append(tree)
        _plot_decision_boundary(ax, tree, X, y,
                                title=f"Arbol bootstrap #{i + 1}")

    # Promedio de probabilidades como ensamble manual.
    class _Avg:
        def __init__(self, trees):
            self.trees = trees

        def predict_proba(self, X):
            return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    _plot_decision_boundary(axes[3], _Avg(boots), X, y,
                            title="Promedio (bagging)")
    fig.suptitle("Cada bootstrap produce un arbol distinto; el promedio se estabiliza",
                 fontsize=11.5, color=C_FG, y=1.04)
    fig.tight_layout()
    _save_fig(fig, "week15_bagging_bootstrap_demo.png")


# ------------------------------------------------------------------
# 5) Random forest vs un solo arbol
# ------------------------------------------------------------------
def fig_rf_vs_tree() -> None:
    X, y = make_moons(n_samples=400, noise=0.30, random_state=3)
    tree = DecisionTreeClassifier(max_depth=None, random_state=0).fit(X, y)
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                random_state=0, n_jobs=1).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    _plot_decision_boundary(axes[0], tree, X, y,
                            title="Arbol unico (alta varianza)")
    _plot_decision_boundary(axes[1], rf, X, y,
                            title="Random forest (200 arboles)")
    fig.suptitle("Promediar muchos arboles suaviza la frontera",
                 fontsize=11.5, color=C_FG, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "week15_rf_vs_tree_boundary.png")


# ------------------------------------------------------------------
# 6) OOB / test score vs n_estimators
# ------------------------------------------------------------------
def fig_rf_n_estimators_oob() -> None:
    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=6, n_redundant=3,
        n_classes=2, class_sep=1.0, flip_y=0.05, random_state=42,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    n_list = [5, 10, 25, 50, 100, 200, 400, 800]
    oob, test = [], []
    for n in n_list:
        rf = RandomForestClassifier(
            n_estimators=n, oob_score=True, bootstrap=True,
            max_features="sqrt", random_state=0, n_jobs=1,
        ).fit(Xtr, ytr)
        oob.append(rf.oob_score_)
        test.append(rf.score(Xte, yte))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(n_list, oob, marker="o", color=C_PROJ, linewidth=1.6,
            label="OOB score")
    ax.plot(n_list, test, marker="s", color=C_RES, linewidth=1.6,
            label="Test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("n_estimators (escala log)", fontsize=10, color=C_FG)
    ax.set_ylabel("accuracy", fontsize=10, color=C_FG)
    ax.set_title("Mas arboles -> menor varianza, hasta saturar",
                 fontsize=11, color=C_FG, pad=8)
    ax.legend(fontsize=9, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week15_rf_n_estimators_oob.png")


# ------------------------------------------------------------------
# 7) Feature importance: impurity vs permutation
# ------------------------------------------------------------------
def _credit_dataset(n: int = 1500, seed: int = 0):
    """Dataset sintetico de riesgo de credito (no es del libro; es
    didactico). Algunas variables son senaladamente predictivas, otras
    son ruido, y dos son moderadamente correlacionadas."""
    rng = np.random.default_rng(seed)
    ingreso = rng.lognormal(mean=10.0, sigma=0.5, size=n)
    deuda = rng.lognormal(mean=9.0, sigma=0.7, size=n)
    util = np.clip(0.4 * (deuda / ingreso) + 0.05 * rng.normal(size=n),
                   0, 1.5)  # correlacionada con deuda/ingreso
    atrasos = rng.poisson(lam=0.6, size=n)
    antig = rng.gamma(shape=2.0, scale=20.0, size=n)
    consultas = rng.poisson(lam=1.5, size=n)
    ruido = rng.normal(size=n)  # variable irrelevante

    z = (
        + 1.6 * (deuda / ingreso)
        + 0.9 * util
        + 0.8 * atrasos
        - 0.02 * antig
        + 0.15 * consultas
        - 4.0
    )
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(int)
    X = np.column_stack([ingreso, deuda, util, atrasos, antig,
                         consultas, ruido])
    feat = ["ingreso", "deuda", "utilizacion",
            "atrasos_previos", "antiguedad",
            "consultas_recientes", "ruido"]
    return X, y, feat


def fig_feature_importance() -> None:
    X, y, feat = _credit_dataset(n=2000, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", random_state=0,
        n_jobs=1,
    ).fit(Xtr, ytr)

    imp = rf.feature_importances_
    perm = permutation_importance(
        rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=1,
    ).importances_mean

    order = np.argsort(imp)[::-1]
    feat_o = [feat[i] for i in order]
    imp_o = imp[order]
    perm_o = perm[order]

    x = np.arange(len(feat))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - width / 2, imp_o, width=width, color=C_ACCENT,
           label="Impurity-based")
    ax.bar(x + width / 2, perm_o, width=width, color=C_RES,
           label="Permutation (test)")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_o, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("importancia", fontsize=10, color=C_FG)
    ax.set_title("Dos formas de medir importancia: a veces no coinciden",
                 fontsize=11, color=C_FG, pad=8)
    ax.legend(fontsize=9, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week15_feature_importance_comparison.png")


# ------------------------------------------------------------------
# 8) Partial dependence plots
# ------------------------------------------------------------------
def fig_partial_dependence() -> None:
    X, y, feat = _credit_dataset(n=2000, seed=0)
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", random_state=0,
        n_jobs=1,
    ).fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    PartialDependenceDisplay.from_estimator(
        rf, X, features=[feat.index("utilizacion")],
        feature_names=feat, ax=axes[0], grid_resolution=40,
        line_kw={"color": C_ACCENT, "linewidth": 1.8},
    )
    PartialDependenceDisplay.from_estimator(
        rf, X, features=[feat.index("atrasos_previos")],
        feature_names=feat, ax=axes[1], grid_resolution=40,
        line_kw={"color": C_RES, "linewidth": 1.8},
    )
    for ax in axes:
        _style_ax(ax)
        ax.set_title(ax.get_title(), fontsize=10, color=C_FG)
    fig.suptitle("Partial dependence: prediccion promedio vs una feature",
                 fontsize=11.5, color=C_FG, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "week15_partial_dependence_example.png")


# ------------------------------------------------------------------
# 9) Sesgo-varianza ilustrado con arboles: cortos vs profundos
# ------------------------------------------------------------------
def fig_bias_variance_trees() -> None:
    """Mismo algoritmo, 4 remuestreos distintos del mismo problema.
    Fila 1: arboles cortos (profundidad maxima 2) -> casi identicos
    (alto sesgo, baja varianza). Fila 2: arboles profundos (sin podar)
    -> muy distintos entre si (bajo sesgo, alta varianza)."""
    X, y = make_moons(n_samples=300, noise=0.30, random_state=11)
    rng = np.random.default_rng(2024)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))
    for col in range(4):
        idx = rng.integers(0, len(X), size=len(X))
        Xb, yb = X[idx], y[idx]

        shallow = DecisionTreeClassifier(
            max_depth=2, random_state=0,
        ).fit(Xb, yb)
        deep = DecisionTreeClassifier(
            max_depth=None, random_state=0,
        ).fit(Xb, yb)

        _plot_decision_boundary(
            axes[0, col], shallow, X, y,
            title=f"profundidad maxima 2  (muestra {col + 1})",
        )
        _plot_decision_boundary(
            axes[1, col], deep, X, y,
            title=f"sin podar  (muestra {col + 1})",
        )

    # Etiquetas laterales explicativas.
    fig.text(0.005, 0.74, "Alto sesgo\nBaja varianza",
             fontsize=11, color=C_FG, ha="left", va="center",
             rotation=90, weight="bold")
    fig.text(0.005, 0.28, "Bajo sesgo\nAlta varianza",
             fontsize=11, color=C_RES, ha="left", va="center",
             rotation=90, weight="bold")

    fig.suptitle(
        "El mismo problema, 4 remuestreos: arboles cortos casi "
        "identicos vs arboles profundos muy distintos",
        fontsize=11.5, color=C_FG, y=1.01,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    _save_fig(fig, "week15_bias_variance_trees.png")


def main() -> None:
    fig_logistic_vs_tree()
    fig_tree_depth_comparison()
    fig_overfitting_tree_depth()
    fig_bagging_bootstrap_demo()
    fig_rf_vs_tree()
    fig_rf_n_estimators_oob()
    fig_feature_importance()
    fig_partial_dependence()
    fig_bias_variance_trees()


if __name__ == "__main__":
    main()
