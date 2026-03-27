import os

import matplotlib
import numpy as np

# Non-interactive backend for CI / headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)


def entropy_coin(theta: np.ndarray) -> np.ndarray:
    """
    Entropía de una moneda Bernoulli como función de theta = P(cara):

        H(theta) = -theta log2 theta - (1-theta) log2(1-theta)

    Definida solo para 0 < theta < 1.
    """
    return -theta * np.log2(theta) - (1.0 - theta) * np.log2(1.0 - theta)


def main() -> None:
    # Evitar log(0) en los extremos
    theta = np.linspace(1e-3, 1.0 - 1e-3, 400)
    H = entropy_coin(theta)

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(theta, H, color="#2c7bb6", linewidth=2.5)

    ax.set_title("Entropia de una moneda")
    ax.set_xlabel("theta = P(cara)")
    ax.set_ylabel("H(theta) [bits]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.1)

    # Marcar el máximo teórico en theta = 0.5, H = 1
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.text(0.52, 1.02, "H(0.5)=1", fontsize=9)

    fig.tight_layout()

    out = os.path.join(OUTDIR, "week06_entropy_coin.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()

