import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("../results")

KD_FILE = RESULTS_DIR / "Cifar10_FD_Cifar10_FD_KD_r1_0.h5"
DKD_FILE = RESULTS_DIR / "Cifar10_FD_Cifar10_FD_DKD_r1_0.h5"

OUT_FIG_PATH = Path("fig_FD_cifar10_ratio1_KD_vs_DKD_curve.png")


def load_rs_test_acc(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        rs_test_acc = np.array(hf["rs_test_acc"])
    return rs_test_acc


def set_style():
    plt.rcParams.update(
        {
            "figure.figsize": (3.2, 2.4),
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 3,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def main():
    set_style()

    acc_kd = load_rs_test_acc(KD_FILE)
    acc_dkd = load_rs_test_acc(DKD_FILE)

    rounds_kd = np.arange(len(acc_kd))
    rounds_dkd = np.arange(len(acc_dkd))

    fig, ax = plt.subplots()

    ax.plot(rounds_kd, acc_kd, "-o", label="KD (ρ=1.0)", color="#1f77b4", markersize=3)
    ax.plot(rounds_dkd, acc_dkd, "--s", label="DKD (ρ=1.0)", color="#d62728", markersize=3)

    ax.set_xlabel("Global round")
    ax.set_ylabel("Global test accuracy")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(OUT_FIG_PATH)
    print(f"[INFO] saved curve figure to {OUT_FIG_PATH.resolve()}")


if __name__ == "__main__":
    main()
