import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


# ===================== 配置区：按你的实验改 =====================

# 数据集和算法名字，对应 ../results/{DATASET}_{ALGO}_{goal}_{i}.h5
DATASET = "cifar10"   # 现在在 cifar10 上画图
ALGO = "FedKD"

# 蒸馏样本比例列表（distill_ratio），按你实际跑的来填
DISTILL_RATIOS: List[float] = [0.10, 0.25, 0.50, 1.00]

# 对每个比例，KD 和 DKD 对应的 goal（也就是你 main.py 里 -go 的取值）
# 例如：
#   KD:  python main.py ... -data cifar10 -algo FedKD --distill_type KD  --distill_ratio 0.25 -go cifar10_KD_r0.25 -t 1
#   DKD: python main.py ... -data cifar10 -algo FedKD --distill_type DKD --distill_ratio 0.25 -go cifar10_DKD_r0.25 -t 1
GOALS: Dict[float, Dict[str, str]] = {
    0.10: {"KD": "cifar10_KD_r0.10", "DKD": "cifar10_DKD_r0.10"},
    0.25: {"KD": "cifar10_KD_r0.25", "DKD": "cifar10_DKD_r0.25"},
    0.50: {"KD": "cifar10_KD_r0.50", "DKD": "cifar10_DKD_r0.50"},
    1.00: {"KD": "cifar10_KD_r1.00", "DKD": "cifar10_DKD_r1.00"},
}

# 每个 (KD/DKD, goal) 跑了多少次，对应 main.py 的 -t
NUM_RUNS = 1

# 使用 best acc 还是最后一轮 acc
USE_BEST_ACC = True  # True: max(rs_test_acc)，False: rs_test_acc[-1]

RESULTS_DIR = Path("../results")
OUT_FIG_PATH = Path("fig_main1_cifar10_acc_vs_ratio.png")


def set_paper_style():
    plt.rcParams.update({
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
        "lines.markersize": 4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_acc_for_config(goal: str) -> np.ndarray:
    """
    读取同一个 (DATASET, ALGO, goal) 多次 run 的精度，
    返回 shape [num_runs] 的 best/last acc 数组。
    """
    accs: List[float] = []
    for i in range(NUM_RUNS):
        file_name = f"{DATASET}_{ALGO}_{goal}_{i}.h5"
        file_path = RESULTS_DIR / file_name
        if not file_path.exists():
            print(f"[WARN] results file not found: {file_path}")
            continue
        with h5py.File(file_path, "r") as hf:
            rs_test_acc = np.array(hf["rs_test_acc"])
        if USE_BEST_ACC:
            acc_val = float(rs_test_acc.max())
        else:
            acc_val = float(rs_test_acc[-1])
        accs.append(acc_val)

    if len(accs) == 0:
        print(f"[WARN] no valid runs for goal={goal}")
        return np.array([])
    return np.array(accs)


def collect_acc_over_ratios():
    """
    对每个 distill_ratio 和 (KD/DKD) 收集 mean/std。
    返回：(ratios_array, kd_mean, kd_std, dkd_mean, dkd_std)
    """
    ratios = []
    kd_mean, kd_std = [], []
    dkd_mean, dkd_std = [], []

    for r in DISTILL_RATIOS:
        cfg = GOALS[r]
        accs_kd = load_acc_for_config(cfg["KD"])
        accs_dkd = load_acc_for_config(cfg["DKD"])

        if accs_kd.size == 0 or accs_dkd.size == 0:
            print(f"[WARN] skip ratio {r} due to missing KD/DKD results")
            continue

        ratios.append(r)

        kd_mean.append(accs_kd.mean())
        if accs_kd.size > 1:
            kd_std.append(accs_kd.std())
        else:
            kd_std.append(0.0)

        dkd_mean.append(accs_dkd.mean())
        if accs_dkd.size > 1:
            dkd_std.append(accs_dkd.std())
        else:
            dkd_std.append(0.0)

    return (np.array(ratios),
            np.array(kd_mean), np.array(kd_std),
            np.array(dkd_mean), np.array(dkd_std))


def plot_main_fig1():
    set_paper_style()

    ratios, kd_mean, kd_std, dkd_mean, dkd_std = collect_acc_over_ratios()
    if ratios.size == 0:
        print("[ERROR] No valid data to plot.")
        return

    fig, ax = plt.subplots()

    ax.errorbar(
        ratios, kd_mean, yerr=kd_std,
        fmt="-o", capsize=3, label="KD",
        color="#1f77b4"
    )

    ax.errorbar(
        ratios, dkd_mean, yerr=dkd_std,
        fmt="--s", capsize=3, label="DKD",
        color="#d62728"
    )

    ax.set_xlabel("Distillation sample ratio $\\rho$")
    ax.set_ylabel("Global test accuracy")
    ax.set_xticks(ratios)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(OUT_FIG_PATH)
    print(f"[INFO] saved main figure 1 to {OUT_FIG_PATH.resolve()}")


if __name__ == "__main__":
    plot_main_fig1()
