import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


FILE_NAME = "cifar10_FedKD_cifar10_KD_r0.10_0.h5"


def main() -> None:
    file_path = Path("../results") / FILE_NAME
    print(f"load from: {file_path}")

    if not file_path.exists():
        print("[ERROR] file does not exist")
        return

    with h5py.File(file_path, "r") as hf:
        keys = list(hf.keys())
        print("keys:", keys)
        if "rs_test_acc" not in hf:
            print("[WARN] rs_test_acc not found in file")
            return
        rs_test_acc = np.array(hf["rs_test_acc"])

    print("rs_test_acc length:", len(rs_test_acc))
    if len(rs_test_acc) == 0:
        return

    print("first 5:", rs_test_acc[:5])
    print("last 5:", rs_test_acc[-5:])
    print("best acc:", float(rs_test_acc.max()))
    print("last acc:", float(rs_test_acc[-1]))

    rounds = np.arange(1, len(rs_test_acc) + 1)
    plt.figure(figsize=(5, 3))
    plt.plot(rounds, rs_test_acc, "-o", markersize=2)
    plt.xlabel("Global round")
    plt.ylabel("Global test accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
