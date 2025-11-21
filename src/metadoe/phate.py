import numpy as np
from matplotlib import pyplot as plt
import phate
import scprep

def apply_phate(file_name):
    data = np.load(file_name)
    positions = data["positions"]
    (T, n, N, K) = positions.shape
    reshaped_positions = np.reshape(positions, (T * n, N * K))
    time = np.repeat(np.arange(T), n)
    phate_op = phate.PHATE()
    Y_phate = phate_op.fit_transform(reshaped_positions)
    return Y_phate, time

def plot_phate(Y_phate, time, title, prefix, font_size=16):
    scprep.plot.scatter2d(Y_phate, figsize=(12,8), c=time, cmap="Spectral",
                      ticks=False, label_prefix="PHATE")

    scprep.plot.scatter2d(Y_phate, figsize=(12, 8), c=time, cmap="Spectral",
                        ticks=False, label_prefix="PHATE")

    plt.title(title, fontsize=font_size)
    plt.savefig(f"{prefix}_phate.png", bbox_inches='tight')