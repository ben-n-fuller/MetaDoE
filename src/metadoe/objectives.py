import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def griewank(X):
    X = np.asarray(X)
    j = np.sqrt(np.arange(1, 4))
    term1 = np.sum(X**2, axis=-1) / 4000
    term2 = np.prod(np.cos(X / j), axis=-1)
    return term1 - term2 + 1

def ackley(X):
    X = np.asarray(X)
    d = X.shape[-1]
    sum_sq = np.sum(X**2, axis=-1)
    cos_sum = np.sum(np.cos(2 * np.pi * X), axis=-1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(cos_sum / d) + 20 + np.e

def rastrigin(X):
    X = np.asarray(X)
    A = 10
    D = X.shape[-1]
    return A * D + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

def rosenbrock(X):
    X = np.asarray(X)
    return np.sum(100 * (X[..., 1:] - X[..., :-1]**2)**2 + (1 - X[..., :-1])**2, axis=-1)

def mesh(xmin, xmax, ymin, ymax, n=300):
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_objective(objective, name, path_prefix=".", dpi=300, font_size=16, figsize=(12, 10)):
    x = np.linspace(-100, 100, 1000)
    y = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Stack into (n_points, 3)
    points = np.stack([X, Y, Z], axis=-1)

    # Evaluate
    F = objective(points)

    # Plot
    fig = plt.figure(figsize=figsize)  # no constrained_layout
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, F, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_title(f"{name} Function Slice (x₃ = 0)", fontsize=font_size)

    # Make room on the right for the z-axis label
    fig.subplots_adjust(
        left=0.05,
        right=0.8,   # <--- pull axes left, more margin on right
        bottom=0.08,
        top=0.92,
    )

    # Force a draw so layout knows true text extents
    fig.canvas.draw()

    fig.savefig(f"{path_prefix}/{name}_slice.png", dpi=dpi)