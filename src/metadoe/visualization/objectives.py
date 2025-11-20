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

def plot_benchmark_surface(
    func,
    x_range,
    y_range,
    dim,
    slice_dims=(0, 1),
    fixed_coords=None,
    n=500,
    title=None,
    save_path=None,
    dpi=300,
):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    Xg, Yg = np.meshgrid(x, y)

    if fixed_coords is None:
        base = np.zeros((n, n, dim), dtype=float)
    else:
        fixed_coords = np.asarray(fixed_coords, dtype=float)
        assert fixed_coords.shape == (dim,)
        base = np.broadcast_to(fixed_coords, (n, n, dim)).copy()

    i, j = slice_dims
    base[..., i] = Xg
    base[..., j] = Yg

    Z = func(base)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Xg, Yg, Z,
        rstride=2, cstride=2,
        linewidth=0,
        antialiased=True,
        cmap="viridis"
    )

    z_min = np.min(Z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    if title is not None:
        ax.set_title(title)

    ax.set_zlim(z_min, np.max(Z))

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label="Elevation")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    plt.tight_layout()
    plt.show()