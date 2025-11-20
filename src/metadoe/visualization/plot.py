import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import phate
import scprep
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

def plot_polyhedron(points, title, location, elev=30, azim=225, dpi=300, figsize=(10, 8)):
    hull = ConvexHull(points)

    # Create high-resolution figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color='blue', s=50)

    # Plot faces
    for simplex in hull.simplices:
        face = points[simplex]
        poly = Poly3DCollection(
            [face],
            alpha=0.5,
            facecolor='lightblue',
            edgecolor='blue'
        )
        ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    plt.title(title)
    plt.tight_layout()

    # Save at desired resolution
    plt.savefig(location, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_polygon_samples(points, samples, title, location):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    fig, ax = plt.subplots()

    # Plot the convex hull as a filled polygon
    polygon = Polygon(hull_points, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5)
    ax.add_patch(polygon)

    # Scatter samples
    ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.6, zorder=5, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(location)

def plot_polyhedron_samples(points, samples, title="My Title", location="convex_hull_3d.png",
                        elev=30, azim=225, dpi=300, figsize=(8, 6)):
    hull = ConvexHull(points)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Plot hull faces
    faces = []
    for simplex in hull.simplices:
        face = points[simplex]
        faces.append(face)

    poly = Poly3DCollection(
        faces,
        facecolor='lightblue',
        edgecolor='blue',
        alpha=0.5
    )
    ax.add_collection3d(poly)

    # Scatter samples
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        s=10,
        alpha=0.6,
        color='blue',
        zorder=5
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elev, azim=azim)

    # Optional: make axes roughly equal in 3D
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    max_range = max(
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0]
    ) / 2.0

    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(location, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_polygon(points, title, location):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    fig, ax = plt.subplots()

    # Plot the convex hull as a filled polygon
    polygon = Polygon(hull_points, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5)
    ax.add_patch(polygon)

    xmin, xmax = hull_points[:, 0].min(), hull_points[:, 0].max()
    ymin, ymax = hull_points[:, 1].min(), hull_points[:, 1].max()
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(location)

def apply_phate(positions):
    (T, n, N, K) = positions.shape
    reshaped_positions = np.reshape(positions, (T * n, N * K))
    time = np.repeat(np.arange(T), n)
    phate_op = phate.PHATE()
    Y_phate = phate_op.fit_transform(reshaped_positions)
    return Y_phate, time

def plot_phate(Y_phate, time, title, prefix):
    scprep.plot.scatter2d(Y_phate, figsize=(12,8), c=time, cmap="Spectral",
                      ticks=False, label_prefix="PHATE")

    scprep.plot.scatter2d(Y_phate, figsize=(12, 8), c=time, cmap="Spectral",
                        ticks=False, label_prefix="PHATE")

    plt.title(title, fontsize=16)
    plt.savefig(f"{prefix}_phate.png", bbox_inches='tight')