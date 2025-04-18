import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import phate
import scprep
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

def plot_scores(scores, min_t, max_t, name, prefix, save_fig=True):
    # Trim scores to match the animation range
    min_scores = np.log1p(np.min(scores, axis=1)[min_t:max_t])
    max_scores = np.log1p(np.max(scores, axis=1)[min_t:max_t])
    average_scores = np.log1p(np.mean(scores, axis=1)[min_t:max_t])
    # min_scores = np.min(scores, axis=1)[min_t:max_t]
    # max_scores = np.max(scores, axis=1)[min_t:max_t]
    # average_scores = np.mean(scores, axis=1)[min_t:max_t]
    timesteps = np.arange(min_t, max_t)

    # Create static scatter plot
    plt.figure(figsize=(10, 6))

    plt.plot(timesteps, min_scores, 'o-', label='Minimum Score', color='green')
    plt.plot(timesteps, average_scores, 'o-', label='Average Score', color='blue')
    plt.plot(timesteps, max_scores, 'o-', label='Maximum Score', color='red')
    # plt.yscale('log')

    plt.xlabel('Timestep')
    # plt.ylabel('Log Fitness Score')
    plt.ylabel(r'$\log(1 + \text{score})$')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{prefix}_scores.png', dpi=150)
    else:
        plt.show()

def barycentric_embed(points):
    n = points.shape[1]
    
    # Create embedding corners
    corners = np.vstack([np.eye(n - 1), np.zeros((1, n - 1))])  # shape (n, n-1)
    
    # Apply affine map
    return points @ corners

def plot_polyhedron_sample(points, samples, title, location, elev=30, azim=225):
    # Compute the convex hull
    hull = ConvexHull(points)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=50)

    # Plot the convex hull faces
    for simplex in hull.simplices:
        face = points[simplex]
        poly = Poly3DCollection([face], alpha=0.5, facecolor='lightblue', edgecolor='blue')
        ax.add_collection3d(poly)

    # Scatter plot using the 3 columns as x, y, z
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    plt.title(title)

    plt.savefig(location)

def plot_polyhedron(points, title, location, elev=30, azim=225):
    # Define the 8 custom vertices manually
    # points = np.load("verts.npy")
    # Compute the convex hull
    hull = ConvexHull(points)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=50)

    # Plot the convex hull faces
    for simplex in hull.simplices:
        face = points[simplex]
        poly = Poly3DCollection([face], alpha=0.5, facecolor='lightblue', edgecolor='blue')
        ax.add_collection3d(poly)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(location)

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

def plot_objective(objective, name):
    x = np.linspace(-100, 100, 1000)
    y = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Stack into (n_points, 3)
    points = np.stack([X, Y, Z], axis=-1)

    # Evaluate Griewank
    F = objective(points)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, F, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel(f"{name}(x₁, x₂, 0)")
    ax.set_title(f"{name} Function Slice (x₃ = 0)")

    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{name}_slice.png")

def plot_particle_movement(positions_og, min_t, max_t, name, objective, fps):
    # Trim positions to desired frame range
    positions = positions_og[min_t:max_t, :, :]

    # Setup 2D plot
    fig, ax = plt.subplots()

    # Generate a grid over the XZ plane
    x = np.linspace(-100, 100, 300)
    z = np.linspace(-100, 100, 300)
    X_grid, Z_grid = np.meshgrid(x, z)

    # Create (N, 3) input with Y fixed at 0
    points = np.stack([X_grid, np.zeros_like(X_grid), Z_grid], axis=-1)
    F = objective(points)

    # Plot contours or heatmap
    contour = ax.contourf(X_grid, Z_grid, F, levels=100, cmap='viridis', alpha=0.7)

    # Initial scatter plot for particles
    scat = ax.scatter([], [], color='red', s=20)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Timestep 0")

    # Animation update function
    def update(frame):
        pos = positions[frame]
        xz = np.stack([pos[:, 0], pos[:, 2]], axis=-1)  # X and Z
        scat.set_offsets(xz)
        ax.set_title(f"Timestep {frame}")
        return scat,

    # Create animation
    anim = FuncAnimation(fig, update, frames=range(positions.shape[0]), interval=50)

    # Save as GIF
    anim.save(f"particle_motion_with_{name}.gif", writer=PillowWriter(fps=fps))

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