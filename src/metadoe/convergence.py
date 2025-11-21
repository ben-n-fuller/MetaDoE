import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_scores(scores, min_t, max_t, name, prefix, save_fig=True, font_size=14):
    # Trim scores to match the animation range
    min_scores = np.log1p(np.min(scores, axis=1)[min_t:max_t])
    max_scores = np.log1p(np.max(scores, axis=1)[min_t:max_t])
    average_scores = np.log1p(np.mean(scores, axis=1)[min_t:max_t])
    timesteps = np.arange(min_t, max_t)

    # Create static scatter plot
    plt.figure(figsize=(10, 6))

    plt.plot(timesteps, min_scores, 'o-', label='Minimum Score', color='green')
    plt.plot(timesteps, average_scores, 'o-', label='Average Score', color='blue')
    plt.plot(timesteps, max_scores, 'o-', label='Maximum Score', color='red')

    plt.xlabel('Timestep', fontsize=font_size)
    plt.ylabel(r'$\log(1 + \text{score})$', fontsize=font_size)
    plt.title(name, fontsize=font_size + 4)
    plt.legend(fontsize=font_size - 2)
    plt.grid(True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f'{prefix}_scores.png', dpi=150)
    else:
        plt.show()


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