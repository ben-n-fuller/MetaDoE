import numpy as np
import matplotlib.pyplot as plt

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