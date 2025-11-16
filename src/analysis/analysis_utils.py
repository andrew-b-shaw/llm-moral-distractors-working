import numpy as np
import matplotlib.pyplot as plt

def plot_token_prob_histograms(
    dist1, 
    dist2, 
    title1='token_prob_a', 
    title2='token_prob_b', 
    save_path=None, 
    log_scale=False
):
    if isinstance(dist1.iloc[0], (list, np.ndarray)):
        dist1 = np.concatenate(dist1.dropna().values)
    if isinstance(dist2.iloc[0], (list, np.ndarray)):
        dist2 = np.concatenate(dist2.dropna().values)

    dist1 = np.array(dist1, dtype=float)
    dist2 = np.array(dist2, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(dist1, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Histogram of {title1}')
    axes[0].set_xlabel(title1)
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 1)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    if log_scale:
        axes[0].set_xscale('log')
        axes[0].set_xlim(1e-9, 1)

    axes[1].hist(dist2, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title(f'Histogram of {title2}')
    axes[1].set_xlabel(title2)
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 1)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    if log_scale:
        axes[1].set_xscale('log')
        axes[1].set_xlim(1e-9, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
