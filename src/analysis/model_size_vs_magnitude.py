"""Plots model size (log-scale, billions of parameters) against the mean magnitude of change
in MMAP (mean |diff| from baseline; see significance_testing.magnitude_mixed_lm) for each
distractor sentiment, using the "size" ablation experiments plus the main low-ambiguity
experiment's open-source models (which fill in the 3-4B gap in the ablation's size range;
GPT-4.1 is excluded since its parameter count isn't publicly known), and reports the
correlation between size and magnitude for each distractor type.

Usage: python -m src.analysis.model_size_vs_magnitude
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this runs as a script, not a notebook,
                        # so plt.show() would otherwise block waiting for a GUI window

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.config import PATH_FIG
from src.analysis.moralchoice_analysis import calculate_moralchoice_results
from src.analysis.results_cache import load_or_compute

OUTPUT_DIR = PATH_FIG / "moralchoice"
SCENARIO_FILE = "moralchoice_low_ambiguity.csv"

# billions of parameters, from each model's published size
MODEL_SIZES_B = {
    "gemma-3-27b-it": 27.0,
    "gemma-3-4b-it": 4.0,
    "gemma-3-1b-it": 1.0,
    "gemma-3-270m-it": 0.27,
    "Llama-3.2-11B-Vision-Instruct": 11.0,
    "Llama-3.2-3B-Instruct": 3.0,
    "Llama-3.2-1B-Instruct": 1.0,
    "Qwen3-32B": 32.0,
    "Qwen3-4B": 4.0,
    "Qwen3-1.7B": 1.7,
    "Qwen3-0.6B": 0.6,
}

# matches the "size" ablation configs used throughout this pipeline (e.g. compute_all_results.py)
SIZE_CONFIGS = [
    ("ablation/size/google_gemma-3-27b-it_moralchoice_low_ambiguity.csv", "gemma-3-27b-it"),
    ("ablation/size/google_gemma-3-1b-it_moralchoice_low_ambiguity.csv", "gemma-3-1b-it"),
    ("ablation/size/google_gemma-3-270m-it_moralchoice_low_ambiguity.csv", "gemma-3-270m-it"),
    ("ablation/size/meta-llama_Llama-3.2-11B-Vision-Instruct_moralchoice_low_ambiguity.csv", "Llama-3.2-11B-Vision-Instruct"),
    ("ablation/size/meta-llama_Llama-3.2-1B-Instruct_low_ambiguity.csv", "Llama-3.2-1B-Instruct"),
    ("ablation/size/Qwen_Qwen3-32B_moralchoice_low_ambiguity.csv", "Qwen3-32B"),
    ("ablation/size/Qwen_Qwen3-1.7B_low_ambiguity.csv", "Qwen3-1.7B"),
    ("ablation/size/Qwen_Qwen3-0.6B_low_ambiguity.csv", "Qwen3-0.6B"),
]

# main low-ambiguity experiment's open-source models, added to fill the 3-4B gap in the
# ablation's size range above (GPT-4.1 excluded: its parameter count isn't public)
MAIN_CONFIGS = [
    ("main/moralchoice_low_ambiguity/google_gemma-3-4b-it_moralchoice_low_ambiguity.csv", "gemma-3-4b-it"),
    ("main/moralchoice_low_ambiguity/meta-llama_Llama-3.2-3B-Instruct_moralchoice_low_ambiguity.csv", "Llama-3.2-3B-Instruct"),
    ("main/moralchoice_low_ambiguity/Qwen_Qwen3-4B_moralchoice_low_ambiguity.csv", "Qwen3-4B"),
]

DISTRACTOR_COLORS = {"positive": "green", "neutral": "orange", "negative": "red"}


def main():
    sizes = []
    magnitudes = {"positive": [], "neutral": [], "negative": []}

    for csv_path, model_key in SIZE_CONFIGS + MAIN_CONFIGS:
        results = load_or_compute(calculate_moralchoice_results, csv_path, SCENARIO_FILE, sig_testing=True)
        sig_magnitude = results['all']['sig_magnitude']
        sizes.append(MODEL_SIZES_B[model_key])
        for condition in magnitudes:
            magnitudes[condition].append(sig_magnitude[f'baseline_vs_{condition}']['coef'])

    sizes = np.array(sizes)
    log_sizes = np.log10(sizes)
    # line of best fit in log10(size)-space, matching the Pearson correlation's basis, drawn
    # across the full observed size range rather than connecting the (unordered) data points
    fit_xs = np.linspace(log_sizes.min(), log_sizes.max(), 100)

    # small vertical offsets (in points) to keep the three r/p labels from overlapping where
    # their fitted lines end up close together at the right edge of the plot
    LABEL_OFFSETS = {"positive": 10, "neutral": -10, "negative": 0}

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, color in DISTRACTOR_COLORS.items():
        ys = np.array(magnitudes[condition])
        ax.scatter(sizes, ys, color=color, label=condition.capitalize(), zorder=3)
        slope, intercept = np.polyfit(log_sizes, ys, 1)
        fit_ys = slope * fit_xs + intercept
        ax.plot(10 ** fit_xs, fit_ys, color=color, alpha=0.6, linestyle='--', zorder=2)

        pearson_r, pearson_p = pearsonr(log_sizes, ys)
        ax.annotate(
            f"$r$={pearson_r:+.2f}, $p$={pearson_p:.3f}",
            xy=(10 ** fit_xs[-1], fit_ys[-1]),
            xytext=(8, LABEL_OFFSETS[condition]),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )

    ax.set_xscale('log')
    ax.set_xlabel("Model Size (billions of parameters, log scale)")
    ax.set_ylabel("Mean Magnitude of Change in MMAP")
    ax.set_title("Model Size vs. Magnitude of Change by Distractor Sentiment")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)  # leave room for the r/p labels past the last point
    ax.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "model_size_vs_magnitude.svg"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {output_path}")

    print()
    print("Correlation between model size and magnitude of change, by distractor type:")
    print("(Pearson computed on log10(size), since sizes span orders of magnitude; Spearman")
    print(" is a scale-invariant rank correlation, shown for robustness against that choice.)")
    for condition in magnitudes:
        ys = np.array(magnitudes[condition])
        pearson_r, pearson_p = pearsonr(log_sizes, ys)
        spearman_r, spearman_p = spearmanr(sizes, ys)
        print(
            f"  {condition.capitalize():9s}: Pearson r (log-size) = {pearson_r:+.3f} (p={pearson_p:.3f}), "
            f"Spearman rho = {spearman_r:+.3f} (p={spearman_p:.3f})"
        )


if __name__ == "__main__":
    main()
