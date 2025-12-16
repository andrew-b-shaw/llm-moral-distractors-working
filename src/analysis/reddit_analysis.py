"""
Analyze Reddit-style verdict CSVs and produce summary statistics and plots.

The script loads every CSV under ``src/analysis/reddit_csvs`` and computes:
1) Verdict mix by distractor polarity (neg/neu/pos) with a focus on ESH rates.
2) Distribution of the dominant moral foundation per response (bar chart).
3) Radar/spider chart of average moral foundation scores per model.

Outputs are written to ``fig/reddit`` relative to the repo root.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MORAL_COLS = ["ch_score", "fc_score", "lb_score", "as_score", "pd_score"]
VERDICT_PALETTE = {
    "YTA": "#4c78a8",   # slate blue
    "NTA": "#9ecae9",   # light steel blue
    "ESH": "#b37ac5",   # violet
    "NAH": "#f28e2c",   # amber
    "INFO": "#8c564b",  # taupe
}
DISTRACTOR_PALETTE = {
    "baseline": "#7f7f7f",   # gray
    "pos": "#009900",        # bright green
    "neu": "#f4a300",        # orange/yellow
    "neg": "#e50000",        # red
}
FOUNDATION_LABELS = {
    "ch_score": "Care/Harm",
    "fc_score": "Fairness/Cheating",
    "lb_score": "Loyalty/Betrayal",
    "as_score": "Authority/Subversion",
    "pd_score": "Purity/Degradation",
}


def repo_root() -> Path:
    """Return the repository root (parent of src)."""
    return Path(__file__).resolve().parents[2]


def load_reddit_frames(data_dir: Path) -> pd.DataFrame:
    """Load and concat all reddit CSVs with cleaning and helper columns."""
    frames: List[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["source_file"] = path.name
        df["model"] = df["model_id"].astype(str)
        df["distractor_type"] = (
            df["distractor_id"]
            .astype(str)
            .str.extract(r"txt_(neg|neu|pos)", expand=False)
            .fillna("baseline")  # treat missing distractor_id as baseline/no distractor
        )
        # Ensure numeric moral columns for safety.
        for col in MORAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def esh_share_by_distractor(df: pd.DataFrame) -> pd.DataFrame:
    """Compute share of verdicts by distractor type for each model."""
    counts = (
        df.groupby(["model", "distractor_type", "verdict"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["model", "distractor_type"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts


def plot_esh_spike(counts: pd.DataFrame, out_path: Path) -> None:
    """Plot ESH rate by distractor type per model."""
    esh = counts[counts["verdict"] == "ESH"].copy()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(
        data=esh,
        x="distractor_type",
        y="share",
        hue="model",
    )
    ax.set_title("ESH share by distractor polarity")
    ax.set_xlabel("Distractor polarity")
    ax.set_ylabel("Proportion of responses labeled ESH")
    ax.set_ylim(0, esh["share"].max() * 1.15 if not esh.empty else 1)
    plt.legend(title="Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def add_top_foundation(df: pd.DataFrame) -> pd.DataFrame:
    """Add column with dominant moral foundation per response."""
    top = df[MORAL_COLS].idxmax(axis=1)
    df = df.copy()
    df["top_foundation"] = top.map(FOUNDATION_LABELS)
    return df


def plot_foundation_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of top moral foundation per response by model."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 4.8))
    order = list(FOUNDATION_LABELS.values())
    counts = (
        df.groupby(["model", "top_foundation"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"] / counts.groupby("model")["count"].transform("sum")
    ax = sns.barplot(
        data=counts,
        x="top_foundation",
        y="share",
        hue="model",
        order=order,
    )
    ax.set_title("Dominant moral foundation per response (share)")
    ax.set_xlabel("Moral foundation")
    ax.set_ylabel("Proportion of responses")
    ax.tick_params(axis="x", rotation=25)
    plt.legend(title="Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def verdict_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Return shares of each verdict per model and distractor type."""
    counts = (
        df.groupby(["model", "distractor_type", "verdict"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"] / counts.groupby(
        ["model", "distractor_type"]
    )["count"].transform("sum")
    return counts


def plot_verdict_shares_by_distractor(shares: pd.DataFrame, out_path: Path) -> None:
    """Plot verdict shares by distractor polarity faceted by model."""
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=shares,
        kind="bar",
        col="model",
        x="distractor_type",
        y="share",
        hue="verdict",
        palette=VERDICT_PALETTE,
        col_wrap=2,
        height=4,
        sharey=False,
        dodge=True,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Distractor polarity", "Proportion of responses")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Verdict distribution by distractor polarity and model")
    for ax in g.axes.flatten():
        ax.set_ylim(0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=300)
    plt.close(g.fig)


def plot_verdict_overall(shares: pd.DataFrame, out_path: Path) -> None:
    """Plot overall verdict mix per model."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6.5, 4))
    ax = sns.barplot(
        data=shares,
        x="model",
        y="share",
        hue="verdict",
        palette=VERDICT_PALETTE,
    )
    ax.set_title("Verdict distribution per model (overall)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Proportion of responses")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_foundation_by_distractor(df: pd.DataFrame, out_path: Path) -> None:
    """Plot average moral scores by distractor polarity per model."""
    melted = df.melt(
        id_vars=["model", "distractor_type"],
        value_vars=MORAL_COLS,
        var_name="foundation",
        value_name="score",
    )
    melted["foundation"] = melted["foundation"].map(FOUNDATION_LABELS)
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=melted,
        kind="bar",
        col="model",
        x="foundation",
        y="score",
        hue="distractor_type",
        palette=DISTRACTOR_PALETTE,
        col_wrap=2,
        height=4.2,
        sharey=False,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Moral foundation", "Average score")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Moral foundation scores by distractor polarity")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, 1.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=300)
    plt.close(g.fig)


def plot_delta_vs_baseline(shares: pd.DataFrame, out_path: Path) -> None:
    """Plot verdict share deltas relative to baseline for each model."""
    # Pivot to ensure missing verdicts get 0 share in baseline.
    all_verdicts = sorted(shares["verdict"].unique())
    base = (
        shares[shares["distractor_type"] == "baseline"]
        .pivot(index="model", columns="verdict", values="share")
        .reindex(columns=all_verdicts, fill_value=0)
        .reset_index()
    )
    base = base.melt(id_vars="model", var_name="verdict", value_name="share_base")

    merged = shares.merge(base, on=["model", "verdict"], how="left")
    merged = merged[merged["distractor_type"] != "baseline"].copy()
    merged["share_base"] = merged["share_base"].fillna(0)
    merged["delta"] = merged["share"] - merged["share_base"]

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=merged,
        kind="bar",
        col="model",
        x="verdict",
        y="delta",
        hue="distractor_type",
        palette=DISTRACTOR_PALETTE,
        col_wrap=2,
        height=4.2,
        sharey=False,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Verdict", "Δ share vs baseline")
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle("Change in verdict share vs baseline (positive = increase)")
    for ax in g.axes.flatten():
        ax.axhline(0, color="black", linewidth=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=300)
    plt.close(g.fig)


def plot_verdict_spider_by_model(shares: pd.DataFrame, out_dir: Path) -> None:
    """Plot verdict mix as a spider chart per model across distractor types."""
    verdicts = sorted(shares["verdict"].unique())
    num_vars = len(verdicts)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    out_dir.mkdir(parents=True, exist_ok=True)

    for model in shares["model"].unique():
        sub = shares[shares["model"] == model]
        pivot = sub.pivot_table(
            index="distractor_type",
            columns="verdict",
            values="share",
            fill_value=0,
        )
        plt.figure(figsize=(6.5, 6))
        ax = plt.subplot(111, polar=True)
        for dtype, row in pivot.iterrows():
            values = [row.get(v, 0) for v in verdicts]
            values += values[:1]
            color = DISTRACTOR_PALETTE.get(dtype, None)
            ax.plot(angles, values, label=dtype, linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(verdicts)
        ax.set_ylim(0, 1)
        ax.set_title(f"Verdict mix by distractor type: {model}", y=1.08)
        ax.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
        plt.tight_layout()
        fname = f"verdict_spider_{model.replace('/', '_')}.png"
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()


def plot_spider(
    averages: pd.DataFrame,
    out_path: Path,
    title: str = "Average moral foundation scores",
) -> None:
    """Plot spider/radar chart for average moral scores."""
    labels = list(FOUNDATION_LABELS.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    for model, row in averages.iterrows():
        values = row[MORAL_COLS].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.15)

    ax.set_title(title, y=1.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    max_val = float(averages[MORAL_COLS].max().max())
    ax.set_ylim(0, max(0.01, max_val * 1.05))
    ax.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def describe_esh_spike(counts: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table of ESH shares per distractor type for printing."""
    esh = counts[counts["verdict"] == "ESH"].copy()
    pivot = esh.pivot_table(
        index="model",
        columns="distractor_type",
        values="share",
        fill_value=0,
    )
    return pivot


def main() -> None:
    root = repo_root()
    data_dir = root / "src/analysis/reddit_csvs"
    fig_dir = root / "fig/reddit"

    df = load_reddit_frames(data_dir)
    df = add_top_foundation(df)

    counts = esh_share_by_distractor(df)
    esh_table = describe_esh_spike(counts)

    verdict_share_table = verdict_shares(df)

    overall_shares = (
        df.groupby(["model", "verdict"]).size().rename("count").reset_index()
    )
    overall_shares["share"] = (
        overall_shares["count"] / overall_shares.groupby("model")["count"].transform("sum")
    )

    foundation_bar_path = fig_dir / "moral_foundation_distribution.png"
    plot_foundation_bars(df, foundation_bar_path)

    esh_plot_path = fig_dir / "esh_by_distractor.png"
    plot_esh_spike(counts, esh_plot_path)

    averages = df.groupby("model")[MORAL_COLS].mean()
    spider_path = fig_dir / "moral_foundation_spider.png"
    plot_spider(averages, spider_path)

    verdict_shares_path = fig_dir / "verdict_by_distractor.png"
    plot_verdict_shares_by_distractor(verdict_share_table, verdict_shares_path)

    verdict_overall_path = fig_dir / "verdict_overall.png"
    plot_verdict_overall(overall_shares, verdict_overall_path)

    foundation_by_dist_path = fig_dir / "moral_foundation_by_distractor.png"
    plot_foundation_by_distractor(df, foundation_by_dist_path)

    delta_path = fig_dir / "verdict_delta_vs_baseline.png"
    plot_delta_vs_baseline(verdict_share_table, delta_path)

    verdict_spider_dir = fig_dir / "verdict_spiders"
    plot_verdict_spider_by_model(verdict_share_table, verdict_spider_dir)

    print("=== ESH share by distractor polarity ===")
    print(esh_table.round(3))
    print("\n=== Verdict shares by distractor polarity (head) ===")
    print(
        verdict_share_table[["model", "distractor_type", "verdict", "share"]]
        .sort_values(["model", "distractor_type", "verdict"])
        .head(15)
        .to_string(index=False)
    )
    print("\nPlots saved to:", fig_dir.resolve())


if __name__ == "__main__":
    main()
