import os

import matplotlib.pyplot as plt
import numpy as np


def plot_ece_comparison(dataset_name, methods, orig_eces, clf_eces, save_dir, clf_label="Truth-Anchored", suffix=""):
    x = np.arange(len(methods))
    width = 0.7
    fig, ax = plt.subplots(figsize=(10.5, 8))
    bars_orig = ax.bar(x, orig_eces, width, label="Vanilla", color="skyblue", hatch="/", edgecolor="black", alpha=0.6, zorder=3)
    bars_clf = ax.bar(x, clf_eces, width, label=clf_label, color="dodgerblue", alpha=0.8, hatch="/", edgecolor="black", zorder=2)
    for bars in (bars_orig, bars_clf):
        for bar in bars:
            bar.set_linewidth(3)
    title = {"trivia": "TriviaQA", "sciq": "SciQ", "popqa": "PopQA"}.get(dataset_name, dataset_name)
    fig.suptitle(title, fontsize=28)
    ax.set_ylabel("ECE", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=26)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(fontsize=26, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=2)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_ece_comparison{suffix}.png"), dpi=400, bbox_inches="tight")
    plt.close()


def plot_auc_comparison(dataset_name, methods, orig_aucs, clf_aucs, save_dir, clf_label="Truth-Anchored", suffix=""):
    x = np.arange(len(methods))
    width = 0.7
    fig, ax = plt.subplots(figsize=(10.5, 8))
    bars_orig = ax.bar(x, orig_aucs, width, label="Vanilla", color="lightgreen", hatch="/", edgecolor="black", alpha=0.6, zorder=3)
    bars_clf = ax.bar(x, clf_aucs, width, label=clf_label, color="seagreen", alpha=0.8, hatch="/", edgecolor="black", zorder=2)
    for bars in (bars_orig, bars_clf):
        for bar in bars:
            bar.set_linewidth(3)
    title = {"trivia": "TriviaQA", "sciq": "SciQ", "popqa": "PopQA"}.get(dataset_name, dataset_name)
    fig.suptitle(title, fontsize=28)
    ax.set_ylabel("AUC", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=26)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(fontsize=26, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=2)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_auc_comparison{suffix}.png"), dpi=400, bbox_inches="tight")
    plt.close()


def plot_calibration_diagram(scores, labels, auroc, ece, title, save_dir, anchored=0, n_bins=10, bar_frac=0.9, suffix=""):
    if np.any(scores < 0) or np.any(scores > 1):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bar_width = bin_width * bar_frac
    bar_left = bin_edges[:-1] + (bin_width - bar_width) / 2
    bin_ids = np.digitize(scores, bin_edges[1:-1], right=False)
    bin_acc = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_acc[b] = labels[mask].mean()
    colors = [plt.get_cmap("Blues")(v) for v in np.linspace(0.25, 0.85, n_bins)]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.8, zorder=0)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.2, alpha=0.9, zorder=1)
    ax.bar(bar_left, bin_acc, width=bar_width, align="edge", color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, pad=2)
    ax.text(
        0.03,
        0.97,
        f"AUC: {auroc*100:.2f}\nECE: {ece*100:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8, boxstyle="round,pad=0.25", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"calibration_{title}{anchored}{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_ece_comparison_3way(dataset_name, methods, orig_eces, mlp_eces, cue_eces, save_dir):
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))
    bars_orig = ax.bar(x - width, orig_eces, width, label="Vanilla", color="skyblue", hatch="/", edgecolor="black", alpha=0.6, zorder=3)
    bars_cue = ax.bar(x, cue_eces, width, label="CUE", color="dodgerblue", alpha=0.9, hatch="\\", edgecolor="black", zorder=2)
    bars_mlp = ax.bar(x + width, mlp_eces, width, label="Truth-Anchored", color="navy", alpha=0.8, hatch="/", edgecolor="black", zorder=2)
    for bars in (bars_orig, bars_cue, bars_mlp):
        for bar in bars:
            bar.set_linewidth(2)
    title = {"trivia": "TriviaQA", "sciq": "SciQ", "popqa": "PopQA"}.get(dataset_name, dataset_name.capitalize())
    fig.suptitle(title, fontsize=28)
    ax.set_ylabel("ECE", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(fontsize=22, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_ece_comparison_3way.png"), dpi=400, bbox_inches="tight")
    plt.close()


def plot_auc_comparison_3way(dataset_name, methods, orig_aucs, mlp_aucs, cue_aucs, save_dir):
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))
    bars_orig = ax.bar(x - width, orig_aucs, width, label="Vanilla", color="lightgreen", hatch="/", edgecolor="black", alpha=0.6, zorder=3)
    bars_cue = ax.bar(x, cue_aucs, width, label="CUE", color="seagreen", alpha=0.9, hatch="\\", edgecolor="black", zorder=2)
    bars_mlp = ax.bar(x + width, mlp_aucs, width, label="Truth-Anchored", color="darkgreen", alpha=0.8, hatch="/", edgecolor="black", zorder=2)
    for bars in (bars_orig, bars_cue, bars_mlp):
        for bar in bars:
            bar.set_linewidth(2)
    title = {"trivia": "TriviaQA", "sciq": "SciQ", "popqa": "PopQA"}.get(dataset_name, dataset_name.capitalize())
    fig.suptitle(title, fontsize=28)
    ax.set_ylabel("AUC", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(fontsize=22, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_auc_comparison_3way.png"), dpi=400, bbox_inches="tight")
    plt.close()
