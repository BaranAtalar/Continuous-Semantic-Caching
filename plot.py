"""Draw the paper's six-panel results figure (Figure 2) from the files written by run.py.

    python plot.py [--results results]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"CUCB-SC-Cont": "#1f77b4", "CLCB-SC-LS-Cont": "#1f77b4", "CUCB-SC": "#8c564b", "CLCB-SC-LS": "#8c564b",
          "Greedy": "#9467bd", "Epsilon-Greedy": "#ff7f0e", "LFU": "#d62728"}
MARKERS = {"CUCB-SC-Cont": "o", "CLCB-SC-LS-Cont": "o", "CUCB-SC": "D", "CLCB-SC-LS": "D",
           "Greedy": "v", "Epsilon-Greedy": "s", "LFU": "^"}
LEGEND = ["CUCB-SC", "CUCB-SC-Cont", "Epsilon-Greedy", "Greedy", "LFU", "CLCB-SC-LS-Cont", "CLCB-SC-LS"]


def offline_panel(ax, x, gaps, xlabel, title):
    for name, values in gaps.items():
        values = np.asarray(values)
        mean, std = values.mean(axis=1), values.std(axis=1, ddof=1)
        ax.plot(x, mean, marker=MARKERS[name], linewidth=2.2, markersize=7, label=name, color=COLORS[name])
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std, alpha=0.2, color=COLORS[name])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Suboptimality Gap")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")


def regret_panel(ax, res, title, ylim):
    for name, curves in res["average_regret"].items():
        curves = np.asarray(curves)
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, linewidth=2.2, label=name, color=COLORS[name])
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.2, color=COLORS[name])
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Average Regret")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linestyle="--")


def runtime_panel(ax, res, title):
    names = list(res["average_regret"])
    final = [np.asarray(res["average_regret"][name])[:, -1] for name in names]
    runtime = [np.asarray(res["runtime"][name]) for name in names]
    x_span = np.ptp([r.mean() for r in runtime]) or 1.0
    y_span = np.ptp([f.mean() for f in final]) or 1.0
    for name, f, r in zip(names, final, runtime):
        ax.errorbar(r.mean(), f.mean(), xerr=r.std(), yerr=f.std(), fmt=MARKERS[name], markersize=9, capsize=5,
                    color=COLORS[name], alpha=0.85, elinewidth=1.5)
        ax.text(r.mean() + 0.04 * x_span, f.mean() + 0.02 * y_span, name, fontsize=9)
    ax.set_xlabel("Runtime (sec)")
    ax.set_ylabel("Final Average Regret")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")


def make_figure(results_dir):
    results_dir = Path(results_dir)
    res = {name: json.loads((results_dir / f"{name}.json").read_text())
           for name in ("stream_length", "epsilon", "online_synthetic", "online_real")}
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    offline_panel(axes[0, 0], res["stream_length"]["lengths"], res["stream_length"]["gaps"],
                  "Stream Length", "(a) Suboptimality Gap vs Stream Length")
    offline_panel(axes[1, 0], res["epsilon"]["epsilons"], res["epsilon"]["gaps"],
                  r"$\epsilon$", r"(b) Suboptimality Gap vs $\epsilon$")
    regret_panel(axes[0, 1], res["online_synthetic"], "(c) Synthetic Online: Average Regret vs Rounds", (0, 0.25))
    runtime_panel(axes[1, 1], res["online_synthetic"], "(d) Synthetic Online: Runtime vs Final Avg Regret")
    regret_panel(axes[0, 2], res["online_real"], "(e) Combined NQ + TriviaQA: Average Regret vs Rounds", (0, 0.35))
    runtime_panel(axes[1, 2], res["online_real"], "(f) Combined NQ + TriviaQA: Runtime vs Final Avg Regret")

    handles = {}
    for ax in (axes[0, 0], axes[1, 0], axes[0, 1], axes[0, 2]):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    labels = [label for label in LEGEND if label in handles]
    fig.legend([handles[label] for label in labels], labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=4, frameon=True, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = results_dir / "figure.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    print(f"Saved {make_figure(parser.parse_args().results)}")
