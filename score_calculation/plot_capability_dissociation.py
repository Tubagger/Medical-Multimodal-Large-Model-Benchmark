#!/usr/bin/env python3
"""Plot C.1 recognition against C.2 localization and C.3 reasoning."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/capability_dissociation_17models.pdf"
FIGURE_PNG = (
    ROOT / "access/supplementary/diagnostics/capability_dissociation_17models.png"
)
MPL_DIR = Path("/tmp/medmmlm-matplotlib")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = [
    ("Medical", "#4E79A7", "o"),
    ("General", "#59A14F", "s"),
    ("Proprietary", "#E15759", "D"),
]
LABELS_LEFT = {"Gemini-3.5-Flash", "Hulu-Med-32B", "Qwen3-VL-32B", "DeepSeek-VL2"}
LABELS_RIGHT = {"Gemini-3.5-Flash", "InternVL3.5-38B", "DeepSeek-VL2", "HuatuoGPT-V-7B"}


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    required = {"model", "group", "C1", "C2", "C3"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('model', '<unknown>')}: missing {missing}")
    return rows


def panel(ax, rows, x_key, y_key, xlabel, ylabel, title, labelled, limits):
    for group, color, marker in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        ax.scatter(
            [row[x_key] for row in group_rows],
            [row[y_key] for row in group_rows],
            s=26,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=group,
            zorder=3,
        )

    x = np.array([row[x_key] for row in rows])
    y = np.array([row[y_key] for row in rows])
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(line_x, slope * line_x + intercept, "--", color="#777777", linewidth=0.8, zorder=1)

    for row in rows:
        if row["model"] in labelled:
            ax.annotate(
                row["model"],
                (row[x_key], row[y_key]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.0,
                color="#222222",
            )

    r = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.03,
        0.96,
        f"Pearson r = {r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel(xlabel, fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=5)
    ax.grid(linestyle="--", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.0, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    panel(
        axes[0],
        rows,
        "C1",
        "C2",
        "C.1 recognition accuracy (%)",
        "C.2 localization score (%)",
        "(a) Recognition vs. localization",
        LABELS_LEFT,
        ((20, 72), (0, 24)),
    )
    panel(
        axes[1],
        rows,
        "C1",
        "C3",
        "C.1 recognition accuracy (%)",
        "C.3 reasoning accuracy (%)",
        "(b) Recognition vs. reasoning",
        LABELS_RIGHT,
        ((20, 72), (28, 68)),
    )
    axes[1].set_ylabel("C.3 reasoning accuracy (%)")
    axes[1].legend(
        loc="lower right",
        fontsize=5.6,
        frameon=False,
        handletextpad=0.3,
        borderpad=0.2,
    )
    fig.tight_layout(pad=0.4, w_pad=1.2)

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURE_PDF, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {FIGURE_PNG}")
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    plot(load_rows())
