#!/usr/bin/env python3
"""Plot the current 17-model by 16-task score matrix."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/task_score_heatmap_17models.pdf"
FIGURE_PNG = (
    ROOT / "access/supplementary/diagnostics/task_score_heatmap_17models.png"
)
MPL_DIR = Path("/tmp/medmmlm-matplotlib")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = [
    ("Capability", ["C1", "C2", "C3"]),
    ("Robustness", ["R1", "R2", "R3", "R4"]),
    ("Fairness", ["F1", "F2"]),
    ("Privacy", ["P1", "P2", "P3", "P4"]),
    ("Safety", ["S1", "S2", "S3"]),
]


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    tasks = [task for _, group in GROUPS for task in group]
    for row in rows:
        missing = [task for task in tasks if task not in row]
        if missing:
            raise ValueError(f"{row.get('model', '<unknown>')}: missing {missing}")
    return sorted(rows, key=lambda row: row["overall"], reverse=True), tasks


def plot(rows, tasks):
    models = [row["model"] for row in rows]
    values = np.array([[row[task] for task in tasks] for row in rows])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(
        [f"{task[0]}.{task[1]}" for task in tasks],
        fontsize=6.6,
        fontweight="bold",
    )
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    for row_index in range(len(models)):
        for column_index in range(len(tasks)):
            value = values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=4.8,
                color="white" if value >= 55 else "#1a1a1a",
            )

    offset = 0
    for _, group in GROUPS:
        if offset:
            ax.axvline(offset - 0.5, color="white", linewidth=1.6)
        offset += len(group)

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.012)
    colorbar.set_label("Score (%)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURE_PDF, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {FIGURE_PNG}")
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    plot(*load_rows())
