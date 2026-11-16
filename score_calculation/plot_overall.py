#!/usr/bin/env python3
"""Plot the five-dimension benchmark profile in the original overall style."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/overall_16task_17models.pdf"
PREVIEW_PNG = (
    ROOT / "access/supplementary/diagnostics/overall_16task_17models.png"
)
MPL_DIR = Path("/tmp/medmmlm-matplotlib")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = [
    ("Capability", "Capability", "#4E79A7"),
    ("Robustness", "Robustness", "#F28E2B"),
    ("Fairness", "Fairness", "#59A14F"),
    ("Privacy", "Privacy", "#76B7B2"),
    ("Safety", "Safety", "#E15759"),
    ("Overall", "overall", "#333333"),
]


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    required = {"model", "overall", *(key for _, key, _ in SERIES)}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('model', '<unknown>')}: missing {missing}")
    return rows


def plot(rows):
    sorted_rows = sorted(rows, key=lambda row: row["overall"], reverse=True)
    models = [row["model"] for row in sorted_rows]
    x_positions = list(range(len(sorted_rows)))

    group_width = 0.84
    bar_width = group_width / len(SERIES)
    center_offset = (len(SERIES) - 1) / 2.0

    fig, ax = plt.subplots(figsize=(22, 8))

    for series_index, (label, score_key, color) in enumerate(SERIES):
        offset = (series_index - center_offset) * bar_width
        bar_positions = [position + offset for position in x_positions]
        values = [row[score_key] for row in sorted_rows]
        bars = ax.bar(
            bar_positions,
            values,
            width=bar_width * 0.88,
            color=color,
            label=label,
            edgecolor="#111111" if score_key == "overall" else "none",
            linewidth=0.6 if score_key == "overall" else 0.0,
        )
        if score_key == "overall":
            ax.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=2,
                fontsize=7.5,
                fontweight="bold",
                rotation=90,
            )

    ax.set_xticks(
        x_positions,
        labels=models,
        rotation=45,
        ha="right",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Models", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=15, fontweight="bold")
    ax.set_title("", fontsize=15)
    ax.set_ylim(0, 100)

    actual_bar_width = bar_width * 0.88
    half_group_span = center_offset * bar_width + actual_bar_width / 2.0
    inter_model_gap = 1.0 - 2.0 * half_group_span
    side_extent = half_group_span + inter_model_gap
    ax.set_xlim(
        -side_extent,
        len(sorted_rows) - 1 + side_extent,
    )
    ax.margins(x=0)

    ax.tick_params(axis="y", labelsize=10)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        prop={"size": 15, "weight": "bold"},
        borderaxespad=0.3,
    )

    fig.tight_layout()
    position = ax.get_position()
    ax.set_position(
        [
            position.x0,
            position.y0 + 0.06,
            position.width,
            position.height * 0.82,
        ]
    )

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PNG, dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        FIGURE_PDF,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"Saved {PREVIEW_PNG}")
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    plot(load_rows())
