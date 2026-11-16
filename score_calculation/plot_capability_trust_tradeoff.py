#!/usr/bin/env python3
"""Plot capability against the mean of the four trust-oriented dimensions."""
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/capability_trust_tradeoff_17models.pdf"
FIGURE_PNG = (
    ROOT / "access/supplementary/diagnostics/capability_trust_tradeoff_17models.png"
)
TABLE_OUT = ROOT / "score_calculation/results/capability_trust_tradeoff.csv"
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
TRUST_DIMENSIONS = ["Robustness", "Fairness", "Privacy", "Safety"]
LABEL_OFFSETS = {
    "GPT-5.4": (5, 3),
    "Gemini-3.5-Flash": (-8, -11),
    "Hulu-Med-32B": (5, -3),
    "MedGemma-4B": (5, 4),
    "Lingshu-32B": (5, 2),
    "DeepSeek-VL2": (5, 3),
    "HuatuoGPT-V-7B": (5, -2),
}


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    required = {"model", "group", "Capability", *TRUST_DIMENSIONS}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('model', '<unknown>')}: missing {missing}")
        row["trust_composite"] = float(np.mean([row[d] for d in TRUST_DIMENSIONS]))
    return sorted(rows, key=lambda row: row["overall"], reverse=True)


def write_table(rows):
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "group",
                "capability",
                "trust_composite",
                "overall",
                "capability_rank",
                "trust_rank",
                "overall_rank",
            ],
        )
        writer.writeheader()
        capability_rank = {
            row["model"]: rank
            for rank, row in enumerate(
                sorted(rows, key=lambda row: row["Capability"], reverse=True), 1
            )
        }
        trust_rank = {
            row["model"]: rank
            for rank, row in enumerate(
                sorted(rows, key=lambda row: row["trust_composite"], reverse=True), 1
            )
        }
        overall_rank = {row["model"]: rank for rank, row in enumerate(rows, 1)}
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "group": row["group"],
                    "capability": f"{row['Capability']:.2f}",
                    "trust_composite": f"{row['trust_composite']:.2f}",
                    "overall": f"{row['overall']:.2f}",
                    "capability_rank": capability_rank[row["model"]],
                    "trust_rank": trust_rank[row["model"]],
                    "overall_rank": overall_rank[row["model"]],
                }
            )


def plot(rows):
    x = np.array([row["Capability"] for row in rows])
    y = np.array([row["trust_composite"] for row in rows])
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    correlation = float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.axvline(x_mean, color="#9a9a9a", linewidth=0.7, linestyle="--", zorder=0)
    ax.axhline(y_mean, color="#9a9a9a", linewidth=0.7, linestyle="--", zorder=0)

    for group, color, marker in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        ax.scatter(
            [row["Capability"] for row in group_rows],
            [row["trust_composite"] for row in group_rows],
            s=32,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            label=group,
            zorder=3,
        )

    for row in rows:
        if row["model"] not in LABEL_OFFSETS:
            continue
        dx, dy = LABEL_OFFSETS[row["model"]]
        ax.annotate(
            row["model"],
            (row["Capability"], row["trust_composite"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.2,
            color="#303030",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
        )

    ax.text(
        46.7,
        35.5,
        f"Pearson r = {correlation:.2f}",
        fontsize=6.1,
        color="#555555",
    )
    ax.set_xlim(20.5, 52.2)
    ax.set_ylim(34.0, 70.2)
    ax.set_xlabel("Capability score (%)", fontsize=6.8)
    ax.set_ylabel("Mean of R, F, P, and S scores (%)", fontsize=6.8)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.2, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.legend(
        loc="upper left",
        fontsize=5.7,
        frameon=False,
        handletextpad=0.3,
        borderpad=0.2,
    )
    fig.tight_layout(pad=0.4)

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURE_PDF, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {FIGURE_PNG}")
    print(f"Saved {FIGURE_PDF}")
    print(f"Saved {TABLE_OUT}")


if __name__ == "__main__":
    rows = load_rows()
    write_table(rows)
    plot(rows)
