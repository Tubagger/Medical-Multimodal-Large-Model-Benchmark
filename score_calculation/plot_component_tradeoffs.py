#!/usr/bin/env python3
"""Plot violation rate against response appropriateness for judged tasks."""
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/component_tradeoffs_17models.pdf"
FIGURE_PNG = ROOT / "access/supplementary/diagnostics/component_tradeoffs_17models.png"
TABLE_OUT = ROOT / "score_calculation/results/component_metrics_current.csv"
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
SLUGS = {
    "GPT-5.4": "gpt-5.4",
    "Gemini-3.5-Flash": "gemini-3.5-flash",
    "InternVL3.5-38B": "internvl3.5-38b",
    "Lingshu-7B": "lingshu-7b",
    "Lingshu-32B": "lingshu-32b",
    "MedGemma-4B": "med-gemma-4b-it",
    "Qwen3-VL-32B": "qwen3-vl-32b-instruct",
    "HealthGPT-Pro-8B": "healthgpt-pro-8b",
    "GLM-4V-9B": "glm-4v-9b",
    "Hulu-Med-32B": "hulu-med-32b",
    "InternVL3.5-14B": "internvl3.5-14b",
    "DeepSeek-VL2": "deepseek-vl2",
    "Hulu-Med-7B": "hulu-med-7b",
    "MedGemma-27B": "med-gemma-27b-it",
    "HuatuoGPT-V-34B": "huatuogpt-v-34b",
    "DeepSeek-VL2-Small": "deepseek-vl2-small",
    "HuatuoGPT-V-7B": "huatuogpt-v-7b",
}
PANELS = [
    ("P3", "privacy/P3", "PPR", "(a) P.3 privacy inference", [70, 80, 90]),
    ("P4", "privacy/P4", "PPR", "(b) P.4 privacy-modified VQA", [30, 40, 50, 60]),
    ("S1", "safety/S1", "HCR", "(c) S.1 risk request", [30, 50, 70, 90]),
    ("S2", "safety/S2", "HCR", "(d) S.2 MCM jailbreak", [50, 70, 90]),
    ("S3", "safety/S3", "HCR", "(e) S.3 safety-modified VQA", [30, 50, 70]),
]
LABEL_MODELS = {
    "P3": set(),
    "P4": set(),
    "S1": {"GPT-5.4", "Gemini-3.5-Flash"},
    "S2": set(),
    "S3": {"GPT-5.4", "Gemini-3.5-Flash"},
}


def load_total(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    total = data.get("total_results")
    if not isinstance(total, dict):
        raise ValueError(f"Missing total_results: {path}")
    return total


def weighted_components(folder, model):
    base = ROOT / "logs" / folder / SLUGS[model]
    numerators = {"PPR": 0.0, "HCR": 0.0, "RAS": 0.0}
    denominators = {"PPR": 0, "HCR": 0, "RAS": 0}
    for path in sorted(base.glob("*.scored.json")):
        values = load_total(path).get("metrics_0_to_100") or {}
        for key in numerators:
            value = values.get(key)
            count = values.get(f"{key}_n")
            if isinstance(value, (int, float)) and isinstance(count, int) and count:
                numerators[key] += float(value) * count
                denominators[key] += count
    if not denominators["RAS"]:
        raise ValueError(f"Missing RAS values: {base}")
    return {
        key: numerators[key] / denominators[key] if denominators[key] else None
        for key in numerators
    }


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    detailed = []
    for row in rows:
        for task, folder, criterion, _, _ in PANELS:
            values = weighted_components(folder, row["model"])
            violation = values[criterion]
            if violation is None:
                raise ValueError(f"Missing {criterion}: {row['model']} {task}")
            detailed.append(
                {
                    "model": row["model"],
                    "group": row["group"],
                    "task": task,
                    "criterion": criterion,
                    "criterion_value": violation,
                    "ras": values["RAS"],
                }
            )
    return detailed


def write_table(rows):
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "group", "task", "criterion", "criterion_value", "ras"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "group": row["group"],
                    "task": row["task"],
                    "criterion": row["criterion"],
                    "criterion_value": f"{row['criterion_value']:.4f}",
                    "ras": f"{row['ras']:.4f}",
                }
            )


def composite(x, y, task):
    if task.startswith("P"):
        return 0.5 * (x + y)
    return 0.5 * ((100.0 - x) + y)


def draw_isoline(ax, task, level, limits):
    xmin, xmax, ymin, ymax = limits
    if task.startswith("P"):
        x1, x2 = xmin, xmax
        y1, y2 = 2 * level - x1, 2 * level - x2
    else:
        x1, x2 = xmin, xmax
        y1, y2 = 2 * level - 100 + x1, 2 * level - 100 + x2
    ax.plot([x1, x2], [y1, y2], color="#9b9b9b", linewidth=0.55, linestyle=":", zorder=1)
    def line_y(value):
        if task.startswith("P"):
            return 2 * level - value
        return 2 * level - 100 + value

    left = xmin + 0.04 * (xmax - xmin)
    right = xmax - 0.04 * (xmax - xmin)
    candidates = [
        (left, line_y(left)),
        (right, line_y(right)),
    ]
    for tx, ty in candidates:
        if ymin + 0.03 * (ymax - ymin) <= ty <= ymax - 0.03 * (ymax - ymin):
            ax.text(
                tx,
                ty + 0.015 * (ymax - ymin),
                f"C={level}",
                fontsize=4.8,
                color="#777777",
                ha="left",
                va="bottom",
            )
            break


def plot(rows):
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.95))
    axes = axes.flatten()

    for ax, (task, _, criterion, title, levels) in zip(axes, PANELS):
        task_rows = [row for row in rows if row["task"] == task]
        xs = np.array([row["criterion_value"] for row in task_rows])
        ys = np.array([row["ras"] for row in task_rows])
        x_pad = max(2.5, 0.08 * (xs.max() - xs.min()))
        y_pad = max(3.0, 0.10 * (ys.max() - ys.min()))
        limits = (
            max(0.0, xs.min() - x_pad),
            min(100.0, xs.max() + x_pad),
            max(0.0, ys.min() - y_pad),
            min(100.0, ys.max() + y_pad),
        )

        for level in levels:
            draw_isoline(ax, task, level, limits)

        for group, color, marker in GROUPS:
            group_rows = [row for row in task_rows if row["group"] == group]
            ax.scatter(
                [row["criterion_value"] for row in group_rows],
                [row["ras"] for row in group_rows],
                s=20,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

        for row in task_rows:
            if row["model"] not in LABEL_MODELS[task]:
                continue
            high_x = row["criterion_value"] > limits[0] + 0.82 * (limits[1] - limits[0])
            high_y = row["ras"] > limits[2] + 0.82 * (limits[3] - limits[2])
            dx = -5 if high_x else 5
            dy = -9 if high_y else 3
            ax.annotate(
                row["model"],
                (row["criterion_value"], row["ras"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=4.8,
                color="#303030",
                ha="right" if high_x else "left",
                va="bottom" if dy >= 0 else "top",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.78,
                ),
            )

        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_title(title, fontsize=6.7, fontweight="bold", pad=3)
        direction = "lower" if criterion == "HCR" else "higher"
        ax.set_xlabel(f"{criterion} (%) {direction} is better", fontsize=5.8)
        if ax in axes[::3]:
            ax.set_ylabel("RAS (%) higher is better", fontsize=5.8)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.28)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=5.4, length=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    axes[-1].axis("off")
    handles = [
        plt.Line2D(
            [],
            [],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=4.5,
            label=group,
        )
        for group, color, marker in GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=5.8,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1), pad=0.35, w_pad=0.7, h_pad=0.8)

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
