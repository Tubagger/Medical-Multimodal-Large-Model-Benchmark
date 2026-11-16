#!/usr/bin/env python3
"""Plot clean accuracy and paired robust accuracy for R.2 and R.3."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "score_calculation/results/dimension_scores_current.json"
FIGURE_PDF = ROOT / "IEEE_ACCESS/figures/attack_degradation_17models.pdf"
FIGURE_PNG = ROOT / "access/supplementary/diagnostics/attack_degradation_17models.png"
MPL_DIR = Path("/tmp/medmmlm-matplotlib")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def load_total(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    total = data.get("total_results")
    if not isinstance(total, dict):
        raise ValueError(f"Missing total_results: {path}")
    return total


def clean_accuracy(task, model):
    stem = "adv-untarget" if task == "R2" else "adv-target"
    path = ROOT / "logs/robustness" / task / SLUGS[model] / f"{stem}-clean.scored.json"
    value = load_total(path).get("rule_mcq_eval:accuracy_score")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing clean accuracy: {path}")
    return 100.0 * value


def load_rows():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: row["overall"], reverse=True)
    for row in rows:
        if row["model"] not in SLUGS:
            raise ValueError(f"Missing slug for {row['model']}")
        row["R2_clean"] = clean_accuracy("R2", row["model"])
        row["R3_clean"] = clean_accuracy("R3", row["model"])
    return rows


def panel(ax, rows, positions, clean_key, score_key, title, color, show_legend=False):
    clean = np.array([row[clean_key] for row in rows])
    robust = np.array([row[score_key] for row in rows])
    for position, clean_value, robust_value in zip(positions, clean, robust):
        ax.plot(
            [robust_value, clean_value],
            [position, position],
            color="#b8b8b8",
            linewidth=1.3,
            zorder=1,
        )
    ax.scatter(
        clean,
        positions,
        s=25,
        facecolor="#d9edf7",
        edgecolor="#555555",
        linewidth=0.6,
        zorder=2,
        label="Clean accuracy",
    )
    ax.scatter(
        robust,
        positions,
        s=26,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label="Paired robust accuracy",
    )
    mean_gap = float(np.mean(clean - robust))
    ax.set_title(
        f"{title}\nmean drop = {mean_gap:.1f} points",
        fontsize=7.4,
        fontweight="bold",
        pad=5,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy / score (%)", fontsize=6.8)
    ax.grid(axis="x", linestyle="--", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.0, length=2.5)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    if show_legend:
        ax.legend(
            loc="lower left",
            fontsize=5.5,
            frameon=False,
            handletextpad=0.3,
            borderpad=0.2,
        )


def plot(rows):
    positions = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.55), sharey=True)
    panel(
        axes[0],
        rows,
        positions,
        "R2_clean",
        "R2",
        "(a) R.2 untargeted attack",
        "#E6862A",
        show_legend=True,
    )
    panel(
        axes[1],
        rows,
        positions,
        "R3_clean",
        "R3",
        "(b) R.3 targeted attack",
        "#C44E52",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([row["model"] for row in rows], fontsize=6.1)
    axes[0].set_ylabel("Models (ordered by overall score)", fontsize=6.8)
    fig.tight_layout(pad=0.4, w_pad=0.8)

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURE_PDF, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {FIGURE_PNG}")
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    plot(load_rows())
