#!/usr/bin/env python3
"""Plot paired-task diagnostics for the five benchmark dimensions."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'score_calculation/results/model_task_scores_16.json'
OUT_DIR = ROOT / 'access/supplementary/diagnostics'
MPL_DIR = Path('/tmp/medmmlm-matplotlib')
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr, spearmanr

NAMES = {
    'lingshu-7b': 'Lingshu-7B',
    'lingshu-32b': 'Lingshu-32B',
    'med-gemma-4b-it': 'MedGemma-4B',
    'med-gemma-27b-it': 'MedGemma-27B',
    'hulu-med-7b': 'Hulu-Med-7B',
    'hulu-med-32b': 'Hulu-Med-32B',
    'huatuogpt-v-7b': 'HuatuoGPT-V-7B',
    'huatuogpt-v-34b': 'HuatuoGPT-V-34B',
    'healthgpt-pro-8b': 'HealthGPT-Pro-8B',
    'qwen3-vl-32b-instruct': 'Qwen3-VL-32B',
    'deepseek-vl2': 'DeepSeek-VL2',
    'deepseek-vl2-small': 'DeepSeek-VL2-Small',
    'internvl3.5-14b': 'InternVL3.5-14B',
    'internvl3.5-38b': 'InternVL3.5-38B',
    'glm-4v-9b': 'GLM-4V-9B',
    'gpt-5.4': 'GPT-5.4',
    'gemini-3.5-flash': 'Gemini-3.5-Flash',
}

# (task A, task B, dimension, label A, label B, color)
PANELS = [
    ('C1', 'C2', 'Capability', 'C.1 Recognition', 'C.2 Localization', '#4E79A7'),
    ('R2', 'R3', 'Robustness', 'R.2 Untargeted', 'R.3 Targeted', '#F28E2B'),
    ('F1', 'F2', 'Fairness', 'F.1 Neutrality', 'F.2 Stability', '#59A14F'),
    ('P1', 'P4', 'Privacy', 'P.1 Recognition', 'P.4 Protection', '#76B7B2'),
    ('S1', 'S3', 'Safety', 'S.1 Direct refusal', 'S.3 After rewrite', '#E15759'),
]


def main():
    rows = json.loads(RESULT.read_text(encoding='utf-8'))
    rows = [row for row in rows if row['model'] in NAMES]
    rows.sort(key=lambda row: row['overall'], reverse=True)
    labels = [NAMES[row['model']] for row in rows]
    positions = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 5, figsize=(23, 10), sharey=True)
    summary = []
    for ax, (key_a, key_b, title, label_a, label_b, color) in zip(axes, PANELS):
        values_a = np.array([row[key_a] for row in rows])
        values_b = np.array([row[key_b] for row in rows])
        for position, value_a, value_b in zip(positions, values_a, values_b):
            ax.plot([value_a, value_b], [position, position],
                    color='#B8B8B8', linewidth=1.1, zorder=1)
        ax.scatter(values_a, positions, s=46, color=color, edgecolor='white',
                   linewidth=0.6, zorder=3)
        ax.scatter(values_b, positions, s=46, color='#4D4D4D', edgecolor='white',
                   linewidth=0.6, zorder=4)
        # Emphasize the largest degradations without crowding every row.
        gaps = values_a - values_b
        for index in np.argsort(gaps)[-3:]:
            ax.annotate(f'{gaps[index]:+.1f}', (values_b[index], positions[index]),
                        xytext=(6, 0), textcoords='offset points', va='center',
                        fontsize=6.5, color='#333333')
        lo = min(values_a.min(), values_b.min())
        hi = max(values_a.max(), values_b.max())
        pad = max(3.0, (hi - lo) * 0.14)
        ax.set_xlim(max(-2, lo - pad), min(102, hi + pad))
        r = pearsonr(values_a, values_b)[0]
        rho = spearmanr(values_a, values_b)[0]
        gap = float(np.mean(np.abs(gaps)))
        ax.set_title(f'{title}\nPearson r={r:.2f}, mean |gap|={gap:.1f}',
                     fontsize=12.5, fontweight='bold')
        ax.set_xlabel('Score (%)', fontsize=9.5, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.legend(
            handles=[
                Line2D([0], [0], marker='o', linestyle='', markersize=6,
                       markerfacecolor=color, markeredgecolor='white', label=label_a),
                Line2D([0], [0], marker='o', linestyle='', markersize=6,
                       markerfacecolor='#4D4D4D', markeredgecolor='white', label=label_b),
            ],
            loc='lower right', fontsize=7.2, frameon=False, handletextpad=0.3,
        )
        summary.append((title, key_a, key_b, r, rho, gap))

    axes[0].set_yticks(positions, labels=labels, fontsize=8.5)
    axes[0].set_ylabel('Models (ordered by overall score)', fontsize=10, fontweight='bold')
    fig.suptitle('Paired-task consistency within each trustworthiness dimension',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / 'dimension_diagnostics_17models.png'
    pdf = OUT_DIR / 'dimension_diagnostics_17models.pdf'
    fig.savefig(png, dpi=400, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    csv_path = ROOT / 'score_calculation/results/dimension_diagnostic_pairs.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dimension', 'task_a', 'task_b', 'pearson_r', 'spearman_rho', 'mean_abs_gap'])
        for title, key_a, key_b, r, rho, gap in summary:
            writer.writerow([title, key_a, key_b, round(float(r), 4), round(float(rho), 4), round(gap, 4)])
    print('Saved', png)
    print('Saved', pdf)
    print('Saved', csv_path)


if __name__ == '__main__':
    main()
