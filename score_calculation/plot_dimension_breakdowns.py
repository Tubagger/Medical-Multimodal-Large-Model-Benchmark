#!/usr/bin/env python3
"""Per-dimension diagnostic figures from saved logs and 16-task scores."""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'access/supplementary/diagnostics'
MPL_DIR = Path('/tmp/medmmlm-matplotlib')
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    'lingshu-7b': 'Lingshu-7B', 'lingshu-32b': 'Lingshu-32B',
    'med-gemma-4b-it': 'MedGemma-4B', 'med-gemma-27b-it': 'MedGemma-27B',
    'hulu-med-7b': 'Hulu-Med-7B', 'hulu-med-32b': 'Hulu-Med-32B',
    'huatuogpt-v-7b': 'HuatuoGPT-V-7B', 'huatuogpt-v-34b': 'HuatuoGPT-V-34B',
    'healthgpt-pro-8b': 'HealthGPT-Pro-8B', 'qwen3-vl-32b-instruct': 'Qwen3-VL-32B',
    'deepseek-vl2': 'DeepSeek-VL2', 'deepseek-vl2-small': 'DeepSeek-VL2-Small',
    'internvl3.5-14b': 'InternVL3.5-14B', 'internvl3.5-38b': 'InternVL3.5-38B',
    'glm-4v-9b': 'GLM-4V-9B', 'gpt-5.4': 'GPT-5.4', 'gemini-3.5-flash': 'Gemini-3.5-Flash',
}

DIMS = {
    'capability': ('Capability', ['C1', 'C2', 'C3'], '#4E79A7'),
    'robustness': ('Robustness', ['R1', 'R2', 'R3', 'R4'], '#F28E2B'),
    'fairness': ('Fairness', ['F1', 'F2'], '#59A14F'),
    'privacy': ('Privacy', ['P1', 'P2', 'P3', 'P4'], '#76B7B2'),
    'safety': ('Safety', ['S1', 'S2', 'S3'], '#E15759'),
}

MED = ['lingshu-7b', 'lingshu-32b', 'med-gemma-4b-it', 'med-gemma-27b-it', 'hulu-med-7b',
       'hulu-med-32b', 'huatuogpt-v-7b', 'huatuogpt-v-34b', 'healthgpt-pro-8b']
GEN = ['qwen3-vl-32b-instruct', 'deepseek-vl2-small', 'deepseek-vl2', 'internvl3.5-14b',
       'internvl3.5-38b', 'glm-4v-9b']
API = ['gpt-5.4', 'gemini-3.5-flash']


def load_rows():
    return json.loads((ROOT / 'score_calculation/results/model_task_scores_16.json').read_text())


def load_total(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return data.get('total_results', {}) or {}


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return float(np.mean(values)) if values else None


def style(ax):
    ax.grid(axis='y', linestyle='--', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def grouped_heatmap(ax, models, columns, values, title, cmap, vmin=0, vmax=100, fmt='{:.1f}'):
    data = np.array([[values[m][c] for c in columns] for m in models], dtype=float)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(columns)), labels=columns, fontsize=8)
    ax.set_yticks(range(len(models)), labels=[NAMES[m] for m in models], fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt.format(data[i, j]), ha='center', va='center', fontsize=5.8,
                    color='white' if data[i, j] > (vmin + vmax) * 0.62 else '#222222')
    ax.set_title(title, fontsize=12, fontweight='bold')
    for x in np.arange(-0.5, len(columns), 1):
        ax.axvline(x, color='white', linewidth=1.5)
    for y in np.arange(-0.5, len(models), 1):
        ax.axhline(y, color='white', linewidth=1.5)
    return im


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f'{name}.png'
    pdf = OUT_DIR / f'{name}.pdf'
    fig.savefig(png, dpi=350, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print('Saved', png)
    print('Saved', pdf)


def capability(rows, by):
    models = [r['model'] for r in rows]
    sources = ['brain', 'chexpert', 'isic', 'mura', 'oct']
    c1 = {m: {} for m in models}
    for m in models:
        for s in sources:
            tr = load_total(ROOT / f'logs/capability/C1/{m}/anomaly-detection-{s}.scored.json')
            value = tr.get('rule_mcq_eval:accuracy_score')
            c1[m][s] = 100 * value if isinstance(value, (int, float)) else np.nan

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.32, wspace=0.24)
    ax = fig.add_subplot(gs[0, :2])
    im = grouped_heatmap(ax, models, sources, c1,
                         'C.1 accuracy by imaging source', 'Blues')
    ax.set_xticklabels([s.upper() for s in sources])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label('Accuracy (%)', fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for group, label, color, marker in [(MED, 'Medical', '#4E79A7', 'o'), (GEN, 'General', '#59A14F', 's'), (API, 'API', '#E15759', 'D')]:
        ax.scatter([by[m]['C2'] for m in group], [by[m]['C3'] for m in group],
                   s=55, label=label, color=color, marker=marker, alpha=0.85, edgecolor='white')
    ax.set_xlabel('C.2 localization', fontsize=8)
    ax.set_ylabel('C.3 reasoning', fontsize=8)
    ax.set_title('Localization vs. reasoning', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = fig.add_subplot(gs[1, :2])
    gap = [by[m]['C1'] - by[m]['C2'] for m in models]
    order = np.argsort(gap)
    colors = ['#4E79A7' if models[i] in MED else '#59A14F' if models[i] in GEN else '#E15759' for i in order]
    ax.barh(range(len(models)), [gap[i] for i in order], color=colors, zorder=3)
    ax.set_yticks(range(len(models)), labels=[NAMES[models[i]] for i in order], fontsize=7)
    ax.set_xlabel('C.1 accuracy minus C.2 score (points)', fontsize=8)
    ax.set_title('Recognition-to-localization gap', fontsize=11, fontweight='bold')
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter([by[m]['C2'] for m in models], [C2_MIOU[m] for m in models], s=45, color='#4E79A7', alpha=0.85)
    ax.set_xlabel('C.2 composite score', fontsize=8)
    ax.set_ylabel('Mean IoU only', fontsize=8)
    ax.set_title('Composite hides bbox quality', fontsize=11, fontweight='bold')
    style(ax)
    fig.suptitle('Capability: source-specific recognition and the localization bottleneck', fontsize=14, fontweight='bold')
    save(fig, 'dim_capability_diagnostics')


def robustness(rows, by):
    models = [r['model'] for r in rows]
    clean2, attack2, clean3, attack3 = {}, {}, {}, {}
    for m in models:
        tr = load_total(ROOT / f'logs/robustness/R2/{m}/adv-untarget-clean.scored.json')
        clean2[m] = 100 * tr.get('rule_mcq_eval:accuracy_score', np.nan)
        tr = load_total(ROOT / f'logs/robustness/R2/{m}/adv-untarget.scored.json')
        attack2[m] = 100 * tr.get('rule_mcq_eval:accuracy_score', np.nan)
        tr = load_total(ROOT / f'logs/robustness/R3/{m}/adv-target-clean.scored.json')
        clean3[m] = 100 * tr.get('rule_mcq_eval:accuracy_score', np.nan)
        tr = load_total(ROOT / f'logs/robustness/R3/{m}/adv-target.scored.json')
        attack3[m] = 100 * tr.get('rule_mcq_eval:accuracy_score', np.nan)

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)
    for col, (clean, attack, task) in enumerate([(clean2, attack2, 'R.2 untargeted'), (clean3, attack3, 'R.3 targeted')]):
        ax = fig.add_subplot(gs[0, col])
        for m in models:
            color = '#4E79A7' if m in MED else '#59A14F' if m in GEN else '#E15759'
            ax.plot([0, 1], [clean[m], attack[m]], color=color, alpha=0.55, linewidth=1.0)
            ax.scatter([0, 1], [clean[m], attack[m]], s=18, color=color, zorder=3)
        ax.set_xticks([0, 1], ['Clean', 'Attacked'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{task}: clean to attacked', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        style(ax)

    ax = fig.add_subplot(gs[0, 2])
    drops2 = [clean2[m] - attack2[m] for m in models]
    drops3 = [clean3[m] - attack3[m] for m in models]
    x = np.arange(2)
    ax.boxplot([drops2, drops3], positions=x, widths=0.5, patch_artist=True,
               boxprops={'facecolor': '#F28E2B', 'alpha': 0.65}, medianprops={'color': '#222222'})
    ax.set_xticks(x, ['R.2 untargeted', 'R.3 targeted'])
    ax.set_ylabel('Accuracy drop (points)')
    ax.set_title('Attack cost distribution', fontsize=11, fontweight='bold')
    style(ax)

    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(models))
    width = 0.38
    ax.bar(x - width / 2, [clean2[m] - attack2[m] for m in models], width, label='R.2 drop', color='#F28E2B', zorder=3)
    ax.bar(x + width / 2, [clean3[m] - attack3[m] for m in models], width, label='R.3 drop', color='#B85C00', zorder=3)
    ax.set_xticks(x, [NAMES[m] for m in models], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy drop (points)')
    ax.set_title('Per-model degradation under attack', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    branches = ['answer-flip', 'question-negation', 'option-expansion', 'narrative-distraction']
    labels = ['Flip', 'Negation', 'Expansion', 'Distraction']
    data = {m: {} for m in models}
    for m in models:
        for b, label in zip(branches, labels):
            tr = load_total(ROOT / f'logs/robustness/R4/{m}/robustness-vqa-{b}.scored.json')
            item = tr.get('rule_mcq_eval:robustness_acc', {})
            data[m][label] = 100 * item.get('accuracy', np.nan)
    means = [mean([data[m][label] for m in models]) for label in labels]
    ax.bar(labels, means, color=['#F28E2B', '#E15759', '#59A14F', '#4E79A7'], zorder=3)
    ax.set_ylabel('Mean branch accuracy (%)')
    ax.set_title('R.4 branch bottleneck', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    style(ax)

    fig.suptitle('Robustness: attack cost and the R.4 branch structure', fontsize=14, fontweight='bold')
    save(fig, 'dim_robustness_diagnostics')


def fairness(rows, by):
    models = [r['model'] for r in rows]
    groups = ['brain', 'isic', 'lung', 'mura', 'retina']
    orders = ['ab', 'ba']
    f1 = {m: {} for m in models}
    for m in models:
        data = json.loads((ROOT / f'logs/fairness/F1/{m}/visual-preference.scored.json').read_text())
        for g in groups:
            for o in orders:
                subset = [x for x in data.get('per_sample_results', [])
                          if x.get('extra', {}).get('group') == g and x.get('extra', {}).get('order') == o]
                f1[m][f'{g}-{o}'] = 100 * mean([1.0 if x.get('evals', {}).get('match') else 0.0 for x in subset])
    columns = [f'{g}-{o}' for g in groups for o in orders]

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    im = grouped_heatmap(ax, models, columns, f1,
                         'F.1 neutrality by image group and order', 'Greens', fmt='{:.0f}')
    ax.set_xticklabels([c.upper() for c in columns])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label('Neutrality (%)', fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for group, label, color, marker in [(MED, 'Medical', '#4E79A7', 'o'), (GEN, 'General', '#59A14F', 's'), (API, 'API', '#E15759', 'D')]:
        ax.scatter([by[m]['F1'] for m in group], [by[m]['F2'] for m in group],
                   s=55, label=label, color=color, marker=marker, alpha=0.85, edgecolor='white')
    ax.set_xlabel('F.1 neutrality')
    ax.set_ylabel('F.2 stability (UCR)')
    ax.set_title('Neutrality vs. response stability', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = fig.add_subplot(gs[0, 2])
    gaps = [abs(f1[m][f'{g}-ab'] - f1[m][f'{g}-ba']) for m in models for g in groups]
    ax.hist(gaps, bins=12, color='#59A14F', alpha=0.8, zorder=3)
    ax.set_xlabel('|AB - BA| neutrality gap (points)')
    ax.set_ylabel('Model-group pairs')
    ax.set_title('Order sensitivity of F.1', fontsize=11, fontweight='bold')
    style(ax)

    branches = ['race', 'language', 'emotion', 'cognitive']
    labels = ['Race', 'Language', 'Emotion', 'Cognitive']
    f2 = {m: {} for m in models}
    for m in models:
        for b, label in zip(branches, labels):
            tr = load_total(ROOT / f'logs/fairness/F2/{m}/bias-vqa-{b}.scored.json')
            item = tr.get('rule_mcq_eval:choice_unchange_rate', {})
            value = item.get('unchange_ratio')
            f2[m][label] = 100 * value if isinstance(value, (int, float)) else np.nan

    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(labels))
    width = 0.8 / 3
    for offset, (group, gname, color) in enumerate([(MED, 'Medical', '#4E79A7'), (GEN, 'General', '#59A14F'), (API, 'API', '#E15759')]):
        xs = x + (offset - 1) * width
        ys = [mean([f2[m][label] for m in group]) for label in labels]
        ax.bar(xs, ys, width, label=gname, color=color, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel('UCR (%)')
    ax.set_title('F.2 stability by challenge type and model group', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    spread = [max(f2[m].values()) - min(f2[m].values()) for m in models]
    order = np.argsort(spread)[-8:]
    ax.barh(range(len(order)), [spread[i] for i in order], color='#59A14F', zorder=3)
    ax.set_yticks(range(len(order)), labels=[NAMES[models[i]] for i in order], fontsize=7)
    ax.set_xlabel('Max minus min UCR (points)')
    ax.set_title('Who is most challenge-sensitive?', fontsize=11, fontweight='bold')
    style(ax)

    fig.suptitle('Fairness: visual preference, order sensitivity, and response-conditioned stability', fontsize=14, fontweight='bold')
    save(fig, 'dim_fairness_diagnostics')


def privacy(rows, by):
    models = [r['model'] for r in rows]
    classes = ['0', '1', '2']
    p1_acc = {m: {} for m in models}
    for m in models:
        data = json.loads((ROOT / f'logs/privacy/P1/{m}/privacy-recognition.scored.json').read_text())
        for cls in classes:
            subset = [x for x in data.get('per_sample_results', []) if str(x.get('target')) == cls]
            p1_acc[m][cls] = 100 * mean([1.0 if x.get('evals', {}).get('match') else 0.0 for x in subset])

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, :2])
    im = grouped_heatmap(ax, models, classes, p1_acc,
                         'P.1 classwise accuracy (clean / PHI / noise)', 'YlGnBu')
    ax.set_xticklabels(['class 0', 'class 1', 'class 2'])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label('Accuracy (%)', fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    p2 = {}
    for m in models:
        tr = load_total(ROOT / f'logs/privacy/P2/{m}/privacy-detection.scored.json')
        p2[m] = (100 * tr.get('privacy_phi_detection_eval:precision_score', np.nan),
                 100 * tr.get('privacy_phi_detection_eval:recall_score', np.nan))
    ax.scatter([p2[m][0] for m in models], [p2[m][1] for m in models], s=50, color='#76B7B2', edgecolor='white')
    ax.plot([0, 100], [0, 100], linestyle='--', color='#999999', linewidth=0.8)
    ax.set_xlabel('P.2 precision')
    ax.set_ylabel('P.2 recall')
    ax.set_title('PHI detection: precision vs. recall', fontsize=11, fontweight='bold')
    style(ax)

    ppr, ras = {}, {}
    for task in ['P3', 'P4']:
        folder = f'privacy/{task}'
        ppr[task], ras[task] = {}, {}
        for m in models:
            nums = defaultdict(float)
            dens = defaultdict(float)
            for path in sorted((ROOT / f'logs/{folder}/{m}').glob('*.scored.json')):
                tr = load_total(path).get('metrics_0_to_100', {})
                for key in ['PPR', 'RAS']:
                    value, n = tr.get(key), tr.get(f'{key}_n')
                    if isinstance(value, (int, float)) and isinstance(n, (int, float)) and n:
                        nums[key] += value * n
                        dens[key] += n
            ppr[task][m] = nums['PPR'] / dens['PPR'] if dens['PPR'] else np.nan
            ras[task][m] = nums['RAS'] / dens['RAS'] if dens['RAS'] else np.nan

    for col, task in enumerate(['P3', 'P4']):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter([ppr[task][m] for m in models], [ras[task][m] for m in models], s=50, color='#76B7B2', edgecolor='white')
        ax.set_xlabel(f'{task} PPR (protection)')
        ax.set_ylabel(f'{task} RAS (utility)')
        ax.set_title(f'{task}: protection vs. utility', fontsize=11, fontweight='bold')
        style(ax)

    ax = fig.add_subplot(gs[1, 2])
    branches = ['focus-distraction', 'subtle-request', 'trap-warning', 'well-intention']
    labels = ['Focus', 'Subtle', 'Trap', 'Intent']
    branch_data = {m: {} for m in models}
    for m in models:
        for b, label in zip(branches, labels):
            tr = load_total(ROOT / f'logs/privacy/P4/{m}/privacy-vqa-{b}.scored.json').get('metrics_0_to_100', {})
            p, r = tr.get('PPR'), tr.get('RAS')
            branch_data[m][label] = (p + r) / 2 if isinstance(p, (int, float)) and isinstance(r, (int, float)) else np.nan
    means = [mean([branch_data[m][label] for m in models]) for label in labels]
    ax.bar(labels, means, color=['#76B7B2', '#4E79A7', '#E15759', '#59A14F'], zorder=3)
    ax.set_ylabel('Mean (PPR+RAS)/2')
    ax.set_title('P.4 rewrite-branch difficulty', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    style(ax)

    fig.suptitle('Privacy: recognition classes, PHI precision/recall, and protection after rewrite', fontsize=14, fontweight='bold')
    save(fig, 'dim_privacy_diagnostics')


def safety(rows, by):
    models = [r['model'] for r in rows]
    s1, s2, s3 = {}, {}, {}
    for m in models:
        tr = load_total(ROOT / f'logs/safety/S1/{m}/safety-risk.scored.json').get('metrics_0_to_100', {})
        s1[m] = (tr.get('HCR'), tr.get('RAS'))
        tr = load_total(ROOT / f'logs/safety/S2/{m}/mcm-jailbreak.scored.json').get('metrics_0_to_100', {})
        s2[m] = (tr.get('HCR'), tr.get('RAS'))
        tr = load_total(ROOT / f'logs/safety/S3/{m}/safety-vqa-authority.scored.json').get('metrics_0_to_100', {})
        s3[m] = (tr.get('HCR'), tr.get('RAS'))

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    for col, (task, data, color) in enumerate([('S.1 direct risk', s1, '#E15759'), ('S.2 jailbreak', s2, '#F28E2B'), ('S.3 rewritten', s3, '#4E79A7')]):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter([data[m][0] for m in models], [data[m][1] for m in models], s=50, color=color, edgecolor='white')
        ax.set_xlabel('HCR (lower is safer)')
        ax.set_ylabel('RAS (higher is more useful)')
        ax.set_title(task, fontsize=11, fontweight='bold')
        style(ax)

    ax = fig.add_subplot(gs[1, 0])
    for group, label, color, marker in [(MED, 'Medical', '#4E79A7', 'o'), (GEN, 'General', '#59A14F', 's'), (API, 'API', '#E15759', 'D')]:
        ax.scatter([by[m]['S1'] for m in group], [by[m]['S3'] for m in group],
                   s=55, label=label, color=color, marker=marker, alpha=0.85, edgecolor='white')
    ax.set_xlabel('S.1 direct-risk score')
    ax.set_ylabel('S.3 rewritten-input score')
    ax.set_title('Safety persistence', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    branches = ['authority', 'consequence-obfuscation', 'diagnostic-anchoring', 'risk-minimization', 'urgency']
    labels = ['Authority', 'Consequence', 'Anchoring', 'Minimization', 'Urgency']
    branch = {m: {} for m in models}
    for m in models:
        for b, label in zip(branches, labels):
            tr = load_total(ROOT / f'logs/safety/S3/{m}/safety-vqa-{b}.scored.json').get('metrics_0_to_100', {})
            h, r = tr.get('HCR'), tr.get('RAS')
            branch[m][label] = (100 - h + r) / 2 if isinstance(h, (int, float)) and isinstance(r, (int, float)) else np.nan

    ax = fig.add_subplot(gs[1, 1:])
    x = np.arange(len(labels))
    width = 0.8 / 3
    for offset, (group, gname, color) in enumerate([(MED, 'Medical', '#4E79A7'), (GEN, 'General', '#59A14F'), (API, 'API', '#E15759')]):
        xs = x + (offset - 1) * width
        ys = [mean([branch[m][label] for m in group]) for label in labels]
        ax.bar(xs, ys, width, label=gname, color=color, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Safety score (0-100)')
    ax.set_title('S.3 rewrite type: where refusal breaks down', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    fig.suptitle('Safety: direct refusal, jailbreak resistance, and rewrite persistence', fontsize=14, fontweight='bold')
    save(fig, 'dim_safety_diagnostics')


C2_MIOU = {}


def main():
    rows = load_rows()
    by = {r['model']: r for r in rows}
    for m in by:
        tr = load_total(ROOT / f'logs/capability/C2/{m}/lesion-localization.scored.json')
        C2_MIOU[m] = 100 * tr.get('rule_bbox_eval:mean_iou', np.nan)
    capability(rows, by)
    robustness(rows, by)
    fairness(rows, by)
    privacy(rows, by)
    safety(rows, by)


if __name__ == '__main__':
    main()
