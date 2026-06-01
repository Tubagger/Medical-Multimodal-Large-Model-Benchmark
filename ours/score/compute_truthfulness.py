import json
from pathlib import Path
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_score(model_id):
    """
    Perception =
        0.7 * Classification
      + 0.3 * Localization

    Reasoning =
        MCQ Accuracy

    Overall =
        0.6 * Perception
      + 0.4 * Reasoning

    Final Score = Overall * 100
    """

    # =========================
    # 文件路径
    # =========================
    chexpert_file = Path(
        f"logs/truthfulness/t1-basic/{model_id}/anomaly-detection.json"
    )

    bbox_file = Path(
        f"logs/truthfulness/t1-basic/{model_id}/lesion-localization.json"
    )

    mcq_file = Path(
        f"logs/truthfulness/t2-logic/{model_id}/logical-reasoning.json"
    )

    # =========================
    # 读取JSON
    # =========================
    if not chexpert_file.exists():
        raise FileNotFoundError(
            f"Missing file: {chexpert_file}"
        )

    if not bbox_file.exists():
        raise FileNotFoundError(
            f"Missing file: {bbox_file}"
        )

    if not mcq_file.exists():
        raise FileNotFoundError(
            f"Missing file: {mcq_file}"
        )
    chexpert = load_json(chexpert_file)
    bbox = load_json(bbox_file)
    mcq = load_json(mcq_file)

    # =========================
    # Classification
    # =========================
    classification = (
        chexpert["total_results"]
        ["rule_chexpert_eval:multiclass_metrics"]
        ["accuracy"]
    )

    # =========================
    # Localization
    # =========================
    acc_iou = bbox["total_results"][
        "rule_bbox_eval:acc_at_iou"
    ]

    mean_iou = bbox["total_results"][
        "rule_bbox_eval:mean_iou"
    ]

    localization = (
        acc_iou + mean_iou
    ) / 2

    # =========================
    # Reasoning
    # =========================
    reasoning = mcq["total_results"][
        "rule_mcq_eval:accuracy_score"
    ]

    # =========================
    # Perception
    # =========================
    perception = (
        0.7 * classification
        + 0.3 * localization
    )

    # =========================
    # Overall
    # =========================
    overall = (
        0.6 * perception
        + 0.4 * reasoning
    )

    score = overall * 100

    return {
        "model_id": model_id,
        "classification": classification,
        "localization": localization,
        "reasoning": reasoning,
        "perception": perception,
        "overall": overall,
        "score": score,
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python compute_truthfulness.py <model_id>"
        )
        sys.exit(1)

    model_id = sys.argv[1]

    result = compute_score(model_id)

    print("\n========== Benchmark Score ==========")
    print(f"Model          : {result['model_id']}")
    print(f"Classification : {result['classification']:.4f}")
    print(f"Localization   : {result['localization']:.4f}")
    print(f"Reasoning      : {result['reasoning']:.4f}")
    print(f"Perception     : {result['perception']:.4f}")
    print(f"Overall        : {result['overall']:.4f}")
    print(f"Final Score    : {result['score']:.2f}/100")