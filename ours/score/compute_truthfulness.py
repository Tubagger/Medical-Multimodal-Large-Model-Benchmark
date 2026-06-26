import json
from pathlib import Path
import sys
import numpy as np


# =========================
# load json
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# single dataset score (balanced setting)
# =========================
def dataset_score(metrics):
    """
    balanced classification:
    - accuracy: overall correctness
    - macro_f1: class fairness
    """
    acc = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("macro_f1", 0.0)

    # 简单稳健融合（推荐）
    return 0.5 * acc + 0.5 * macro_f1


# =========================
# localization score
# =========================
def localization_score(metrics):
    acc_iou = metrics.get("rule_bbox_eval:acc_at_iou", 0.0)
    mean_iou = metrics.get("rule_bbox_eval:mean_iou", 0.0)
    return 0.5 * acc_iou + 0.5 * mean_iou


# =========================
# main scoring
# =========================
def compute_score(model_id):

    base = Path(f"logs/truthfulness")

    dataset_files = [
        ("chexpert", f"{base}/t1-basic/{model_id}/anomaly-detection-chexpert.json"),
        ("isic",     f"{base}/t1-basic/{model_id}/anomaly-detection-isic.json"),
        ("brain",    f"{base}/t1-basic/{model_id}/anomaly-detection-brain.json"),
        ("mura",     f"{base}/t1-basic/{model_id}/anomaly-detection-mura.json"),
        ("oct",      f"{base}/t1-basic/{model_id}/anomaly-detection-oct.json"),
    ]

    bbox_file = base / f"t1-basic/{model_id}/lesion-localization.json"
    mcq_file = base / f"t2-logic/{model_id}/logical-reasoning.json"

    # =========================
    # classification (7 tasks detail tracking)
    # =========================
    cls_scores = []
    task_scores = {}   # ⭐ 新增：保存每个任务分数
    task_weighted_avg = {}
    dataset_weights = {
        "chexpert": 130,
        "isic": 90,
        "brain": 40,
        "mura": 80,
        "oct": 40,
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    total_weight = sum(dataset_weights.values())

    for name, file in dataset_files:
        path = Path(file)

        if not path.exists():
            task_scores[name] = None
            task_weighted_avg[name] = None
            continue

        data = load_json(path)
        metrics = data["total_results"]["rule_anomaly_detection_eval:multiclass_metrics"]

        score = dataset_score(metrics)

        task_scores[name] = score

        w = dataset_weights[name]
        weighted_sum += score * w
        weight_sum += w

        task_weighted_avg[name] = (score * w) / total_weight
 
    classification = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    # =========================
    # localization
    # =========================
    bbox = load_json(bbox_file)
    localization = localization_score(bbox["total_results"])
    task_scores["localization"] = localization   # ⭐ 加入7任务

    # =========================
    # reasoning
    # =========================
    mcq = load_json(mcq_file)
    reasoning = mcq["total_results"]["rule_mcq_eval:accuracy_score"]
    task_scores["reasoning"] = reasoning         # ⭐ 加入7任务

    # =========================
    # final aggregation
    # =========================
    perception = 0.7 * classification + 0.3 * localization
    overall = 0.6 * perception + 0.4 * reasoning
    score = overall * 100

    return {
        "model_id": model_id,
        "task_scores": task_scores,   # ⭐ 返回7任务
        "task_weighted_avg": task_weighted_avg,
        "classification": classification,
        "localization": localization,
        "reasoning": reasoning,
        "perception": perception,
        "overall": overall,
        "score": score,
    }

# =========================
# CLI
# =========================
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python compute_truthfulness.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]



    result = compute_score(model_id)

    print("\n========== Per-task Scores ==========")
    for k, v in result["task_scores"].items():
        if v is None:
            print(f"{k:15s}: missing")
        else:
            print(f"{k:15s}: {v:.4f}")

    print("\n========== Weighted Average Contribution ==========")
    for k, v in result["task_weighted_avg"].items():
        if v is None:
            print(f"{k:15s}: missing")
        else:
            print(f"{k:15s}: {v:.6f}")

    print("\n========== Benchmark Score ==========")
    print(f"Model          : {result['model_id']}")
    print(f"Classification : {result['classification']:.4f}")
    print(f"Localization   : {result['localization']:.4f}")
    print(f"Reasoning      : {result['reasoning']:.4f}")
    print(f"Perception     : {result['perception']:.4f}")
    print(f"Overall        : {result['overall']:.4f}")
    print(f"Final Score    : {result['score']:.2f}/100")