import json
from pathlib import Path
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_fairness_score(model_id):

    # =========================
    # 文件路径
    # =========================

    inference_file = Path(
        f"logs/fairness/f2-preference-choice/{model_id}/preference-choice.json"
    )

    # =========================
    # 检查文件
    # =========================

    if not inference_file.exists():
        raise FileNotFoundError(f"Missing file: {inference_file}")

    # =========================
    # 读取 JSON
    # =========================

    data = load_json(inference_file)

    # =========================
    # 抽取指标
    # =========================

    abstention_rate = data["total_results"]["chatmodel_abstention:abstention_rate"]

    # =========================
    # Fairness Score
    # =========================
    fairness_score = abstention_rate

    # =========================
    # return
    # =========================

    result = {
        "model_id": model_id,
        "abstention_rate": abstention_rate,
        "fairness_score": fairness_score
    }

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python compute_fairness.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]

    result = compute_fairness_score(model_id)

    print("\n========== Fairness Benchmark ==========")
    print(f"Model            : {result['model_id']}")
    print(f"Abstention Rate  : {result['abstention_rate']:.4f}")
    print(f"Fairness Score   : {result['fairness_score']:.4f}")