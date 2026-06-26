import argparse
import json
import os
from typing import Any, Dict, List, Set


def compute_acc_intersection(
    json_files: List[str],
    output_file: str = "robustness-vqa.json",
) -> Dict[str, Any]:
    """
    计算多个 JSON 文件中 remain_ids 的交集，并保存结果。

    输入格式:
    {
        "total_results": {
            "rule_mcq_eval:robustness_acc": {
                "accuracy": 0.03,
                "remain_ids": [...]
            },
            "rule_mcq_eval:processed_summary": {
                "processed_preds": [...],
                "processed_labels": [...]
            }
        }
    }

    输出格式:
    {
        "total_results": {
            "rule_mcq_eval:robustness_acc": {
                "accuracy": ...,
                "remain_ids": [...]
            },
            "rule_mcq_eval:processed_summary": {
                "processed_preds": [...],
                "processed_labels": [...]
            }
        }
    }
    """

    # =========================
    # 检查文件是否存在
    # =========================
    missing_files = [f for f in json_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            "以下 JSON 文件不存在：\n" + "\n".join(missing_files)
        )

    remain_sets: List[Set[int]] = []

    processed_labels = []
    total_num = 0

    # =========================
    # 加载所有 JSON
    # =========================
    for i, json_path in enumerate(json_files):

        print(f"Loading: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        robustness_result = data["total_results"].get(
            "rule_mcq_eval:robustness_acc", {}
        )

        remain_ids = set(
            robustness_result.get("remain_ids", [])
        )

        remain_sets.append(remain_ids)

        print(
            f"{os.path.basename(json_path)}: "
            f"{len(remain_ids)} remain ids"
        )

        # 只从第一个文件读取 processed_summary
        if i == 0:
            summary = data["total_results"].get(
                "rule_mcq_eval:processed_summary",
                {},
            )

            processed_labels = summary.get(
                "processed_labels", []
            )

            total_num = len(
                summary.get("processed_preds", [])
            )

    # =========================
    # 求交集
    # =========================
    if not remain_sets:
        intersection = []
    else:
        intersection = sorted(
            list(set.intersection(*remain_sets))
        )

    # =========================
    # 计算 accuracy
    # =========================
    accuracy = (
        len(intersection) / total_num
        if total_num > 0
        else 0.0
    )

    # =========================
    # 根据交集重建 processed_preds
    # remain_ids 对应位置为 1
    # =========================
    new_processed_preds = [0] * total_num

    for idx in intersection:
        if 0 <= idx < total_num:
            new_processed_preds[idx] = 1

    # =========================
    # 保存结果
    # =========================
    result = {
        "total_results": {
            "rule_mcq_eval:robustness_acc": {
                "accuracy": accuracy,
                "remain_ids": intersection,
            },
            "rule_mcq_eval:processed_summary": {
                "processed_preds": new_processed_preds,
                "processed_labels": processed_labels,
            },
        }
    }

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nSaved to {output_file}")
    print(f"Intersection size: {len(intersection)}")
    print(f"Accuracy: {accuracy:.6f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute intersection of remain_ids across multiple JSON files."
    )

    parser.add_argument(
        "--json_files",
        nargs="+",
        required=True,
        help="Input JSON files.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="robustness-vqa.json",
        help="Output JSON file.",
    )

    args = parser.parse_args()

    compute_acc_intersection(
        json_files=args.json_files,
        output_file=args.output_file,
    )