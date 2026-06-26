from typing import Any, Dict, List, Set
import os
import json
import argparse


def compute_unchange_intersection(
    json_files,
    output_file="bias-vqa.json",
):
    # 检查文件是否都存在
    missing_files = [f for f in json_files if not os.path.exists(f)]
    if missing_files:
        print("Error: 以下 JSON 文件不存在，无法计算交集：")
        for f in missing_files:
            print(f"  - {f}")
        return None

    unchange_sets = []
    processed_preds = []
    processed_labels = []

    # =========================
    # 加载所有 JSON
    # =========================
    for i, json_path in enumerate(json_files):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids = set(
            data["total_results"][
                "rule_mcq_eval:choice_unchange_rate"
            ].get("unchange_ids", [])
        )

        unchange_sets.append(ids)

        print(
            f"{os.path.basename(json_path)}: "
            f"{len(ids)} unchange ids"
        )

        # 只读取第一个文件中的 processed_summary
        if i == 0:
            summary = data["total_results"].get(
                "rule_mcq_eval:processed_summary",
                {}
            )

            processed_preds = summary.get(
                "processed_preds",
                []
            )

            processed_labels = summary.get(
                "processed_labels",
                []
            )

    # =========================
    # 求交集
    # =========================
    if len(unchange_sets) == 0:
        intersection = []
    else:
        intersection = sorted(
            set.intersection(*unchange_sets)
        )

    # =========================
    # 计算比例
    # =========================
    total_num = len(processed_preds)

    unchange_ratio = (
        len(intersection) / total_num
        if total_num > 0
        else 0.0
    )

    # =========================
    # 根据交集重建 processed_preds
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
            "rule_mcq_eval:choice_unchange_rate": {
                "unchange_ratio": unchange_ratio,
                "unchange_ids": intersection,
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

    print(f"Saved to {output_file}")
    print(f"Intersection size: {len(intersection)}")
    print(f"Unchange ratio: {unchange_ratio:.6f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_files",
        nargs=4,
        required=True,
        help="Paths to bias-vqa-race/language/emotion/cognitive json files",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Output json file",
    )

    args = parser.parse_args()

    compute_unchange_intersection(
        json_files=args.json_files,
        output_file=args.output_file,
    )