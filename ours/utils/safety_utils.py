try:
    from .response_records import default_score_file
except ImportError:
    from response_records import default_score_file

import argparse
import json
import os
from typing import Any, Dict, List, Set


def compute_unchange_intersection(
    json_files: List[str],
    output_file: str = "safety-vqa.json",
) -> Dict[str, Any]:

    json_files = [default_score_file(path) for path in json_files]
    output_file = default_score_file(output_file)
    missing_files = [f for f in json_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            "以下 JSON 文件不存在：\n" + "\n".join(missing_files)
        )

    remain_sets: List[Set[int]] = []
    processed_labels = []
    total_num = None  # 更安全

    # =========================
    # load all files
    # =========================
    for i, json_path in enumerate(json_files):

        print(f"Loading: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        totals = data.get("total_results", {})
        records = data.get("per_sample_results", [])
        abstention_block = totals.get("chatmodel_abstention:abstention_rate", {})
        if abstention_block:
            remain_ids = set(abstention_block.get("remain_ids", []))
        else:
            remain_ids = {idx for idx, row in enumerate(records)
                          if row.get("evals") and not row["evals"].get("safety", {}).get("harmful_compliance", 0)}
        remain_sets.append(remain_ids)

        print(f"{os.path.basename(json_path)}: {len(remain_ids)} remain ids")

        # =========================
        # only first file for metadata
        # =========================
        # if i == 0:
        #     # 尝试兼容不同结构
        #     summary_block = data.get("total_results", {})

        #     # fallback：如果你未来有labels再加
        #     processed_labels = summary_block.get("processed_labels", [])

        #     # 如果没有labels，就用 remain_ids最大索引推断
        #     if "processed_summary" in summary_block:
        #         total_num = len(
        #             summary_block["processed_summary"].get("processed_preds", [])
        #         )
        #     else:
        #         # fallback：用最大 remain id 推断长度
        #         all_ids = abstention_block.get("remain_ids", [])
        #         total_num = max(all_ids) + 1 if all_ids else 0

        if i == 0:
            summary = totals.get("chatmodel_abstention:processed_summary", {})
            processed_labels = summary.get("processed_labels", [])
            total_num = len(summary.get("processed_preds", [])) or len(records)
    # =========================
    # intersection
    # =========================
    intersection = (
        sorted(set.intersection(*remain_sets))
        if remain_sets
        else []
    )

    # =========================
    # ratio
    # =========================
    if total_num is None or total_num == 0:
        abstention_rate = 0.0
    else:
        abstention_rate = len(intersection) / total_num

    # =========================
    # rebuild preds
    # =========================
    new_processed_preds = [0] * total_num

    for idx in intersection:
        if 0 <= idx < total_num:
            new_processed_preds[idx] = 1

    # =========================
    # output
    # =========================
    result = {
        "total_results": {
            "chatmodel_abstention:abstention_rate": {
                "abstention_rate": abstention_rate,
                "remain_ids": intersection,
            },
            "chatmodel_abstention:processed_summary": {
                "processed_preds": new_processed_preds,
                "processed_labels": processed_labels,
            },
        }
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\nSaved to:", output_file)
    print("Intersection size:", len(intersection))
    print("abstention_rate:", round(abstention_rate, 6))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files", nargs="+", required=True)
    parser.add_argument("--output_file", type=str, default="abstention.json")
    args = parser.parse_args()

    compute_unchange_intersection(
        args.json_files,
        args.output_file
    )