import json
import re


def extract_option(response):
    """
    从 response 中提取选项字母。

    支持：
        A
        A: xxx
        A. xxx
        (A)
        (A) xxx
        Answer: B
        Final Answer: C
    """
    if not isinstance(response, str):
        return None

    text = response.strip().upper()

    # 1. Answer: A / Final Answer: B
    m = re.search(
        r"(?:FINAL ANSWER|FINAL|ANSWER|ANS)\s*[:\-]?\s*\(?([A-Z])\)?",
        text,
    )
    if m:
        return m.group(1)

    # 2. 开头就是 A / A: / A. / (A)
    m = re.match(
        r"^\s*\(?([A-Z])\)?(?:\s*[:\.])?",
        text,
    )
    if m:
        return m.group(1)

    return None


def update_json(json_path, output_path=None):
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["per_sample_results"]

    processed_preds = []

    # 遍历所有样本
    for sample in samples:
        response = sample.get("response", "")

        # 从 response 提取 pred_opt
        pred_opt = extract_option(response)

        # 更新 pred_opt
        sample["evals"]["pred_opt"] = pred_opt

        # 获取 true_opt
        true_opt = sample["evals"].get("true_opt")

        # 更新 match
        match = (pred_opt == true_opt)
        sample["evals"]["match"] = match

        # 更新 processed_preds
        processed_preds.append(1 if match else 0)

    # 更新 processed_summary
    data["total_results"]["rule_mcq_eval:processed_summary"][
        "processed_preds"
    ] = processed_preds

    # 更新 unchange_ids
    unchange_ids = [
        idx
        for idx, value in enumerate(processed_preds)
        if value == 1
    ]

    data["total_results"]["rule_mcq_eval:choice_unchange_rate"][
        "unchange_ids"
    ] = unchange_ids

    # 更新 unchange_ratio
    if len(processed_preds) > 0:
        unchange_ratio = len(unchange_ids) / len(processed_preds)
    else:
        unchange_ratio = 0.0

    data["total_results"]["rule_mcq_eval:choice_unchange_rate"][
        "unchange_ratio"
    ] = unchange_ratio

    # 默认覆盖原文件
    if output_path is None:
        output_path = json_path

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("=" * 60)
    print(f"Processed samples : {len(samples)}")
    print(f"Correct samples   : {len(unchange_ids)}")
    print(f"Unchange ratio    : {unchange_ratio:.6f}")
    print(f"Saved to          : {output_path}")


if __name__ == "__main__":
    # 输入你的 JSON 文件路径
    json_path = "logs/fairness/f1-bias-ref/lingshu-7b/bias-vqa-cognitive-error.json"

    # 如果想覆盖原文件，保持 None 即可
    # output_path = None

    # 如果想另存为新文件，例如：
    output_path = "logs/fairness/f1-bias-ref/lingshu-7b/bias-vqa-cognitive.json"

    update_json(json_path, output_path)