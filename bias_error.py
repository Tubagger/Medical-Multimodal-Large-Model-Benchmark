import json
import re


def extract_option_from_response(response):
    """
    从 response 中提取选项字母。

    支持：
        A
        A: xxx
        A. xxx
        (A)
        (A) xxx
    """
    if not isinstance(response, str):
        return None

    response = response.strip().upper()

    m = re.match(r"^\s*\(?([A-Z])\)?(?:\s*[:\.])?", response)
    if m:
        return m.group(1)

    return None


def check_pred_opt(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 你的结果保存在 per_sample_results 中
    if "per_sample_results" not in data:
        raise ValueError("Cannot find 'per_sample_results' in JSON.")

    samples = data["per_sample_results"]

    mismatch_ids = []

    for sample in samples:
        sample_id = sample.get("id")

        response = sample.get("response", "")
        pred_opt = sample.get("evals", {}).get("pred_opt")

        response_opt = extract_option_from_response(response)

        if response_opt != pred_opt:
            mismatch_ids.append(sample_id)

            print("=" * 60)
            print(f"id           : {sample_id}")
            print(f"response     : {response}")
            print(f"response_opt : {response_opt}")
            print(f"pred_opt     : {pred_opt}")

    print("\n================ Summary ================")
    print(f"Total samples    : {len(samples)}")
    print(f"Mismatch count   : {len(mismatch_ids)}")
    print(f"Mismatch IDs     : {mismatch_ids}")


if __name__ == "__main__":
    check_pred_opt(
        "logs/robustness/r2-adv-attack/lingshu-7b/adv-target-clean.json"
    )