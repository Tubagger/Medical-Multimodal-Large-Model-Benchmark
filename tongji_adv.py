import os
import json
from collections import defaultdict

SUFFIXES = ["clean", "eps4", "eps8", "eps16"]


def detect_suffix(filename):
    filename = filename.lower()

    if "eps4" in filename:
        return "eps4"
    if "eps8" in filename:
        return "eps8"
    if "eps16" in filename:
        return "eps16"
    if "clean" in filename:
        return "clean"

    return None


def detect_type(filename):
    filename = filename.lower()

    if "adv-target" in filename:
        return "target"
    if "adv-untarget" in filename:
        return "untarget"

    return None


def stat_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["total_results"]["rule_mcq_eval:accuracy_score"]


def main(root="r2-adv-attack"):

    result = defaultdict(lambda: defaultdict(dict))

    for model in os.listdir(root):
        model_path = os.path.join(root, model)

        if not os.path.isdir(model_path):
            continue

        print(f"\n[INFO] model: {model}")

        for file in os.listdir(model_path):

            if not file.endswith(".json"):
                continue

            suffix = detect_suffix(file)
            ftype = detect_type(file)

            if suffix is None or ftype is None:
                continue

            path = os.path.join(model_path, file)

            try:
                acc = stat_file(path)
            except Exception as e:
                print(f"[WARN] {file}: {e}")
                continue

            # ⭐ 核心：双键存储
            result[model][suffix][ftype] = acc

    # -----------------------
    # 输出
    # -----------------------
    print("\n=========== RESULT ===========\n")

    for model, data in result.items():
        print(f"===== {model} =====")

        for s in SUFFIXES:
            if s in data:
                t = data[s].get("target", None)
                u = data[s].get("untarget", None)

                print(
                    f"{s:8s} | "
                    f"target: {t} | "
                    f"untarget: {u}"
                )
            else:
                print(f"{s:8s} | MISSING")


if __name__ == "__main__":
    main("logs/robustness/r2-adv-attack")