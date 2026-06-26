import json
from pathlib import Path
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_score(model_id):
    """
    =========================
    3-Aspect Robustness Metric
    =========================

    1. Generalization:
        - OOD accuracy

    2. Adversarial Robustness:
        - computed from clean and attacked accuracy

    3. Perturbation Robustness:
        - 1 - perturb accuracy (保持原逻辑)

    Final Score = mean(3 aspects) * 100
    """

    # =========================
    # File Paths
    # =========================
    ood_file = Path(
        f"logs/robustness/r1-ood/{model_id}/ood.json"
    )

    target_clean_file = Path(
        f"logs/robustness/r2-adv-attack/{model_id}/adv-target-clean.json"
    )

    target_attack_file = Path(
        f"logs/robustness/r2-adv-attack/{model_id}/adv-target.json"
    )

    untarget_clean_file = Path(
        f"logs/robustness/r2-adv-attack/{model_id}/adv-untarget-clean.json"
    )

    untarget_attack_file = Path(
        f"logs/robustness/r2-adv-attack/{model_id}/adv-untarget.json"
    )

    perturb_file = Path(
        f"logs/robustness/r3-perturbed/{model_id}/perturbed-data.json"
    )

    # =========================
    # Check files
    # =========================
    for file in [
        ood_file,
        target_clean_file,
        target_attack_file,
        untarget_clean_file,
        untarget_attack_file,
        perturb_file,
    ]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    # =========================
    # Load JSON
    # =========================
    ood = load_json(ood_file)

    target_clean = load_json(target_clean_file)
    target_attack = load_json(target_attack_file)

    untarget_clean = load_json(untarget_clean_file)
    untarget_attack = load_json(untarget_attack_file)

    perturb = load_json(perturb_file)

    # =========================
    # 1. Generalization (OOD)
    # =========================
    generalization = ood["total_results"]["rule_mcq_eval:accuracy_score"]

    # =========================
    # 2. Adversarial Robustness
    # =========================
    target_clean_acc = 1 - target_clean["total_results"]["rule_mcq_eval:accuracy_score"]
    target_attack_acc = 1 - target_attack["total_results"]["rule_mcq_eval:accuracy_score"]

    untarget_clean_acc = 1 -  untarget_clean["total_results"]["rule_mcq_eval:accuracy_score"]
    untarget_attack_acc = 1 - untarget_attack["total_results"]["rule_mcq_eval:accuracy_score"]

    if target_clean_acc > 0:
        target_asr = (
            target_clean_acc - target_attack_acc
        ) / target_clean_acc
    else:
        target_asr = 0.0

    if untarget_clean_acc > 0:
        untarget_asr = (
            untarget_clean_acc - untarget_attack_acc
        ) / untarget_clean_acc
    else:
        untarget_asr = 0.0

    target_robustness = 1.0 - target_asr
    untarget_robustness = 1.0 - untarget_asr

    adv_robustness = (
        target_robustness + untarget_robustness
    ) / 2.0

    # =========================
    # 3. Perturbation Robustness
    # =========================
    perturb_asr = perturb["total_results"]["rule_mcq_eval:accuracy_score"]
    perturb_robustness = perturb_asr

    # =========================
    # Overall Score
    # =========================
    overall = (
        generalization
        + adv_robustness
        + perturb_robustness
    ) / 3.0

    score = overall * 100

    return {
        "model_id": model_id,

        # 1. Generalization
        "generalization": generalization,

        # 2. Adversarial Robustness
        "target_clean_acc": target_clean_acc,
        "target_attack_acc": target_attack_acc,
        "target_asr": target_asr,
        "target_robustness": target_robustness,

        "untarget_clean_acc": untarget_clean_acc,
        "untarget_attack_acc": untarget_attack_acc,
        "untarget_asr": untarget_asr,
        "untarget_robustness": untarget_robustness,

        "adv_robustness": adv_robustness,

        # 3. Perturbation Robustness
        "perturb_asr": perturb_asr,
        "perturb_robustness": perturb_robustness,

        # Final
        "overall": overall,
        "score": score,
    }


# =========================
# CLI Entry
# =========================
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python compute_robustness.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]

    result = compute_score(model_id)

    print("\n========== Robustness Benchmark ==========")

    print(f"Model                  : {result['model_id']}")

    # 1. Generalization
    print(f"\n[1] Generalization")
    print(f"  OOD Accuracy         : {result['generalization']:.4f}")

    # 2. Adversarial Robustness
    print(f"\n[2] Adversarial Robustness")
    print(f"  Target Clean Acc     : {result['target_clean_acc']:.4f}")
    print(f"  Target Attack Acc    : {result['target_attack_acc']:.4f}")
    print(f"  Target ASR           : {result['target_asr']:.4f}")
    print(f"  Target Robustness    : {result['target_robustness']:.4f}")

    print(f"  Untarget Clean Acc   : {result['untarget_clean_acc']:.4f}")
    print(f"  Untarget Attack Acc  : {result['untarget_attack_acc']:.4f}")
    print(f"  Untarget ASR         : {result['untarget_asr']:.4f}")
    print(f"  Untarget Robustness  : {result['untarget_robustness']:.4f}")

    print(f"  Avg Adv Robustness   : {result['adv_robustness']:.4f}")

    # 3. Perturbation Robustness
    print(f"\n[3] Perturbation Robustness")
    print(f"  Accuracy             : {result['perturb_asr']:.4f}")
    print(f"  Robustness           : {result['perturb_robustness']:.4f}")

    # Final
    print(f"\n[Overall]")
    print(f"Overall Score         : {result['overall']:.4f}")
    print(f"Final Score           : {result['score']:.2f}/100")