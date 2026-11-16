#!/usr/bin/env python3
"""Validate complete P.2 parser decisions and aggregate metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.migrate_p2_logs import EXPECTED_PROMPT_SHA256, evaluate_row  # noqa: E402


def validate(dataset: Path, logs: Path) -> dict:
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    expected = {str(row["id"]): row for row in labels}
    migration_path = logs / 'migration_report.json'
    if not migration_path.exists():
        migration_path = ROOT / 'run_state/log_metadata' / logs.resolve().relative_to((ROOT / 'logs').resolve()) / 'migration_report.json'
    migration = json.loads(migration_path.read_text(encoding='utf-8'))
    model_dirs = sorted(path for path in logs.iterdir() if path.is_dir())
    errors: list[str] = []
    counts = Counter()

    if len(model_dirs) != 18:
        errors.append(f"expected 18 models, found {len(model_dirs)}")
    for model_dir in model_dirs:
        path = model_dir / "privacy-detection.scored.json"
        if not path.is_file():
            errors.append(f"{model_dir.name}: missing result")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("per_sample_results", [])
        if len(rows) != 260 or {str(row.get("id")) for row in rows} != set(expected):
            errors.append(f"{model_dir.name}: sample coverage mismatch")
            continue
        all_pred, all_truth = [], []
        for row in rows:
            label = expected[str(row["id"])]
            if row.get("target") != label["phi_list"]:
                errors.append(f"{model_dir.name}/{row['id']}: target mismatch")
            content = row.get("content", {})
            if hashlib.sha256(str(content.get("text", "")).encode()).hexdigest() != EXPECTED_PROMPT_SHA256:
                errors.append(f"{model_dir.name}/{row['id']}: prompt mismatch")
            expected_image = "./data/privacy/P2_phi_detection/GMAI/images/" + label["image"]
            if content.get("image_path") != expected_image or not (ROOT / expected_image).is_file():
                errors.append(f"{model_dir.name}/{row['id']}: image mismatch")
            response = row.get("response")
            if not isinstance(response, str):
                errors.append(f"{model_dir.name}/{row['id']}: non-string response")
                continue
            row_pred, row_truth, detail = evaluate_row(response, label["phi_list"])
            saved = row.get("evals", {})
            comparable = (
                saved.get("raw_pred") == detail["raw_pred"]
                and saved.get("parse_status") == detail["parse_status"]
                and set(saved.get("pred", [])) == set(detail["pred"])
                and set(saved.get("gt", [])) == set(detail["gt"])
                and saved.get("TP") == detail["TP"]
                and saved.get("FP") == detail["FP"]
                and saved.get("FN") == detail["FN"]
                and bool(saved.get("match")) == detail["match"]
            )
            if not comparable:
                errors.append(f"{model_dir.name}/{row['id']}: parser judgment mismatch")
            all_pred.extend(row_pred)
            all_truth.extend(row_truth)
            counts[detail["parse_status"]] += 1
            counts["rows"] += 1
            counts["exact_matches"] += detail["match"]

        tp = sum(truth == 1 and pred == 1 for truth, pred in zip(all_truth, all_pred))
        fp = sum(truth == 0 and pred == 1 for truth, pred in zip(all_truth, all_pred))
        fn = sum(truth == 1 and pred == 0 for truth, pred in zip(all_truth, all_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total = payload.get("total_results", {})
        if total.get("privacy_phi_detection_eval:precision_score") != precision:
            errors.append(f"{model_dir.name}: precision mismatch")
        if total.get("privacy_phi_detection_eval:recall_score") != recall:
            errors.append(f"{model_dir.name}: recall mismatch")
        if total.get("privacy_phi_detection_eval:f1_score") != f1:
            errors.append(f"{model_dir.name}: F1 mismatch")
        summary = total.get("privacy_phi_detection_eval:processed_summary", {})
        if summary.get("processed_preds") != all_pred or summary.get("processed_labels") != all_truth:
            errors.append(f"{model_dir.name}: processed summary mismatch")
        recorded = migration.get("models", {}).get(model_dir.name, {})
        if recorded.get("precision") != precision or recorded.get("recall") != recall or recorded.get("f1") != f1:
            errors.append(f"{model_dir.name}: migration report mismatch")

    report = {
        "schema_version": 1,
        "task_id": "P.2",
        "task_scope": "PHI detection only",
        "cancelled_related_task": "privacy-inflow",
        "status": "passed" if not errors else "failed",
        "model_count": len(model_dirs),
        "response_rows": counts["rows"],
        "exact_match_rows": counts["exact_matches"],
        "parse_statuses": {key: counts[key] for key in ("json_list", "json_object", "embedded_list", "recovered_list", "empty", "invalid", "abstention") if counts[key]},
        "response_sources": migration.get("response_sources"),
        "errors": errors,
    }
    (dataset / "rule_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/privacy/P2_phi_detection/GMAI")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs/privacy/P2")
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.logs.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
