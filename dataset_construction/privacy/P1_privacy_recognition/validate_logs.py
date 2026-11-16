#!/usr/bin/env python3
"""Validate migrated P.1 responses, parser decisions, and summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ours.utils.p1_parser import parse_p1_class  # noqa: E402
from scripts.migrate_p1_logs import EXPECTED_PROMPT_SHA256, macro_f1  # noqa: E402
import hashlib  # noqa: E402


def validate(dataset: Path, logs: Path) -> dict:
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    expected = {str(row["id"]): row for row in labels}
    migration_path = logs / 'migration_report.json'
    if not migration_path.exists():
        migration_path = ROOT / 'run_state/log_metadata' / logs.resolve().relative_to((ROOT / 'logs').resolve()) / 'migration_report.json'
    migration = json.loads(migration_path.read_text(encoding='utf-8'))
    model_dirs = sorted(path for path in logs.iterdir() if path.is_dir())
    errors: list[str] = []
    total_rows = total_matches = 0

    if len(model_dirs) != 18:
        errors.append(f"expected 18 model directories, found {len(model_dirs)}")
    for model_dir in model_dirs:
        path = model_dir / "privacy-recognition.scored.json"
        if not path.is_file():
            errors.append(f"{model_dir.name}: missing result file")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("per_sample_results", [])
        if len(rows) != 260 or {str(row.get("id")) for row in rows} != set(expected):
            errors.append(f"{model_dir.name}: sample coverage mismatch")
            continue
        truth, predicted = [], []
        for row in rows:
            label = expected[str(row["id"])]
            target = int(label["label"])
            if int(row.get("target")) != target:
                errors.append(f"{model_dir.name}/{row['id']}: target mismatch")
            content = row.get("content", {})
            if hashlib.sha256(str(content.get("text", "")).encode()).hexdigest() != EXPECTED_PROMPT_SHA256:
                errors.append(f"{model_dir.name}/{row['id']}: prompt mismatch")
            expected_image = "./data/privacy/P1_privacy_recognition/GMAI/images/" + label["image"]
            if content.get("image_path") != expected_image or not (ROOT / expected_image).is_file():
                errors.append(f"{model_dir.name}/{row['id']}: image path mismatch")
            parsed = parse_p1_class(row.get("response"))
            if parsed is None:
                errors.append(f"{model_dir.name}/{row['id']}: unparseable response")
                continue
            saved = row.get("evals", {})
            if saved.get("pred") != str(parsed) or saved.get("gt") != str(target):
                errors.append(f"{model_dir.name}/{row['id']}: parsed class mismatch")
            match = parsed == target
            if bool(saved.get("match")) != match:
                errors.append(f"{model_dir.name}/{row['id']}: row score mismatch")
            truth.append(target)
            predicted.append(parsed)
            total_rows += 1
            total_matches += match
        total = payload.get("total_results", {})
        accuracy = sum(a == b for a, b in zip(truth, predicted)) / 260
        f1 = macro_f1(truth, predicted)
        if total.get("privacy-recognition-eval:accuracy_score") != accuracy:
            errors.append(f"{model_dir.name}: accuracy mismatch")
        if total.get("privacy-recognition-eval:macro-f1") != f1:
            errors.append(f"{model_dir.name}: macro-F1 mismatch")
        summary = total.get("privacy-recognition-eval:processed_summary", {})
        if summary.get("processed_preds") != predicted or summary.get("processed_labels") != truth:
            errors.append(f"{model_dir.name}: processed summary mismatch")
        recorded = migration.get("models", {}).get(model_dir.name, {})
        if recorded.get("accuracy") != accuracy or recorded.get("macro_f1") != f1:
            errors.append(f"{model_dir.name}: migration report mismatch")

    report = {
        "schema_version": 1,
        "task_id": "P.1",
        "status": "passed" if not errors else "failed",
        "model_count": len(model_dirs),
        "response_rows": total_rows,
        "correct_rows": total_matches,
        "source_counts": migration.get("source_counts"),
        "parser": "ours.utils.p1_parser.parse_p1_class",
        "unparseable_responses": sum("unparseable response" in error for error in errors),
        "errors": errors,
    }
    (dataset / "rule_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/privacy/P1_privacy_recognition/GMAI")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs/privacy/P1")
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.logs.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
