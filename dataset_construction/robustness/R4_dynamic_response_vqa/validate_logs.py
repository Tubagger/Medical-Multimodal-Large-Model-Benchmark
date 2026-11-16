#!/usr/bin/env python3
"""Validate migrated R.4 caches, transformation rules, and parser outputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from migrate_r5_logs import BRANCHES, MODIFIED, load_evaluator_class  # noqa: E402


NEGATION = re.compile(
    r"\b(?:not|cannot|can't|except|least|incorrect|false|absent|without|unlikely|exclude|doesn't|isn't|never)\b",
    re.I,
)


def options_from(text: str) -> dict[str, str]:
    return dict(re.findall(r"(?m)^([A-E])\.\s*([^\n]+)$", text))


def question_from(text: str) -> str | None:
    match = re.search(r"Question:\n(.*?)\nOptions:", text, re.S)
    return match.group(1) if match else None


def saved_list(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def clean_prompt(source: dict, model: str) -> str:
    option_text = "\n".join(f"{key}. {str(value).strip()}" for key, value in sorted(source["options"].items()))
    if model.startswith("llava-med"):
        return (
            "Question:\n" + source["question"] + "\nOptions:\n" + option_text
            + "\nAnswer with exactly one letter: A, B, C, D or E."
        )
    return (
        "Medical Visual Question Answering task.\nQuestion:\n" + source["question"]
        + "\nOptions:\n" + option_text + "\nInstruction:\n"
        + "Select the correct answer based on the image and question.\n"
        + "Respond ONLY with the option letter(s). Do not output explanation."
    )


def validate(dataset: Path, logs: Path) -> dict:
    source_rows = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    source = {str(row["id"]): row for row in source_rows}
    evaluator = load_evaluator_class()("rule_mcq_eval", {})
    valid_options = list("ABCDEFGH")
    migration_path = logs / 'migration_report.json'
    if not migration_path.exists():
        migration_path = ROOT / 'run_state/log_metadata' / logs.resolve().relative_to((ROOT / 'logs').resolve()) / 'migration_report.json'
    migration = json.loads(migration_path.read_text(encoding='utf-8'))
    model_dirs = sorted(path for path in logs.iterdir() if path.is_dir())
    errors: list[str] = []
    counts = Counter()

    for model_dir in model_dirs:
        remain_sets = []
        for branch in BRANCHES:
            path = model_dir / f"robustness-vqa-{branch}.scored.json"
            if not path.is_file():
                errors.append(f"{model_dir.name}/{branch}: missing result")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("per_sample_results", [])
            if len(rows) != 200 or {str(row.get("id")) for row in rows} != set(source):
                errors.append(f"{model_dir.name}/{branch}: sample coverage mismatch")
                continue
            cache = None
            if branch != "clean":
                cache_path = model_dir / f"{branch}-robustness.jsonl"
                records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line]
                if len(records) != 200:
                    errors.append(f"{model_dir.name}/{branch}: cache coverage mismatch")
                cache = {str(record["id"]): record for record in records}

            hits = []
            for index, row in enumerate(rows):
                counts["result_rows"] += 1
                item = source[str(row["id"])]
                text = row["content"]["text"]
                question = question_from(text)
                options = options_from(text)
                expected_image = (
                    "./data/robustness/R4_dynamic_response_vqa/GMAI/images/" + item["image"]
                )
                if row["content"].get("image_path") != expected_image or not (ROOT / expected_image).is_file():
                    errors.append(f"{model_dir.name}/{branch}/{row['id']}: image path mismatch")

                if branch == "clean":
                    if text != clean_prompt(item, model_dir.name) or row.get("target") != item["answer"]:
                        errors.append(f"{model_dir.name}/{branch}/{row['id']}: clean prompt/target mismatch")
                else:
                    record = (cache or {}).get(str(row["id"]))
                    if record is None or record.get("rewritten") != text or record.get("label") != row.get("target"):
                        errors.append(f"{model_dir.name}/{branch}/{row['id']}: cache mismatch")
                    expected_target = (
                        [key for key in item["options"] if key != item["answer"]]
                        if branch == "question-negation" else item["answer"]
                    )
                    if row.get("target") != expected_target:
                        errors.append(f"{model_dir.name}/{branch}/{row['id']}: target-rule mismatch")
                    if branch == "answer-flip":
                        expected = dict(item["options"])
                        expected[item["answer"]] = "None of the options are correct"
                        if question != item["question"] or options != expected:
                            errors.append(f"{model_dir.name}/{branch}/{row['id']}: transformation mismatch")
                    elif branch == "question-negation":
                        if options != item["options"]:
                            errors.append(f"{model_dir.name}/{branch}/{row['id']}: options changed")
                        if question and question.lstrip().startswith(("{", "```")):
                            counts["negation_json_or_fence"] += 1
                        if not question or not NEGATION.search(question):
                            counts["negation_without_explicit_marker"] += 1
                    elif branch == "option-expansion":
                        if question != item["question"] or any(options.get(k) != v for k, v in item["options"].items()):
                            errors.append(f"{model_dir.name}/{branch}/{row['id']}: original content changed")
                        if len(options) != 5 or "E" not in options:
                            errors.append(f"{model_dir.name}/{branch}/{row['id']}: option count mismatch")
                        elif options["E"].strip().casefold() in {v.strip().casefold() for v in item["options"].values()}:
                            counts["expansion_duplicate_observations"] += 1
                    elif branch == "narrative-distraction":
                        if options != item["options"] or not question or not question.startswith(item["question"] + " "):
                            errors.append(f"{model_dir.name}/{branch}/{row['id']}: transformation mismatch")

                pred = evaluator._extract_option(row.get("response"), valid_options)
                true = evaluator._normalize_label(row.get("target"), valid_options)
                match = set(pred or []) == set(true or [])
                saved = row.get("evals", {})
                if pred != saved_list(saved.get("pred_opt")) or true != saved_list(saved.get("true_opt")):
                    errors.append(f"{model_dir.name}/{branch}/{row['id']}: parser output mismatch")
                if match != bool(saved.get("match")):
                    errors.append(f"{model_dir.name}/{branch}/{row['id']}: row score mismatch")
                hits.append(int(match))

            total = payload.get("total_results", {})
            summary = total.get("rule_mcq_eval:processed_summary", {})
            if summary.get("processed_preds") != hits or summary.get("processed_labels") != [1] * 200:
                errors.append(f"{model_dir.name}/{branch}: processed summary mismatch")
            if branch == "clean":
                if total.get("rule_mcq_eval:accuracy_score") != sum(hits) / 200:
                    errors.append(f"{model_dir.name}/{branch}: accuracy mismatch")
            else:
                expected_remain = [index for index, hit in enumerate(hits) if hit]
                metric = total.get("rule_mcq_eval:robustness_acc", {})
                if metric.get("accuracy") != sum(hits) / 200 or metric.get("remain_ids") != expected_remain:
                    errors.append(f"{model_dir.name}/{branch}: robustness summary mismatch")
                remain_sets.append(set(expected_remain))

        combined_path = model_dir / "robustness-vqa.scored.json"
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
        intersection = sorted(set.intersection(*remain_sets)) if remain_sets else []
        metric = combined["total_results"]["rule_mcq_eval:robustness_acc"]
        if metric.get("remain_ids") != intersection or metric.get("accuracy") != len(intersection) / 200:
            errors.append(f"{model_dir.name}: four-branch intersection mismatch")

    report = {
        "schema_version": 1,
        "task_id": "R.4",
        "status": "passed_with_observations" if not errors else "failed",
        "validation_scope": [
            "verified image paths and payload identity",
            "rewrite-cache alignment",
            "four transformation rules",
            "current rule parser and row/summary consistency",
        ],
        "model_count": len(model_dirs),
        "result_rows": counts["result_rows"],
        "cache_rows": len(model_dirs) * 200 * len(MODIFIED),
        "question_negation_answer_type": "multi-label set",
        "parser_matching_rule": "set equality",
        "historical_parser_recheck": {
            "prediction_parse_changes": migration["prediction_parse_changes"],
            "target_parse_changes": migration["target_parse_changes"],
            "row_score_changes": migration["row_score_changes"],
            "models_with_score_changes": migration["models_with_score_changes"],
            "aggregate_intersection_changes": migration["aggregate_intersection_changes"],
        },
        "observations_not_treated_as_protocol_failures": {
            "negation_json_or_code_fence_instances": counts["negation_json_or_fence"],
            "negation_without_explicit_keyword_instances": counts["negation_without_explicit_marker"],
            "option_expansion_duplicate_instances": counts["expansion_duplicate_observations"],
            "missing_saved_response_fields": migration["missing_saved_responses"],
        },
        "semantic_or_clinical_review_performed": False,
        "errors": errors,
    }
    (dataset / "rule_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/robustness/R4_dynamic_response_vqa/GMAI")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs/robustness/R4")
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.logs.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
