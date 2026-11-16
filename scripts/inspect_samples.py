#!/usr/bin/env python3
"""Build and serve a local browser for inspecting saved model results.

The viewer indexes the question, image path, answer, and score records under
``logs/``. Dataset labels under ``data/`` are attached only as auxiliary
metadata. It groups all conditions of one sample (for example R4
clean/negation/attack branches) under the same task/subset/id key, then shows
every model answer and the exact image supplied to the model.

Usage:
    python scripts/inspect_samples.py
    python scripts/inspect_samples.py --rebuild
    python scripts/inspect_samples.py --build-only
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import sqlite3
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, urlparse


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
DEFAULT_DB = ROOT / "run_state" / "sample_inspector.sqlite3"
SCHEMA_VERSION = 1

TASK_ORDER = [
    "C1", "C2", "C3",
    "R1", "R2", "R3", "R4",
    "P1", "P2", "P3", "P4",
    "S1", "S2", "S3",
    "F1", "F2",
]

TASK_INFO = {
    "C1": ("C.1", "Capability", "Abnormality Recognition"),
    "C2": ("C.2", "Capability", "Lesion Localization"),
    "C3": ("C.3", "Capability", "Medical Reasoning"),
    "R1": ("R.1", "Robustness", "OOD Medical VQA"),
    "R2": ("R.2", "Robustness", "Untargeted Adversarial Attack"),
    "R3": ("R.3", "Robustness", "Targeted Adversarial Attack"),
    "R4": ("R.4", "Robustness", "Dynamic Response Robustness"),
    "P1": ("P.1", "Privacy", "Privacy Recognition"),
    "P2": ("P.2", "Privacy", "PHI Detection"),
    "P3": ("P.3", "Privacy", "Privacy Inference"),
    "P4": ("P.4", "Privacy", "Privacy-Modified VQA"),
    "S1": ("S.1", "Safety", "Medical Risk Requests"),
    "S2": ("S.2", "Safety", "MCM Jailbreak"),
    "S3": ("S.3", "Safety", "Dynamic Safety VQA"),
    "F1": ("F.1", "Fairness", "Visual Preference"),
    "F2": ("F.2", "Fairness", "Response-Adaptive VQA"),
}

C1_SUBSETS = {
    "anomaly-detection-brain": ("MRI", "Brain MRI"),
    "anomaly-detection-chexpert": ("CheXpert", "Chest X-ray"),
    "anomaly-detection-isic": ("ISIC", "Dermoscopy"),
    "anomaly-detection-mura": ("MURA", "Musculoskeletal X-ray"),
    "anomaly-detection-oct": ("OCT", "Retinal OCT"),
}

TASK_CONDITIONS = {
    "C1": [
        ("anomaly-detection-brain", "Brain MRI"),
        ("anomaly-detection-chexpert", "Chest X-ray"),
        ("anomaly-detection-isic", "Dermoscopy"),
        ("anomaly-detection-mura", "Musculoskeletal X-ray"),
        ("anomaly-detection-oct", "Retinal OCT"),
    ],
    "C2": [("lesion-localization", "Lesion localization")],
    "C3": [("logical-reasoning", "Medical reasoning")],
    "R1": [("ood", "OOD medical VQA")],
    "R2": [
        ("adv-untarget-clean", "Clean image"),
        ("adv-untarget", "Untargeted attack"),
    ],
    "R3": [
        ("adv-target-clean", "Clean image"),
        ("adv-target", "Targeted attack"),
    ],
    "R4": [
        ("robustness-vqa-clean", "Clean"),
        ("robustness-vqa-answer-flip", "Answer flip"),
        ("robustness-vqa-narrative-distraction", "Narrative distraction"),
        ("robustness-vqa-option-expansion", "Option expansion"),
        ("robustness-vqa-question-negation", "Question negation"),
    ],
    "P1": [("privacy-recognition", "Privacy recognition")],
    "P2": [("privacy-detection", "PHI detection")],
    "P3": [("privacy-inference", "Privacy inference")],
    "P4": [
        ("privacy-vqa-focus-distraction", "Focus distraction"),
        ("privacy-vqa-subtle-request", "Subtle request"),
        ("privacy-vqa-trap-warning", "Trap warning"),
        ("privacy-vqa-well-intention", "Well intention"),
    ],
    "S1": [("safety-risk", "Medical risk request")],
    "S2": [("mcm-jailbreak", "MCM jailbreak")],
    "S3": [
        ("safety-vqa-clean", "Clean"),
        ("safety-vqa-authority", "Authority"),
        ("safety-vqa-consequence-obfuscation", "Consequence obfuscation"),
        ("safety-vqa-diagnostic-anchoring", "Diagnostic anchoring"),
        ("safety-vqa-risk-minimization", "Risk minimization"),
        ("safety-vqa-urgency", "Urgency"),
    ],
    "F1": [("visual-preference", "Paired visual preference")],
    "F2": [
        ("bias-ref", "Baseline"),
        ("bias-vqa-race", "Race"),
        ("bias-vqa-language", "Language"),
        ("bias-vqa-emotion", "Emotion"),
        ("bias-vqa-cognitive", "Cognitive"),
        ("bias-vqa-text-race", "Text: race"),
        ("bias-vqa-text-language", "Text: language"),
        ("bias-vqa-text-emotion", "Text: emotion"),
        ("bias-vqa-text-cognitive", "Text: cognitive"),
    ],
}

LABEL_FILES = {
    "C1": {
        "MRI": ROOT / "data/capability/C1_abnormality_detection/MRI/labels.json",
        "CheXpert": ROOT / "data/capability/C1_abnormality_detection/CheXpert/labels.json",
        "ISIC": ROOT / "data/capability/C1_abnormality_detection/ISIC/labels.json",
        "MURA": ROOT / "data/capability/C1_abnormality_detection/MURA/labels.json",
        "OCT": ROOT / "data/capability/C1_abnormality_detection/OCT/labels.json",
    },
    "C2": {"": ROOT / "data/capability/C2_lesion_localization/DeepLesion/labels.json"},
    "C3": {"": ROOT / "data/capability/C3_medical_reasoning/GMAI/labels.json"},
    "R1": {"": ROOT / "data/robustness/R1_ood/GMAI/labels.json"},
    "R2": {"": ROOT / "data/robustness/R2_untargeted_attack/GMAI/labels.json"},
    "R3": {"": ROOT / "data/robustness/R3_targeted_attack/GMAI/labels.json"},
    "R4": {"": ROOT / "data/robustness/R4_dynamic_response_vqa/GMAI/labels.json"},
    "P1": {"": ROOT / "data/privacy/P1_privacy_recognition/GMAI/labels.json"},
    "P2": {"": ROOT / "data/privacy/P2_phi_detection/GMAI/labels.json"},
    "P3": {"": ROOT / "data/privacy/P3_privacy_inference/labels.jsonl"},
    "P4": {"": ROOT / "data/privacy/P4_privacy_modified_vqa/GMAI/labels.json"},
    "S1": {"": ROOT / "data/safety/S1_medical_risk_requests/GMAI/labels.jsonl"},
    "S2": {"": ROOT / "data/safety/S2_mcm_jailbreak/GMAI/labels.json"},
    "S3": {"": ROOT / "data/safety/S3_dynamic_safety_vqa/GMAI/labels.jsonl"},
    "F1": {"": ROOT / "data/fairness/F1_visual_preference/labels.json"},
    "F2": {"": ROOT / "data/fairness/F2_response_adaptive_vqa/bias_vqa.json"},
}

MODEL_PRIORITY = [
    "gpt-5.4",
    "deepseek-vl2",
    "deepseek-vl2-small",
    "gemini-3.5-flash",
    "glm-4v-9b",
    "healthgpt-pro-8b",
    "huatuogpt-v-34b",
    "huatuogpt-v-7b",
    "hulu-med-32b",
    "hulu-med-7b",
    "internvl3.5-14b",
    "internvl3.5-38b",
    "lingshu-32b",
    "lingshu-7b",
    "llava-med-v1.5-mistral-7b",
    "med-gemma-27b-it",
    "med-gemma-4b-it",
    "qwen3-vl-32b-instruct",
]

PATH_KEYS_TO_HIDE = {
    "image",
    "image_path",
    "source_image",
    "original_image",
    "original_path",
    "source_left",
    "source_right",
}


def make_base_key(task: str, subset: str, sample_id: Any) -> str:
    if subset:
        return f"{task}::{subset}::{sample_id}"
    return f"{task}::{sample_id}"


def make_sample_key(base_key: str, condition: str) -> str:
    return f"{base_key}::{condition}"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", value).strip().lower()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def text_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def clean_image_path(value: Any) -> str:
    if not value:
        return ""
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def condition_meta(task: str, condition: str) -> tuple[str, int]:
    entries = TASK_CONDITIONS.get(task, [])
    for index, (name, label) in enumerate(entries, start=1):
        if name == condition:
            return label, index
    return condition, len(entries) + 1


def task_meta(task: str) -> tuple[str, str, str]:
    return TASK_INFO.get(task, (task, "Other", task))


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "samples", "records", "cases", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            records = [payload]
    else:
        records = []
    for item in records:
        if isinstance(item, dict):
            yield item


def label_question(record: dict[str, Any]) -> str:
    for key in ("question", "malicious_prompt", "attack_prompt", "patient_self-report"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def label_search_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "question",
        "malicious_prompt",
        "attack_prompt",
        "category",
        "malicious_category",
        "attribute_name",
        "class",
        "label",
        "phi_list",
        "persona",
    ):
        value = record.get(key)
        if value is not None:
            parts.append(text_value(value))
    return normalize_text(" ".join(parts))


def load_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for task, subsets in LABEL_FILES.items():
        for subset, path in subsets.items():
            if not path.exists():
                continue
            try:
                for record in iter_records(path):
                    sample_id = record.get("id", record.get("case_id"))
                    if sample_id is None:
                        continue
                    labels[make_base_key(task, subset, sample_id)] = record
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[warn] cannot read labels {path}: {exc}")
    return labels


def load_scored_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] cannot read scores {path}: {exc}")
        return {}

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("per_sample_results")
        if not isinstance(records, list):
            records = payload.get("results")
    else:
        records = None
    if not isinstance(records, list):
        return {}

    result: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        sample_id = record.get("id")
        if sample_id is not None and record.get("evals") is not None:
            result[str(sample_id)] = record["evals"]
    return result


def compact_eval(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: compact_eval(item)
            for key, item in value.items()
            if key not in {"attempts", "usage", "completion_id", "served_model", "cost"}
        }
    if isinstance(value, list):
        return [compact_eval(item) for item in value]
    return value


def summarize_eval(evals: Any) -> tuple[str, str, str]:
    if not isinstance(evals, dict) or not evals:
        return "none", "", ""

    local = evals.get("local_judges")
    if isinstance(local, dict):
        judgments = local.get("judgments")
        if isinstance(judgments, dict):
            parts: list[str] = []
            reasons: list[str] = []
            has_value = False
            risky = False
            for group, values in judgments.items():
                if not isinstance(values, dict):
                    continue
                for key, value in values.items():
                    if key == "reason":
                        if isinstance(value, str) and value.strip():
                            reasons.append(value.strip())
                        continue
                    if isinstance(value, bool):
                        number = int(value)
                    elif isinstance(value, (int, float)):
                        number = value
                    else:
                        continue
                    has_value = True
                    parts.append(f"{group}.{key}={number}")
                    if group == "privacy" and key == "violation" and number > 0:
                        risky = True
                    if group == "safety" and key == "harmful_compliance" and number > 0:
                        risky = True
                    if group == "response" and key == "appropriateness" and number == 0:
                        risky = True
            if has_value:
                return (
                    "incorrect" if risky else "correct",
                    " · ".join(parts),
                    " | ".join(dict.fromkeys(reasons)),
                )
        return "judged", str(local.get("status", "judged")), ""

    if "match" in evals:
        matched = bool(evals.get("match"))
        parts = ["正确" if matched else "错误"]
        if isinstance(evals.get("iou"), (int, float)):
            parts.append(f"IoU={evals['iou']:.3f}")
        if evals.get("parse_error"):
            parts.append("解析失败")
        return "correct" if matched else "incorrect", " · ".join(parts), ""

    if "TP" in evals or "FP" in evals or "FN" in evals:
        matched = bool(evals.get("match"))
        summary = (
            f"TP={evals.get('TP', '—')} · FP={evals.get('FP', '—')} · "
            f"FN={evals.get('FN', '—')}"
        )
        return "correct" if matched else "incorrect", summary, ""

    if "neutral" in evals or "valid" in evals:
        parts = []
        for key in ("valid", "neutral", "parse_error"):
            if key in evals:
                parts.append(f"{key}={evals[key]}")
        return "judged", " · ".join(parts) or "judged", ""

    summary = json.dumps(compact_eval(evals), ensure_ascii=False)
    if len(summary) > 240:
        summary = summary[:237] + "..."
    return "judged", summary, ""


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE samples (
            sample_key TEXT PRIMARY KEY,
            base_key TEXT NOT NULL,
            task TEXT NOT NULL,
            task_code TEXT NOT NULL,
            task_name TEXT NOT NULL,
            dimension TEXT NOT NULL,
            subset TEXT NOT NULL,
            subset_label TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            condition TEXT NOT NULL,
            condition_label TEXT NOT NULL,
            condition_order INTEGER NOT NULL,
            question TEXT NOT NULL,
            label_question TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            target_json TEXT NOT NULL,
            label_json TEXT NOT NULL,
            response_count INTEGER NOT NULL,
            model_count INTEGER NOT NULL,
            search_text TEXT NOT NULL
        );
        CREATE TABLE responses (
            sample_key TEXT NOT NULL,
            model TEXT NOT NULL,
            response TEXT NOT NULL,
            image_path TEXT NOT NULL,
            target_json TEXT NOT NULL,
            eval_json TEXT NOT NULL,
            eval_status TEXT NOT NULL,
            eval_summary TEXT NOT NULL,
            eval_reason TEXT NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (sample_key, model)
        );
        """
    )


def build_index(db_path: Path) -> None:
    started = time.time()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    labels = load_labels()
    print(f"Loaded {len(labels)} label records from {len(LABEL_FILES)} tasks.")

    conn = sqlite3.connect(tmp_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    create_schema(conn)

    sample_state: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    response_count = 0
    parse_errors = 0
    response_files = sorted(LOG_ROOT.glob("*/*/*/*.responses.jsonl"))

    for file_index, response_path in enumerate(response_files, start=1):
        parts = response_path.relative_to(LOG_ROOT).parts
        if len(parts) != 4:
            continue
        dimension, task, model, filename = parts
        condition = filename.removesuffix(".responses.jsonl")
        task_code, task_dimension, task_name = task_meta(task)
        task_dimension = dimension.capitalize()
        condition_label, condition_order = condition_meta(task, condition)
        subset, subset_label = C1_SUBSETS.get(condition, ("", "")) if task == "C1" else ("", "")
        scored_path = response_path.with_name(
            filename.removesuffix(".responses.jsonl") + ".scored.json"
        )
        scored = load_scored_map(scored_path)

        with response_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(record, dict) or record.get("id") is None:
                    continue

                sample_id = record["id"]
                base_key = make_base_key(task, subset, sample_id)
                sample_key = make_sample_key(base_key, condition)
                content = record.get("content")
                if isinstance(content, str):
                    content = {"text": content}
                if not isinstance(content, dict):
                    content = {}
                prompt = content.get("text") or ""
                image_path = clean_image_path(content.get("image_path") or content.get("image"))
                target = record.get("target")
                response = record.get("response")
                if response is None and task == "F2" and condition == "bias-ref":
                    response = record.get("major_answer")
                evals = scored.get(str(sample_id))
                eval_status, eval_summary, eval_reason = summarize_eval(evals)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO responses
                    (sample_key, model, response, image_path, target_json, eval_json,
                     eval_status, eval_summary, eval_reason, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_key,
                        model,
                        "" if response is None else text_value(response),
                        image_path,
                        json_text(target),
                        json_text(compact_eval(evals)) if evals is not None else "",
                        eval_status,
                        eval_summary,
                        eval_reason,
                        str(response_path.relative_to(ROOT)),
                    ),
                )
                response_count += 1
                models.add(model)

                state = sample_state.get(sample_key)
                if state is None:
                    state = {
                        "base_key": base_key,
                        "task": task,
                        "task_code": task_code,
                        "task_name": task_name,
                        "dimension": task_dimension,
                        "subset": subset,
                        "subset_label": subset_label,
                        "sample_id": str(sample_id),
                        "condition": condition,
                        "condition_label": condition_label,
                        "condition_order": condition_order,
                        "prompt": prompt,
                        "images": set(),
                        "targets": set(),
                        "models": set(),
                    }
                    sample_state[sample_key] = state
                if not state["prompt"] and prompt:
                    state["prompt"] = prompt
                if image_path:
                    state["images"].add(image_path)
                if target is not None:
                    state["targets"].add(json_text(target))
                state["models"].add(model)

        if file_index % 50 == 0 or file_index == len(response_files):
            print(
                f"Indexed {file_index}/{len(response_files)} response files, "
                f"{len(sample_state)} condition samples, {response_count} answers."
            )

    for sample_key, state in sample_state.items():
        label = labels.get(state["base_key"], {})
        label_q = label_question(label)
        question = state["prompt"] or label_q
        target_values = [json.loads(item) for item in sorted(state["targets"])]
        if not target_values:
            target_json = "null"
        elif len(target_values) == 1:
            target_json = json_text(target_values[0])
        else:
            target_json = json_text(target_values)
        images = sorted(state["images"])
        search_text = normalize_text(
            " ".join(
                [
                    question,
                    state["task"],
                    state["task_code"],
                    state["task_name"],
                    state["subset"],
                    state["subset_label"],
                    state["condition_label"],
                    state["sample_id"],
                ]
            )
        )
        conn.execute(
            """
            INSERT INTO samples
            (sample_key, base_key, task, task_code, task_name, dimension, subset,
             subset_label, sample_id, condition, condition_label, condition_order,
             question, label_question, image_paths, target_json, label_json,
             response_count, model_count, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_key,
                state["base_key"],
                state["task"],
                state["task_code"],
                state["task_name"],
                state["dimension"],
                state["subset"],
                state["subset_label"],
                state["sample_id"],
                state["condition"],
                state["condition_label"],
                state["condition_order"],
                question,
                label_q,
                json_text(images),
                target_json,
                json_text(label),
                len(state["models"]),
                len(state["models"]),
                search_text,
            ),
        )

    conn.executescript(
        """
        CREATE INDEX idx_samples_base ON samples(base_key);
        CREATE INDEX idx_samples_task ON samples(task);
        CREATE INDEX idx_responses_sample ON responses(sample_key);
        """
    )

    model_order = sorted(models, key=model_sort_key)
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_samples": str(len({state["base_key"] for state in sample_state.values()})),
        "condition_samples": str(len(sample_state)),
        "responses": str(response_count),
        "models": json_text(model_order),
    }
    conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", meta.items())
    conn.commit()
    conn.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.replace(db_path)

    elapsed = time.time() - started
    print(
        f"Built {db_path.relative_to(ROOT)}: {meta['base_samples']} base samples, "
        f"{meta['condition_samples']} condition samples, {meta['responses']} answers, "
        f"{len(model_order)} models in {elapsed:.1f}s."
    )
    if parse_errors:
        print(f"Skipped {parse_errors} malformed response lines.")


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return MODEL_PRIORITY.index(model), model
    except ValueError:
        return len(MODEL_PRIORITY), model


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_index(db_path: Path, rebuild: bool) -> None:
    if not rebuild and db_path.exists():
        try:
            conn = open_db(db_path)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            conn.close()
            if row and row["value"] == str(SCHEMA_VERSION):
                return
        except sqlite3.Error:
            pass
    build_index(db_path)


def get_models(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute("SELECT value FROM meta WHERE key='models'").fetchone()
    if row:
        try:
            models = json.loads(row["value"])
            if isinstance(models, list):
                return [str(model) for model in models]
        except json.JSONDecodeError:
            pass
    rows = conn.execute("SELECT DISTINCT model FROM responses").fetchall()
    return sorted((row["model"] for row in rows), key=model_sort_key)


def get_stats(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def escape(value: Any) -> str:
    return html.escape(text_value(value), quote=True)


def shorten(value: Any, limit: int) -> str:
    text = text_value(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def detail_url(base_key: str) -> str:
    return "/sample?key=" + quote(base_key, safe="")


def image_url(path: str) -> str:
    return "/image?path=" + quote(path, safe="")


def task_display(row: sqlite3.Row | dict[str, Any]) -> str:
    parts = [f"{row['task_code']} {row['task_name']}"]
    if row["subset_label"]:
        parts.append(row["subset_label"])
    return " · ".join(parts)


def search_samples(
    conn: sqlite3.Connection, query: str, task: str, limit: int = 100
) -> tuple[list[dict[str, Any]], bool]:
    clauses: list[str] = []
    params: list[Any] = []
    raw = query.strip()
    task = task.strip().upper()

    if task:
        clauses.append("UPPER(s.task) = ?")
        params.append(task)

    if raw:
        id_match = re.fullmatch(r"(?:id:|#)\s*(\S+)", raw, flags=re.IGNORECASE)
        base_match = re.fullmatch(
            r"(C\d|R\d|P\d|S\d|F\d):([A-Za-z0-9_-]+):(\S+)", raw, flags=re.IGNORECASE
        )
        task_id_match = re.fullmatch(
            r"(C\d|R\d|P\d|S\d|F\d):(\S+)", raw, flags=re.IGNORECASE
        )
        bare_id_match = re.fullmatch(r"\d+", raw)
        if base_match:
            task_name = base_match.group(1).upper()
            clauses.extend(
                ["UPPER(s.task) = ?", "LOWER(s.subset) = LOWER(?)", "s.sample_id = ?"]
            )
            params.extend([task_name, base_match.group(2), base_match.group(3)])
        elif id_match:
            clauses.append("s.sample_id = ?")
            params.append(id_match.group(1))
        elif bare_id_match:
            clauses.append("s.sample_id = ?")
            params.append(raw)
        elif task_id_match:
            clauses.extend(["UPPER(s.task) = ?", "s.sample_id = ?"])
            params.extend([task_id_match.group(1).upper(), task_id_match.group(2)])
        else:
            for token in raw.lower().split():
                clauses.append("instr(s.search_text, ?) > 0")
                params.append(token)

    where = " AND ".join(clauses) if clauses else "1=1"
    rows = conn.execute(
        f"""
        SELECT s.base_key,
               MIN(s.task) AS task,
               MIN(s.task_code) AS task_code,
               MIN(s.task_name) AS task_name,
               MIN(s.dimension) AS dimension,
               MIN(s.subset) AS subset,
               MIN(s.subset_label) AS subset_label,
               MIN(s.sample_id) AS sample_id,
               COUNT(DISTINCT s.condition) AS condition_count,
               MIN(s.question) AS question,
               SUM(s.response_count) AS answer_count,
               (SELECT s2.image_paths FROM samples s2
                WHERE s2.base_key = s.base_key
                ORDER BY s2.condition_order LIMIT 1) AS image_paths,
               (SELECT s2.target_json FROM samples s2
                WHERE s2.base_key = s.base_key
                ORDER BY s2.condition_order LIMIT 1) AS target_json
        FROM samples s
        WHERE {where}
        GROUP BY s.base_key
        ORDER BY s.base_key
        LIMIT 2000
        """,
        params,
    ).fetchall()

    phrase = (
        normalize_text(raw)
        if raw and not id_match and not base_match and not task_id_match and not bare_id_match
        else ""
    )

    def sort_key(row: sqlite3.Row) -> tuple[int, int, int, str, int, str]:
        task_rank = TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99
        exact_rank = 0 if phrase and phrase in normalize_text(row["question"]) else 1
        question_length = len(normalize_text(row["question"]))
        try:
            number = int(str(row["sample_id"]))
        except ValueError:
            number = 10**9
        return exact_rank, task_rank, question_length, row["subset"], number, str(row["sample_id"])

    ordered = sorted(rows, key=sort_key)
    more = len(ordered) > limit
    return [dict(row) for row in ordered[:limit]], more


def base_row_sort_key(row: sqlite3.Row | dict[str, Any]) -> tuple[int, int, int, str]:
    c1_subset_rank = {
        subset: index for index, (subset, _) in enumerate(C1_SUBSETS.values())
    }
    task_rank = TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99
    subset_rank = c1_subset_rank.get(row["subset"], 0) if row["task"] == "C1" else 0
    try:
        number = int(str(row["sample_id"]))
    except ValueError:
        number = 10**9
    return task_rank, subset_rank, number, str(row["sample_id"])


def list_url(task: str, subset: str = "") -> str:
    url = "/list?task=" + quote(task)
    if subset:
        url += "&subset=" + quote(subset)
    return url


def get_task_groups(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.task,
               MIN(s.task_code) AS task_code,
               MIN(s.task_name) AS task_name,
               MIN(s.dimension) AS dimension,
               s.subset,
               MIN(s.subset_label) AS subset_label,
               COUNT(DISTINCT s.base_key) AS sample_count,
               COUNT(*) AS condition_count,
               SUM(s.response_count) AS answer_count,
               (SELECT s2.image_paths FROM samples s2
                WHERE s2.task = s.task AND s2.subset = s.subset
                ORDER BY CAST(s2.sample_id AS INTEGER), s2.condition_order
                LIMIT 1) AS image_paths
        FROM samples s
        GROUP BY s.task, s.subset
        """
    ).fetchall()
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: base_row_sort_key({**row, "sample_id": 0}),
    )


def list_task_samples(conn: sqlite3.Connection, task: str, subset: str = "") -> list[dict[str, Any]]:
    task = task.strip().upper()
    subset = subset.strip()
    clauses = ["UPPER(s.task) = ?"]
    params: list[Any] = [task]
    if subset:
        clauses.append("LOWER(s.subset) = LOWER(?)")
        params.append(subset)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""
        SELECT s.base_key,
               MIN(s.task) AS task,
               MIN(s.task_code) AS task_code,
               MIN(s.task_name) AS task_name,
               MIN(s.dimension) AS dimension,
               MIN(s.subset) AS subset,
               MIN(s.subset_label) AS subset_label,
               MIN(s.sample_id) AS sample_id,
               COUNT(DISTINCT s.condition) AS condition_count,
               MIN(s.question) AS question,
               SUM(s.response_count) AS answer_count,
               (SELECT s2.image_paths FROM samples s2
                WHERE s2.base_key = s.base_key
                ORDER BY s2.condition_order LIMIT 1) AS image_paths,
               (SELECT s2.target_json FROM samples s2
                WHERE s2.base_key = s.base_key
                ORDER BY s2.condition_order LIMIT 1) AS target_json
        FROM samples s
        WHERE {where}
        GROUP BY s.base_key
        """,
        params,
    ).fetchall()
    return [dict(row) for row in sorted(rows, key=base_row_sort_key)]


def format_target(task: str, target: Any) -> str:
    if target is not None:
        return text_value(target)
    if task in {"P3", "P4", "S1", "S2", "S3"}:
        return "无固定标准答案（按评分标准判定）"
    if task == "F2":
        return "该参考分支无固定答案（以模型自身基线为参照）"
    return "未记录标准答案"


def card_thumbnail(image_paths: str, href: str) -> str:
    images = parse_json(image_paths, [])
    if not isinstance(images, list) or not images:
        return '<div class="card-thumb no-image">无图</div>'
    path = str(images[0])
    return (
        f'<a class="card-thumb" href="{escape(href)}">'
        f'<img src="{escape(image_url(path))}" alt="{escape(path)}" loading="lazy">'
        f'</a>'
    )


def render_cards(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        key = row["base_key"]
        url = detail_url(key)
        label = task_display(row)
        condition_text = (
            f"{row['condition_count']} 个条件"
            if row["condition_count"] > 1
            else "单条件"
        )
        target = parse_json(row.get("target_json", "null"), None)
        target_text = format_target(row["task"], target)
        answer_class = "no-answer" if target is None else ""
        cards.append(
            f'<article class="result-card">'
            f'{card_thumbnail(row.get("image_paths", "[]"), url)}'
            f'<div class="result-main">'
            f'<div class="result-head"><a href="{escape(url)}">'
            f'<span class="task-code">{escape(row["task_code"])}</span> '
            f'{escape(row["task_name"])}</a>'
            f'<span class="id-badge">ID {escape(row["sample_id"])}</span></div>'
            f'<div class="result-meta">{escape(label)} · {condition_text} · '
            f'{escape(row["answer_count"])} 条回答</div>'
            f'<div class="answer-line {answer_class}"><span class="answer-badge">标准答案</span> '
            f'<span class="answer-value">{escape(target_text)}</span></div>'
            f'<div class="question">{escape(shorten(row["question"], 420))}</div>'
            f'<a class="open-link" href="{escape(url)}">查看图片与全部模型回答 →</a>'
            f'</div>'
            f'</article>'
        )
    return "".join(cards)


def render_index(conn: sqlite3.Connection, query: str, task: str, limit: int) -> bytes:
    task = task.strip().upper()
    stats = get_stats(conn)
    results, more = search_samples(conn, query, task, limit=limit)
    models = get_models(conn)
    counts = {
        row["task"]: row["count"]
        for row in conn.execute(
            "SELECT task, COUNT(DISTINCT base_key) AS count FROM samples GROUP BY task"
        ).fetchall()
    }

    options = []
    for code in TASK_ORDER:
        task_code, _, task_name = TASK_INFO[code]
        count = counts.get(code, 0)
        selected = " selected" if code == task else ""
        options.append(
            f'<option value="{escape(code)}"{selected}>'
            f'{escape(task_code)} {escape(task_name)} ({count})</option>'
        )

    summary = "".join(
        f'<a class="task-chip" href="{escape(list_url(code))}">'
        f'<b>{escape(TASK_INFO[code][0])}</b> {escape(TASK_INFO[code][2])}'
        f'<span>{counts.get(code, 0)}</span></a>'
        for code in TASK_ORDER
    )

    if query:
        if results:
            result_html = render_cards(results)
            if more:
                result_html += '<p class="note">结果较多，仅显示前 100 条。请输入更具体的关键词。</p>'
        else:
            result_html = (
                '<div class="empty">没有匹配样本。可尝试更短的关键词，'
                '或使用 <code>id:12</code> / <code>C1:MRI:0</code> 精确查找。</div>'
            )
    else:
        result_html = (
            '<div class="empty">输入题干中的一段英文关键词开始查找。'
            '例如 <code>chest pain</code>、<code>brain MRI</code> 或 <code>privacy</code>。</div>'
        )

    body = f"""
    <header class="topbar"><div class="wrap">
      <div class="eyebrow">MedMMLM-Trust · sample inspector</div>
      <h1>示例内容检索</h1>
      <p class="subtitle">题干、图片、模型回答和评分均读取自 <code>logs/</code>；按题干定位样本后核对全部分支。</p>
      <form method="get" action="/" class="search-form">
        <input name="q" value="{escape(query)}" autofocus
               placeholder="输入题干关键词，例如 chest pain / brain MRI / privacy">
        <select name="task">{''.join(options)}</select>
        <button type="submit">搜索</button>
      </form>
      <div class="stats">{escape(stats.get('base_samples', '?'))} 个基础样本 ·
        {escape(stats.get('condition_samples', '?'))} 个条件样本 ·
        {escape(stats.get('responses', '?'))} 条模型回答 ·
        {len(models)} 个模型</div>
      <a class="task-entry" href="/tasks">按任务浏览全部样本 →</a>
      <div class="hint">题干检索基于 logs 的 <code>content.text</code>。直接输入 <code>12</code> 精确查 ID；任务内 ID：<code>R4:0</code>；
        C1 子集：<code>C1:MRI:0</code>。数据更新后运行 <code>--rebuild</code>。</div>
    </div></header>
    <main class="wrap">
      <section class="task-chips">{summary}</section>
      <section class="results">{result_html}</section>
    </main>
    """
    return page("MedMMLM-Trust 示例检索", body).encode("utf-8")


def render_tasks(conn: sqlite3.Connection) -> bytes:
    groups = get_task_groups(conn)
    sample_count = sum(group["sample_count"] for group in groups)
    condition_count = sum(group["condition_count"] for group in groups)
    cards = []
    for group in groups:
        url = list_url(group["task"], group["subset"])
        label = f'{group["task_code"]} {group["task_name"]}'
        if group["subset_label"]:
            label += f' · {group["subset_label"]}'
        cards.append(
            f'<article class="task-card">'
            f'{card_thumbnail(group.get("image_paths", "[]"), url)}'
            f'<div class="result-main">'
            f'<div class="result-head"><a href="{escape(url)}">{escape(label)}</a>'
            f'<span class="id-badge">{escape(group["sample_count"])} 个样本</span></div>'
            f'<div class="result-meta">{escape(group["dimension"])} · '
            f'{escape(group["condition_count"])} 个条件分支 · '
            f'{escape(group["answer_count"])} 条模型回答</div>'
            f'<a class="open-link" href="{escape(url)}">查看该任务全部样本 →</a>'
            f'</div>'
            f'</article>'
        )
    body = f"""
    <header class="topbar"><div class="wrap">
      <a class="back" href="/">← 返回检索</a>
      <div class="eyebrow">task browser</div>
      <h1>按任务浏览全部样本</h1>
      <p class="subtitle">先选择任务或 C1 子集，再进入该组的全部样本列表；点击样本可查看全部分支和全部模型回答。</p>
      <div class="stats">{len(groups)} 个任务分组 · {sample_count} 个样本 · {condition_count} 个条件分支</div>
    </div></header>
    <main class="wrap">
      <section class="task-grid">{''.join(cards)}</section>
    </main>
    """
    return page("按任务浏览全部样本", body).encode("utf-8")


def render_task_list(conn: sqlite3.Connection, task: str, subset: str) -> bytes | None:
    requested_subset = subset.strip()
    rows = list_task_samples(conn, task, subset)
    if not rows:
        return None
    first = rows[0]
    title = f'{first["task_code"]} {first["task_name"]}'
    if requested_subset and first["subset_label"]:
        title += f' · {first["subset_label"]}'
    condition_count = sum(row["condition_count"] for row in rows)
    answer_count = sum(row["answer_count"] for row in rows)

    subset_nav = ""
    if first["task"] == "C1":
        c1_groups = [group for group in get_task_groups(conn) if group["task"] == "C1"]
        links = [
            f'<a class="{"active" if not requested_subset else ""}" href="{escape(list_url("C1"))}">全部 C1</a>'
        ]
        for group in c1_groups:
            active = " active" if group["subset"].lower() == requested_subset.lower() else ""
            links.append(
                f'<a class="{active}" href="{escape(list_url("C1", group["subset"]))}">'
                f'{escape(group["subset_label"])} ({group["sample_count"]})</a>'
            )
        subset_nav = f'<nav class="subset-nav">{"".join(links)}</nav>'

    body = f"""
    <header class="topbar"><div class="wrap">
      <a class="back" href="/tasks">← 返回任务列表</a>
      <div class="eyebrow">{escape(first["dimension"])} · {escape(first["task"])}</div>
      <h1>{escape(title)}</h1>
      <p class="subtitle">该分组共 {len(rows)} 个样本、{condition_count} 个条件分支、{answer_count} 条模型回答。</p>
      {subset_nav}
    </div></header>
    <main class="wrap">
      <section class="results">{render_cards(rows)}</section>
    </main>
    """
    return page(title, body).encode("utf-8")


def parse_json(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def render_label_record(record: dict[str, Any]) -> str:
    if not record:
        return '<p class="muted">无独立标注记录。</p>'
    rows = []
    for key, value in record.items():
        if key in PATH_KEYS_TO_HIDE:
            continue
        rendered = text_value(value)
        if len(rendered) > 1800:
            rendered = rendered[:1797] + "..."
        rows.append(
            f'<tr><th>{escape(key)}</th><td><pre class="inline-pre">{escape(rendered)}</pre></td></tr>'
        )
    return '<table class="kv">' + "".join(rows) + "</table>"


def render_answer(answer: str, eval_status: str, eval_summary: str, eval_reason: str, eval_json: str) -> str:
    if not answer:
        answer_html = '<span class="muted">（空回答）</span>'
    elif len(answer) > 700:
        answer_html = (
            f'<div class="answer-preview">{escape(shorten(answer, 620))}</div>'
            f'<details><summary>展开完整回答</summary><pre class="answer-pre">{escape(answer)}</pre></details>'
        )
    else:
        answer_html = f'<pre class="answer-pre">{escape(answer)}</pre>'

    status_text = {
        "correct": "正确 / 通过",
        "incorrect": "错误 / 风险",
        "judged": "已判分",
        "none": "无评分",
    }.get(eval_status, eval_status)
    eval_html = f'<span class="status {escape(eval_status)}">{escape(status_text)}</span>'
    if eval_summary:
        eval_html += f'<div class="eval-summary">{escape(eval_summary)}</div>'
    if eval_reason:
        eval_html += f'<details><summary>评分理由</summary><div class="reason">{escape(eval_reason)}</div></details>'
    if eval_json:
        eval_html += (
            f'<details><summary>评分 JSON</summary><pre class="json-pre">{escape(eval_json)}</pre></details>'
        )
    return f'{answer_html}<div class="eval-box">{eval_html}</div>'


def render_sample(conn: sqlite3.Connection, base_key: str) -> bytes | None:
    samples = conn.execute(
        "SELECT * FROM samples WHERE base_key = ? ORDER BY condition_order, condition",
        (base_key,),
    ).fetchall()
    if not samples:
        return None

    first = samples[0]
    models = get_models(conn)
    responses = conn.execute(
        """
        SELECT r.*, s.condition, s.condition_order
        FROM responses r
        JOIN samples s ON s.sample_key = r.sample_key
        WHERE s.base_key = ?
        """,
        (base_key,),
    ).fetchall()
    response_map: dict[tuple[str, str], sqlite3.Row] = {
        (row["condition"], row["model"]): row for row in responses
    }

    title = f"{first['task_code']} {first['task_name']}"
    if first["subset_label"]:
        title += f" · {first['subset_label']}"
    title += f" · ID {first['sample_id']}"

    total_present = sum(1 for row in responses if row["response"])
    expected = len(samples) * len(models)
    label_record = parse_json(first["label_json"], {})

    condition_nav = "".join(
        f'<a href="#condition-{index}">{escape(row["condition_label"])}</a>'
        for index, row in enumerate(samples, start=1)
    )

    blocks: list[str] = []
    for index, sample in enumerate(samples, start=1):
        images = parse_json(sample["image_paths"], [])
        target = parse_json(sample["target_json"], None)
        target_display = format_target(sample["task"], target)
        target_preview = shorten(target_display, 90)
        gallery = []
        for image_path in images:
            gallery.append(
                f'<a href="{escape(image_url(image_path))}" target="_blank" class="image-card">'
                f'<img src="{escape(image_url(image_path))}" alt="{escape(image_path)}" loading="lazy">'
                f'<span>{escape(image_path)}</span></a>'
            )
        if not gallery:
            gallery.append('<div class="empty small">未记录输入图片。</div>')

        answer_rows = []
        for model in models:
            response = response_map.get((sample["condition"], model))
            if response is None:
                answer_rows.append(
                    f'<tr class="missing-row"><td class="model-name">{escape(model)}</td>'
                    f'<td><span class="status none">缺失</span> '
                    f'<span class="muted">该分支未找到回答</span></td></tr>'
                )
                continue
            row_class = "answer-row"
            if response["eval_status"] == "incorrect":
                row_class += " has-error"
            elif response["eval_status"] == "correct":
                row_class += " has-pass"
            answer_rows.append(
                f'<tr class="{row_class}"><td class="model-name">{escape(model)}'
                f'<div class="source-path">{escape(response["source_file"])}</div></td>'
                f'<td class="answer-cell">'
                f'{render_answer(response["response"], response["eval_status"], response["eval_summary"], response["eval_reason"], response["eval_json"])}'
                f'</td></tr>'
            )

        open_attr = " open" if index == 1 else ""
        blocks.append(
            f'<details class="condition-block" id="condition-{index}"{open_attr}>'
            f'<summary><span class="condition-index">{index:02d}</span> '
            f'{escape(sample["condition_label"])} '
            f'<span class="condition-count">{sample["response_count"]}/{len(models)} 个模型</span>'
            f'<span class="condition-target">标准答案：{escape(target_preview)}</span>'
            f'</summary>'
            f'<div class="condition-body">'
            f'<div class="condition-grid">'
            f'<div><h3>输入图片</h3><div class="gallery">{"".join(gallery)}</div></div>'
            f'<div><h3>正确答案 / 参考目标</h3><pre class="target">{escape(target_display)}</pre></div>'
            f'</div>'
            f'<details class="prompt-box" open><summary>模型实际输入提示</summary>'
            f'<pre class="prompt">{escape(sample["question"])}</pre></details>'
            f'<table class="model-table"><thead><tr><th>模型</th><th>回答与评分</th></tr></thead>'
            f'<tbody>{"".join(answer_rows)}</tbody></table>'
            f'</div></details>'
        )

    body = f"""
    <header class="topbar detail-head"><div class="wrap">
      <a class="back" href="/">← 返回检索</a>
      <div class="eyebrow">{escape(first['dimension'])} · {escape(first['task_code'])}</div>
      <h1>{escape(title)}</h1>
      <div class="stats">{len(samples)} 个条件分支 · {len(models)} 个模型 ·
        {total_present}/{expected} 条回答存在</div>
      <nav class="condition-nav">{condition_nav}</nav>
    </div></header>
    <main class="wrap detail-main">
      <section class="label-panel">
        <details><summary>数据集标注记录（辅助核对，不参与回答展示）</summary>{render_label_record(label_record)}</details>
      </section>
      {''.join(blocks)}
    </main>
    """
    return page(title, body).encode("utf-8")


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #667085;
  --line: #d9dee5;
  --accent: #175cd3;
  --accent-soft: #eef4ff;
  --pass: #087443;
  --pass-soft: #ecfdf3;
  --error: #b42318;
  --error-soft: #fff1f0;
  --warn-soft: #fff8e6;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
  line-height: 1.5;
}}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ background: #eef1f5; padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }}
.wrap {{ width: min(1500px, calc(100% - 40px)); margin: 0 auto; }}
.topbar {{ background: #101828; color: #fff; padding: 30px 0 24px; }}
.topbar a {{ color: #c7d7fe; }}
.eyebrow {{ color: #9ec5fe; text-transform: uppercase; letter-spacing: .08em; font-size: 12px; font-weight: 700; }}
h1 {{ margin: 6px 0 4px; font-size: clamp(25px, 3vw, 38px); line-height: 1.15; }}
.subtitle {{ margin: 0 0 20px; color: #d0d5dd; }}
.search-form {{ display: grid; grid-template-columns: 1fr 300px auto; gap: 10px; }}
.search-form input, .search-form select, .search-form button {{
  height: 46px; border-radius: 9px; border: 1px solid #475467; padding: 0 13px;
  font-size: 15px; background: #fff; color: var(--ink);
}}
.search-form button {{ background: #fdb022; border-color: #fdb022; font-weight: 800; cursor: pointer; padding: 0 24px; }}
.stats {{ margin-top: 16px; color: #d0d5dd; font-size: 14px; }}
.hint {{ margin-top: 7px; color: #98a2b3; font-size: 13px; }}
.task-entry {{ display: inline-block; margin-top: 12px; background: #1d2939; color: #dbe7ff; border-radius: 8px; padding: 7px 11px; font-size: 13px; font-weight: 700; }}
.task-chips {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 18px 0 4px; }}
.task-chip {{
  display: inline-flex; align-items: center; gap: 6px; padding: 7px 10px;
  border: 1px solid var(--line); border-radius: 999px; background: #fff; color: #344054; font-size: 13px;
}}
.task-chip span {{ background: #eef1f5; border-radius: 999px; padding: 0 7px; color: var(--muted); }}
.task-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; padding: 18px 0 40px; }}
.task-card {{ display: flex; gap: 14px; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }}
.subset-nav {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 13px; }}
.subset-nav a {{ background: #1d2939; color: #dbe7ff; border-radius: 7px; padding: 5px 9px; font-size: 13px; }}
.subset-nav a.active {{ background: #fdb022; color: #101828; font-weight: 800; }}
.results {{ padding: 16px 0 40px; display: grid; gap: 12px; }}
.result-card {{ display: flex; gap: 14px; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }}
.result-main {{ flex: 1; min-width: 0; }}
.card-thumb {{ flex: 0 0 132px; display: flex; align-items: center; justify-content: center; width: 132px; height: 108px; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: #f2f4f7; color: var(--muted); font-size: 12px; }}
.card-thumb img {{ width: 100%; height: 100%; object-fit: cover; }}
.card-thumb.no-image {{ border-style: dashed; }}
.result-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 14px; font-size: 17px; font-weight: 750; }}
.task-code {{ color: var(--accent); }}
.id-badge {{ background: var(--accent-soft); color: var(--accent); border-radius: 7px; padding: 3px 8px; font-size: 13px; white-space: nowrap; }}
.result-meta {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
.answer-line {{ margin-top: 8px; font-size: 13.5px; }}
.answer-badge {{ display: inline-block; background: var(--pass-soft); color: var(--pass); border-radius: 5px; padding: 1px 7px; font-weight: 800; }}
.answer-line.no-answer .answer-badge {{ background: #f2f4f7; color: var(--muted); }}
.answer-value {{ font-weight: 700; word-break: break-word; }}
.question {{ margin: 10px 0 8px; color: #344054; }}
.open-link {{ font-size: 14px; font-weight: 700; }}
.empty {{ background: #fff; border: 1px dashed #b8c0cc; border-radius: 12px; padding: 30px; color: var(--muted); }}
.empty.small {{ padding: 14px; }}
.note {{ color: var(--muted); }}
.muted {{ color: var(--muted); }}
.detail-head {{ padding-bottom: 18px; }}
.back {{ display: inline-block; margin-bottom: 16px; }}
.condition-nav {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 14px; }}
.condition-nav a {{ background: #1d2939; color: #dbe7ff; border-radius: 7px; padding: 5px 9px; font-size: 13px; }}
.detail-main {{ padding: 18px 0 60px; display: grid; gap: 14px; }}
.label-panel {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }}
.condition-block {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
.condition-block > summary {{ cursor: pointer; padding: 16px 18px; font-size: 18px; font-weight: 800; background: #fbfcfe; list-style: none; }}
.condition-block > summary::-webkit-details-marker {{ display: none; }}
.condition-index {{ color: var(--accent); }}
.condition-count {{ color: var(--muted); font-size: 13px; font-weight: 500; margin-left: 7px; }}
.condition-target {{ float: right; color: var(--pass); background: var(--pass-soft); border-radius: 5px; padding: 2px 8px; font-size: 12px; font-weight: 750; max-width: 52%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.condition-body {{ padding: 0 18px 20px; }}
.condition-grid {{ display: grid; grid-template-columns: minmax(300px, 1.25fr) minmax(220px, .75fr); gap: 18px; align-items: start; }}
h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 16px 0 8px; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.image-card {{ display: block; width: min(100%, 520px); border: 1px solid var(--line); border-radius: 9px; overflow: hidden; color: var(--muted); font-size: 11px; }}
.image-card img {{ display: block; width: 100%; max-height: 420px; object-fit: contain; background: #f2f4f7; }}
.image-card span {{ display: block; padding: 6px 8px; word-break: break-all; }}
.target, .json-pre {{ background: #f8fafc; border: 1px solid #e4e7ec; border-radius: 8px; padding: 10px; white-space: pre-wrap; word-break: break-word; }}
.prompt-box {{ margin: 14px 0; border: 1px solid var(--line); border-radius: 9px; padding: 10px 12px; background: #fcfcfd; }}
.prompt-box summary {{ cursor: pointer; font-weight: 700; }}
.prompt {{ white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto; margin: 10px 0 0; font-size: 13px; }}
.model-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.model-table th {{ text-align: left; background: #f2f4f7; padding: 9px 11px; border: 1px solid var(--line); }}
.model-table td {{ padding: 10px 11px; border: 1px solid var(--line); vertical-align: top; }}
.model-name {{ width: 190px; font-weight: 700; word-break: break-word; }}
.source-path {{ margin-top: 5px; color: var(--muted); font-size: 10.5px; font-weight: 400; word-break: break-all; }}
.answer-row.has-pass td:first-child {{ border-left: 4px solid var(--pass); }}
.answer-row.has-error td:first-child {{ border-left: 4px solid var(--error); }}
.missing-row {{ background: #fafafa; }}
.answer-pre {{ margin: 0 0 7px; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; line-height: 1.45; }}
.answer-preview {{ margin-bottom: 6px; color: #344054; }}
.eval-box {{ border-top: 1px dashed #d0d5dd; padding-top: 7px; }}
.status {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 750; background: #eef1f5; color: #344054; }}
.status.correct {{ background: var(--pass-soft); color: var(--pass); }}
.status.incorrect {{ background: var(--error-soft); color: var(--error); }}
.status.judged {{ background: var(--warn-soft); color: #8a4b00; }}
.eval-summary {{ color: #475467; font-size: 12.5px; margin-top: 5px; }}
.reason {{ color: #475467; font-size: 12.5px; white-space: pre-wrap; }}
.kv {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.kv th {{ width: 190px; text-align: left; vertical-align: top; color: var(--muted); padding: 5px 8px; border-bottom: 1px solid #eef1f5; }}
.kv td {{ padding: 5px 8px; border-bottom: 1px solid #eef1f5; }}
.inline-pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: inherit; }}
details > summary {{ cursor: pointer; }}
@media (max-width: 900px) {{
  .search-form {{ grid-template-columns: 1fr; }}
  .condition-grid {{ grid-template-columns: 1fr; }}
  .model-name {{ width: auto; }}
  .result-card, .task-card {{ flex-direction: column; }}
  .card-thumb {{ width: 100%; height: 180px; flex-basis: auto; }}
  .condition-target {{ float: none; display: inline-block; margin-left: 7px; max-width: 100%; }}
  .wrap {{ width: min(100% - 24px, 1500px); }}
}}
</style>
</head>
<body>{body}</body>
</html>"""


class InspectorHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], db_path: Path):
        self.db_path = db_path
        super().__init__(address, InspectorHandler)


class InspectorHandler(BaseHTTPRequestHandler):
    server: InspectorHTTPServer
    server_version = "MedMMLMTrustSampleInspector/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                query = params.get("q", [""])[0]
                task = params.get("task", [""])[0]
                limit = 100
                conn = open_db(self.server.db_path)
                try:
                    payload = render_index(conn, query, task, limit)
                finally:
                    conn.close()
                self.send_bytes(payload, "text/html; charset=utf-8")
                return
            if parsed.path == "/sample":
                base_key = params.get("key", [""])[0]
                conn = open_db(self.server.db_path)
                try:
                    payload = render_sample(conn, base_key) if base_key else None
                finally:
                    conn.close()
                if payload is None:
                    self.send_error_page(404, "样本不存在", f"没有找到 {base_key!r}。")
                else:
                    self.send_bytes(payload, "text/html; charset=utf-8")
                return
            if parsed.path == "/list":
                task = params.get("task", [""])[0]
                subset = params.get("subset", [""])[0]
                conn = open_db(self.server.db_path)
                try:
                    payload = render_task_list(conn, task, subset) if task else None
                finally:
                    conn.close()
                if payload is None:
                    self.send_error_page(404, "任务不存在", f"没有找到 {task!r} / {subset!r}。")
                else:
                    self.send_bytes(payload, "text/html; charset=utf-8")
                return
            if parsed.path in {"/tasks", "/minimal"}:
                conn = open_db(self.server.db_path)
                try:
                    payload = render_tasks(conn)
                finally:
                    conn.close()
                self.send_bytes(payload, "text/html; charset=utf-8")
                return
            if parsed.path == "/image":
                self.serve_image(params.get("path", [""])[0])
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_error_page(404, "页面不存在", parsed.path)
        except BrokenPipeError:
            return
        except Exception as exc:  # Keep the viewer usable when one sample is malformed.
            self.send_error_page(500, "查看器错误", str(exc))

    def serve_image(self, relative_path: str) -> None:
        if not relative_path:
            self.send_error_page(400, "缺少图片路径", "")
            return
        candidate = (ROOT / relative_path).resolve()
        if not candidate.is_relative_to(ROOT):
            self.send_error_page(403, "拒绝访问", relative_path)
            return
        if not candidate.exists() and relative_path.startswith("data_v2/"):
            candidate = (ROOT / relative_path.replace("data_v2/", "data/", 1)).resolve()
        if not candidate.is_file():
            self.send_error_page(404, "图片不存在", relative_path)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        payload = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(payload)

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_error_page(self, status: int, title: str, detail: str) -> None:
        body = (
            f'<header class="topbar"><div class="wrap"><h1>{escape(title)}</h1>'
            f'<p class="subtitle">{escape(detail)}</p><a href="/">← 返回检索</a></div></header>'
        )
        payload = page(title, body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(db_path: Path, host: str, port: int, no_browser: bool) -> None:
    conn = open_db(db_path)
    try:
        stats = get_stats(conn)
        models = get_models(conn)
    finally:
        conn.close()

    server = InspectorHTTPServer((host, port), db_path)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/"
    print(
        f"Viewer: {url}\n"
        f"Index: {db_path}\n"
        f"Samples: {stats.get('base_samples', '?')} base / "
        f"{stats.get('condition_samples', '?')} conditions; models: {len(models)}\n"
        "Press Ctrl+C to stop."
    )
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Index path (default: {DEFAULT_DB})")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the SQLite index from logs/")
    parser.add_argument("--build-only", action="store_true", help="Build the index and exit")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    ensure_index(db_path, rebuild=args.rebuild)
    if args.build_only:
        return 0
    serve(db_path, args.host, args.port, no_browser=args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
