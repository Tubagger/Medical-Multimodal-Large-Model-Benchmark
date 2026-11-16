"""Persistence helpers for model responses before evaluation.

Response records deliberately use JSONL so existing code that scans ``*.json``
continues to see only evaluated result files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional


EVALUATION_FIELDS = {"evals", "subset_evals"}
ERROR_PREFIXES = ('Error in generation:', 'Error generating response:', 'API Error:', 'Error:')


def final_answer_only(response: Any):
    """Remove explicit thinking; an unclosed block has no final answer after it."""
    if not isinstance(response, str):
        return response
    cleaned = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', response,
                     flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.split(r'<think\b[^>]*>', cleaned, maxsplit=1,
                       flags=re.IGNORECASE)[0]
    return cleaned.strip()


def has_final_answer(response: Any) -> bool:
    return (isinstance(response, str) and bool(response.strip())
            and not response.lstrip().startswith(ERROR_PREFIXES)
            and bool(final_answer_only(response)))


def cached_answer_complete(record: Dict[str, Any]) -> bool:
    """Whether a saved record contains a usable answer from an earlier run.

    A record is complete only if at least one of its answer fields is a
    non-error final answer.  Repeated sampling stores ``pred`` as a list and
    ``major_answer`` as the aggregate; both paths are supported.
    """
    if not isinstance(record, dict):
        return False
    if has_final_answer(record.get('response')):
        return True
    if has_final_answer(record.get('major_answer')):
        return True
    preds = record.get('pred')
    return (isinstance(preds, list) and bool(preds)
            and any(has_final_answer(value) for value in preds))


def canonical_content(content: Any):
    """Return content with legacy ``data_v2/`` image paths mapped to ``data/``."""
    if not isinstance(content, dict):
        return content
    normalized = dict(content)
    image_path = normalized.get('image_path')
    if isinstance(image_path, str) and image_path.startswith('data_v2/'):
        normalized['image_path'] = 'data/' + image_path[len('data_v2/'):]
    return normalized


def equivalent_content(cached: Any, expected: Any) -> bool:
    return canonical_content(cached) == canonical_content(expected)


def _result_stem(path):
    text = str(path)
    for suffix in ('.responses.jsonl', '.scored.json', '.rejudged.json', '.json'):
        if text.endswith(suffix):
            return text[:-len(suffix)]
    value = Path(text)
    return str(value.with_suffix('')) if value.suffix else text


def default_score_file(log_file: Optional[str]) -> Optional[str]:
    """Normalize legacy or canonical result paths to one scoring filename."""
    return _result_stem(log_file) + '.scored.json' if log_file else None


def default_response_file(log_file: Optional[str]) -> Optional[str]:
    return _result_stem(log_file) + '.responses.jsonl' if log_file else None


def without_evaluation(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the inference-stage fields from an evaluated result row."""
    clean = {key: value for key, value in row.items() if key not in EVALUATION_FIELDS}
    extra = clean.get('extra')
    if isinstance(extra, dict):
        extra = {key: value for key, value in extra.items()
                 if key not in {'input_fingerprint', 'shared_source_model', 'generated_on'}}
        if extra:
            clean['extra'] = extra
        else:
            clean.pop('extra', None)
    return clean


def load_response_records(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []

    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: response record must be an object")
            records.append(without_evaluation(record))
    return records


def write_response_records(path: Optional[str], records: Iterable[Dict[str, Any]]) -> None:
    """Atomically replace a JSONL response file."""
    if not path:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            for record in records:
                handle.write(json.dumps(without_evaluation(record), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    os.replace(temporary, destination)


def upsert_response_records(
    existing: Iterable[Dict[str, Any]],
    new_records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge new records by sample ID while preserving the original order."""
    merged = [without_evaluation(record) for record in existing]
    seen = set()
    for record in merged:
        key = record.get('id')
        if key is None:
            continue
        key = str(key)
        if key in seen:
            raise ValueError(f'Duplicate cached answer id={key}')
        seen.add(key)
    positions = {
        str(record["id"]): index
        for index, record in enumerate(merged)
        if record.get("id") is not None
    }

    for record in new_records:
        clean = without_evaluation(record)
        sample_id = clean.get("id")
        key = str(sample_id) if sample_id is not None else None
        if key is not None and key in positions:
            merged[positions[key]] = clean
        else:
            if key is not None:
                positions[key] = len(merged)
            merged.append(clean)
    return merged
