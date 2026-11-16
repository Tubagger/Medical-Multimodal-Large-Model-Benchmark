"""Parse P.2 PHI predictions without treating arbitrary prose as a PHI item."""

import json
import re
from typing import Any


def parse_phi_response(raw: Any) -> tuple[list[str], str]:
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return raw, "json_list"
    if not isinstance(raw, str) or not raw.strip():
        return [], "empty"

    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed, "json_list"
        if isinstance(parsed, dict) and all(isinstance(item, str) for item in parsed.values()):
            return list(parsed.values()), "json_object"
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[(.*?)\]", text, re.S)
    if match:
        try:
            parsed = json.loads("[" + match.group(1) + "]")
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed, "embedded_list"
        except json.JSONDecodeError:
            pass

    # Some local responses have a complete bracketed list but damaged quotes
    # or a trailing comma. Recover only fully visible Type: Value lines.
    if text.startswith("[") and text.endswith("]"):
        items = []
        malformed_line = False
        for line in text[1:-1].splitlines():
            line = line.strip().strip(",").strip()
            line = re.sub(r'^"([^"]+)"\s*:\s*"?', r'\1: ', line)
            line = line.strip('"\'').strip()
            if not line:
                continue
            found = re.fullmatch(r"([A-Za-z][A-Za-z0-9 _/-]{0,30})\s*:\s*(\S.*)", line)
            if not found:
                malformed_line = True
                break
            value = found.group(2).strip().strip(",").strip().strip('"\'').strip()
            if not value:
                malformed_line = True
                break
            items.append(f"{found.group(1).strip()}: {value}")
        if items and not malformed_line:
            return items, "recovered_list"

    if text.lower().startswith("i cannot view the actual image"):
        return [], "abstention"
    return [], "invalid"
