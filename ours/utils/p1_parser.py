"""Parse an unambiguous P.1 class from a model response."""

import re
from typing import Any, Optional


ALIASES = {
    0: {"clean", "no text", "no watermark", "no phi", "no phi detected"},
    1: {"phi", "phi exists", "contains phi", "yes", "true"},
    2: {"noise", "random text", "non-phi", "irrelevant text"},
}


def parse_p1_class(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw in (0, 1, 2) else None
    if not isinstance(raw, str) or not raw.strip():
        return None

    text = raw.strip()
    normalized = text.lower().rstrip(".").strip()
    for label, names in ALIASES.items():
        if normalized in names:
            return label
    if re.fullmatch(r"[012](?:\.0+)?", normalized):
        return int(float(normalized))

    candidates = []
    boundary = r"(?=$|\s|[),;:]|\.(?!\d))"
    leading = re.match(r"^([012])" + boundary, text)
    if leading:
        candidates.append(int(leading.group(1)))
    for match in re.finditer(
        r"\b(?:classified\s+as|classification\s+is|category|class|answer\s+is)\s*[:#]?\s*([012])" + boundary,
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(int(match.group(1)))
    return candidates[0] if candidates and len(set(candidates)) == 1 else None
