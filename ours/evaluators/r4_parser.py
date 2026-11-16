"""Conservative answer recovery for R.4's fixed A-E anatomy questions.

Only call this when the ordinary MCQ parser returned no answer.  A mention of
an option in a discussion is not a final choice.
"""

import re


def recover_r4_option(response, option_map):
    if not isinstance(response, str) or not isinstance(option_map, dict):
        return None
    text = response.strip()
    valid = {key: value for key, value in option_map.items()
             if key in "ABCDE" and isinstance(value, str) and value.strip()}
    if not text or not valid:
        return None

    # Explicit choice at the *end* of the answer.  This also accepts the
    # observed "The correct answer is:\n\nE" and boxed final-answer forms.
    tail = text[-180:]
    patterns = (
        r"(?:final answer\s*:\s*(?:the final answer is\s*)?|"
        r"(?:the )?most likely answer\s+is\s*:?|"
        r"(?:the )?correct (?:answer|option)\s+is\s*:?)"
        r"\s*(?:\$\s*)?(?:\\boxed\{)?([A-E])\}?\$?"
        r"(?:\s*[.:)]\s*[A-Za-z]+)?[.!*\s]*$",
        r"(?:final answer\s*:\s*(?:the final answer is\s*)?)"
        r"\$\\boxed\{([A-E])\}\$[!*\s]*$",
    )
    for pattern in patterns:
        found = re.search(pattern, tail, re.IGNORECASE)
        if found and found.group(1).upper() in valid:
            return [found.group(1).upper()]

    # A final, explicitly named option can be written without its letter.
    # Match the option value exactly instead of looking for arbitrary mentions.
    named_final = re.search(
        r'"([^"\n]+)"(?:\s*\([^)]*\))?\s+is the most likely intended answer\.?\s*$',
        text, re.IGNORECASE,
    )
    if named_final:
        matches = [letter for letter, value in valid.items()
                   if named_final.group(1).casefold() == value.strip().casefold()]
        if len(matches) == 1:
            return matches

    # A clear selection followed only by more analysis of rejected options.
    plausible = list(re.finditer(r'\b([A-E]) is the most plausible answer\b', text, re.IGNORECASE))
    if plausible:
        choice = plausible[-1]
        remaining = text[choice.end():]
        if (choice.group(1).upper() in valid and
                not re.search(r'\b(?:correct|final|most likely|most plausible)\s+(?:answer|option)\b',
                              remaining, re.IGNORECASE)):
            return [choice.group(1).upper()]

    # The opening conclusion may name a structure before a long, truncated
    # option-by-option explanation.  Require an emphasized exact option value
    # and no later revision signal; an arbitrary keyword mention is unsafe.
    opening = re.search(r'\bcharacteristic of the\s+\*\*([^*\n]+)\*\*\.', text[:400],
                        re.IGNORECASE)
    if opening and not re.search(r'\b(?:however|instead|actually|on second thought)\b',
                                 text[opening.end():], re.IGNORECASE):
        matches = [letter for letter, value in valid.items()
                   if opening.group(1).casefold() == value.strip().casefold()]
        if len(matches) == 1:
            return matches

    # Short declarative responses from LLaVA-Med name one structure instead
    # of a letter.  Read only the first assertion; do not scan a discussion
    # where rejected alternatives can appear later.
    if len(text) > 400:
        return None
    if re.search(r"\b(?:but|however|perhaps|maybe|uncertain)\b", text, re.IGNORECASE):
        return None
    lead = re.match(
        r"^The\s+(?:organ in the image is|image (?:shows|is most likely showing)|"
        r"most likely organ depicted in the image is)\s+"
        r"(?:the|a|an)\s+(?:(?:right|left)\s+)?"
        r"([A-Za-z][A-Za-z -]*?)(?=[,.]|\s+which\b|\s+as\b|$)",
        text, re.IGNORECASE,
    )
    if not lead:
        return None
    named = re.sub(r"\s+organ$", "", lead.group(1).strip().casefold())
    matches = [letter for letter, value in valid.items()
               if named == value.strip().casefold()]
    return matches if len(matches) == 1 else None
