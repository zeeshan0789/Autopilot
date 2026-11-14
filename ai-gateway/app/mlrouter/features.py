"""Features of a prompt that are known before any model has answered.

Everything here is computed from the messages only: lengths, digits, operators, quotes and a
few cue words that name the kind of task ("json", "sort", "reverse", "reply with only"...).
The cues are English only. A router trained on generated prompts may partly learn the
templates of those prompts, which is why it is also tested on hand written ones.
"""

import re

from app.evals.checks import numbers_in

_CUE_PATTERNS = {
    "cue_only": r"\bonly\b",
    "cue_exactly": r"\bexactly\b",
    "cue_at_most_least": r"\bat (?:most|least)\b",
    "cue_json": r"\bjson\b",
    "cue_list": r"\b(?:list|bulleted|numbered)\b",
    "cue_sort": r"\bsort\b",
    "cue_reverse": r"\breverse\b",
    "cue_sentence": r"\bsentences?\b",
    "cue_word": r"\bwords?\b",
    "cue_letter": r"\bletters?\b",
    "cue_capital": r"\bcapital\b",
    "cue_symbol": r"\bsymbol\b",
    "cue_date": r"\bdate\b",
    "cue_total": r"\btotal\b",
    "cue_sequence": r"\bsequence\b",
    "cue_how_many": r"\bhow many\b",
    "cue_reply_with": r"\breply with\b",
    "cue_all_caps": r"\ball capital letters\b|\bin capitals\b|\ball caps\b",
    "cue_end_with": r"\bend (?:your|the) (?:reply|answer|response)\b",
}
_CUES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in _CUE_PATTERNS.items()}
_OPERATION = re.compile(r"\d\s*[-+*/^]\s*\d")
_DIGIT_RUN = re.compile(r"\d+")

_NUMERIC = [
    "chars", "words", "lines", "total_chars", "messages", "has_system", "system_chars",
    "digits", "numbers", "max_number_digits", "question_marks", "operators", "quotes",
    "commas", "parentheses", "upper_ratio", "avg_word_len", "longest_word",
]
FEATURE_NAMES = _NUMERIC + list(_CUES)


def extract_features(messages: list[dict]) -> list[float]:
    """One number per name in FEATURE_NAMES. The last non system message is the text."""
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m["content"] for m in messages if m["role"] != "system"]
    text = turns[-1] if turns else ""
    everything = system + "\n" + text  # the system prompt names rules the router must see

    words = text.split()
    letters = [c for c in text if c.isalpha()]
    digit_runs = _DIGIT_RUN.findall(text)

    values = {
        "chars": len(text),
        "words": len(words),
        "lines": text.count("\n") + 1 if text else 0,
        "total_chars": sum(len(m["content"]) for m in messages),
        "messages": len(messages),
        "has_system": 1 if system else 0,
        "system_chars": len(system),
        "digits": sum(len(run) for run in digit_runs),
        "numbers": len(numbers_in(text)),
        "max_number_digits": max((len(run) for run in digit_runs), default=0),
        "question_marks": text.count("?"),
        "operators": len(_OPERATION.findall(text)),
        "quotes": text.count("'") + text.count('"'),
        "commas": text.count(","),
        "parentheses": text.count("("),
        "upper_ratio": sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0,
        "avg_word_len": sum(len(w) for w in words) / len(words) if words else 0.0,
        "longest_word": max((len(w) for w in words), default=0),
    }
    for name, pattern in _CUES.items():
        values[name] = 1 if pattern.search(everything) else 0
    return [float(values[name]) for name in FEATURE_NAMES]