"""Safety guard for the semantic cache.

Embedding similarity says two prompts are about the same thing, not that they have
the same answer ("Apple revenue in 2024" vs "2025" embed almost identically).
The guard compares the parts of a prompt that change the answer: numbers, named
entities and negations. It is deliberately strict: when unsure it blocks, because a
missed cache hit costs a model call but a false hit returns a wrong answer.

These are heuristics, not NLP. The false hit rate must be measured on a labeled
sample (phase 2, step P2.6), not assumed.
"""

import re
from dataclasses import dataclass

# ---------- Numbers (digits and number words, in order of appearance) ----------

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

_GROUPED = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,000 and 1,000.50
_PLAIN = r"\d+(?:\.\d+)*"  # 2024, 3.11, 1.2.3 (digits inside words too: Q3)
_WORDS = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_NUMBER_RE = re.compile(rf"\b(?:{_WORDS})\b|{_GROUPED}|{_PLAIN}", re.IGNORECASE)


def extract_numbers(text: str) -> list[str]:
    """Numbers in order of appearance. Only thousands commas are normalized.

    "3.10" and "3.1" stay different on purpose (they can be versions, not decimals).
    """
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).lower()
        numbers.append(_NUMBER_WORDS.get(token, token.replace(",", "")))
    return numbers


# ---------- Named entities (capitalization heuristic, no NLP library) ----------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:['\u2019][A-Za-z]+)?")

def _words(text: str) -> set[str]:
    """All words of a text in lowercase, possessives stripped (France's becomes france)."""
    words = set()
    for token in _WORD_RE.findall(text):
        word = token.lower().replace("\u2019", "'")
        if word.endswith("'s"):
            word = word[:-2]
        words.add(word)
    return words

# Capitalized words that are not entities: question words, auxiliaries, pronouns
# and common request verbs that start a sentence.
_NOT_ENTITIES = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "has", "have", "had", "the", "a", "an", "and", "or", "but", "if", "then",
    "so", "because", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "to", "with", "about", "please", "tell", "explain", "describe", "give",
    "write", "list", "show", "summarize", "summarise", "translate", "define",
    "compare", "convert", "calculate", "find", "create", "make", "help",
    "provide", "suggest", "recommend", "generate", "draft", "rewrite", "fix",
    "review", "analyze", "analyse", "answer", "i", "i'm", "i'll", "i've",
    "i'd", "my", "me", "we", "you", "your", "it", "its", "this", "that",
    "these", "those", "there", "here", "hi", "hello", "hey", "thanks", "thank",
    "yes", "no", "not", "also", "any", "some", "let", "need", "want", "like",
}


def _looks_like_entity(token: str) -> bool:
    # Capitalized (Paris, USD) or capitals inside (iPhone, OpenAI)
    return token[0].isupper() or any(c.isupper() for c in token[1:])


def extract_entities(text: str) -> set[str]:
    entities = set()
    for token in _WORD_RE.findall(text):
        lowered = token.lower().replace("\u2019", "'")
        if lowered in _NOT_ENTITIES or not _looks_like_entity(token):
            continue
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        entities.add(lowered)
    return entities


# ---------- Negations ----------

_NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "arent",
    "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt", "hasnt",
    "havent", "hadnt",
    # Words that flip the meaning without being a plain negation
    "illegal", "unlawful", "unsafe", "false", "untrue", "incorrect",
    "impossible", "invalid", "unhealthy", "forbidden", "prohibited",
}


def extract_negations(text: str) -> set[str]:
    found = set()
    for token in _WORD_RE.findall(text):
        normalized = token.lower().replace("'", "").replace("\u2019", "")
        if normalized in _NEGATION_WORDS:
            found.add(normalized)
    return found


# ---------- Time sensitive prompts (never cache) ----------

_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|now|nowadays|currently|current|"
    r"latest|newest|recent|recently|real\s*time|up\s+to\s+date|"
    r"(?:this|last|next|past|previous)\s+(?:week|month|year|quarter|morning|"
    r"evening|weekend|season))\b",
    re.IGNORECASE,
)


def is_time_sensitive(text: str) -> bool:
    """True when the answer depends on when it is asked (these must bypass the cache)."""
    return _TIME_SENSITIVE_RE.search(text) is not None

# ---------- Style words ----------

# Words that change how long or in what form the answer should be. "What is X?" and
# "Explain X briefly" are about the same thing but need different answers.
_STYLE_WORDS = {
    "brief", "briefly", "short", "shorter", "shortest", "concise", "concisely",
    "detailed", "detail", "details", "thorough", "thoroughly", "comprehensive",
    "simple", "simply", "simplified", "eli5", "tldr", "bullet", "bullets",
    "table", "essay", "paragraph", "paragraphs",
}


def extract_style_words(text: str) -> set[str]:
    return {w for w in _words(text) if w in _STYLE_WORDS}


# ---------- The guard and the final decision ----------


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str  # "ok", "number_mismatch", "entity_mismatch", "negation_mismatch"


def check_guard(new_text: str, cached_text: str) -> GuardResult:
    if extract_numbers(new_text) != extract_numbers(cached_text):
        return GuardResult(False, "number_mismatch")

    # A name written in capitals on one side and in lowercase on the other is the same
    # name, so each name must appear (in any case) as a word in the other text
    new_entities = extract_entities(new_text)
    cached_entities = extract_entities(cached_text)
    if not (new_entities <= _words(cached_text) and cached_entities <= _words(new_text)):
        return GuardResult(False, "entity_mismatch")

    if extract_negations(new_text) != extract_negations(cached_text):
        return GuardResult(False, "negation_mismatch")
    if extract_style_words(new_text) != extract_style_words(cached_text):
        return GuardResult(False, "style_mismatch")
    return GuardResult(True, "ok")


@dataclass(frozen=True)
class CacheDecision:
    hit: bool
    # "hit", "below_threshold", "number_mismatch", "entity_mismatch", "negation_mismatch"
    reason: str
    similarity: float


def decide_hit(
    similarity: float, new_text: str, cached_text: str, threshold: float
) -> CacheDecision:
    """A cache hit needs enough similarity AND a passing guard.

    There is no default threshold on purpose: it must be tuned on a labeled sample.
    """
    if similarity < threshold:
        return CacheDecision(False, "below_threshold", similarity)
    guard = check_guard(new_text, cached_text)
    if not guard.ok:
        return CacheDecision(False, guard.reason, similarity)
    return CacheDecision(True, "hit", similarity)