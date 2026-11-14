"""Programmatic eval cases for training and testing a learned router.

Every case has a known correct answer and deterministic checks, in 12 families and 3
difficulty levels (the level is written in the case id and is only for analysis, a router
cannot know it at request time). Every case also has a reference answer that passes its own
checks, which the tests verify. The cases are synthetic and template based: a router trained
on them may learn the templates, so it is also tested on the hand written cases.
"""

import json
import random
from pathlib import Path

from app.evals.cases import EvalCase
from app.evals.checks import validate_check

SEED = 7

NAMES = ["Sara", "Omar", "Lina", "Ahmed", "Zara", "Bilal", "Hira", "Usman", "Maya", "Danish", "Noor", "Faraz"]
THINGS = ["apples", "pencils", "stickers", "marbles", "books", "cards"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

LIST_ITEMS = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "cherry", "lemon"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"],
    "animals": ["cat", "dog", "horse", "tiger", "rabbit", "camel", "eagle", "whale"],
    "vegetables": ["carrot", "potato", "onion", "spinach", "tomato", "cabbage", "pumpkin", "radish"],
    "sports": ["cricket", "football", "tennis", "hockey", "rugby", "boxing", "golf", "squash"],
    "tools": ["hammer", "saw", "wrench", "drill", "pliers", "chisel", "screwdriver", "spanner"],
    "countries": ["Pakistan", "Japan", "Brazil", "Canada", "Egypt", "France", "Kenya", "Norway"],
}
TOPICS = [
    "the moon", "a bicycle", "rain", "coffee", "the internet",
    "a forest", "chess", "the ocean", "a library", "music",
]
KEY_POOL = ["name", "age", "city", "email", "job", "country", "score", "id"]

CAPITALS = {
    1: [
        ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("India", "New Delhi"),
        ("China", "Beijing"), ("Russia", "Moscow"), ("United Kingdom", "London"),
        ("Mexico", "Mexico City"), ("Greece", "Athens"), ("Ireland", "Dublin"), ("Sweden", "Stockholm"),
    ],
    2: [
        ("Brazil", "Brasilia"), ("Australia", "Canberra"), ("Turkey", "Ankara"), ("Pakistan", "Islamabad"),
        ("Nigeria", "Abuja"), ("Kenya", "Nairobi"), ("Argentina", "Buenos Aires"), ("Peru", "Lima"),
        ("Norway", "Oslo"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Thailand", "Bangkok"),
        ("Vietnam", "Hanoi"), ("Morocco", "Rabat"), ("Iran", "Tehran"),
    ],
    3: [
        ("Burkina Faso", "Ouagadougou"), ("Kazakhstan", "Astana"), ("Myanmar", "Naypyidaw"),
        ("Mongolia", "Ulaanbaatar"), ("Kyrgyzstan", "Bishkek"), ("Tajikistan", "Dushanbe"),
        ("Eritrea", "Asmara"), ("Suriname", "Paramaribo"), ("Slovenia", "Ljubljana"),
        ("Latvia", "Riga"), ("Bhutan", "Thimphu"), ("Laos", "Vientiane"),
        ("Malawi", "Lilongwe"), ("Botswana", "Gaborone"),
    ],
}
CAPITAL_ALTERNATIVES = {
    "Brasilia": ["Brasilia", "Brasília"],
    "Astana": ["Astana", "Nur-Sultan"],
    "Naypyidaw": ["Naypyidaw", "Nay Pyi Taw"],
}
ELEMENTS = {
    1: [("gold", "Au"), ("helium", "He"), ("iron", "Fe"), ("copper", "Cu")],
    2: [("silver", "Ag"), ("sodium", "Na"), ("potassium", "K")],
    3: [("lead", "Pb"), ("tin", "Sn"), ("mercury", "Hg"), ("tungsten", "W")],
}
LETTER_WORDS = {
    1: ["banana", "apple", "coffee", "pepper", "letter", "summer", "bubble", "cookie"],
    2: ["necessary", "committee", "parallel", "football", "blueberry", "umbrella", "coconut", "address"],
    3: ["strawberry", "bookkeeper", "successful", "accommodate", "millennium", "onomatopoeia", "unnecessarily", "mississippi"],
}
REVERSE_WORDS = {
    1: ["cat", "door", "moon", "tree", "fish", "lamp", "sand"],
    2: ["planet", "garden", "window", "purple", "castle", "bridge", "market"],
    3: ["mountain", "elephant", "umbrella", "chocolate", "skeleton", "telephone", "pineapple"],
}


def _case(family, level, index, user, checks, reference, system=None) -> EvalCase:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    for check in checks:
        validate_check(check)
    return EvalCase(
        id=f"gen_{family}_d{level}_{index:03d}",
        category=family,
        messages=messages,
        checks=checks,
        reference=reference,
    )


def _numeric(value):
    return [{"type": "numeric_answer", "value": value}]


def _fill(cases, seen, family, level, count, make, rng):
    """Draw random cases until there are `count` with a prompt that was not used before."""
    index = 0
    attempts = 0
    while index < count:
        attempts += 1
        if attempts > count * 300:
            raise RuntimeError(f"cannot find {count} different cases for {family} level {level}")
        user, checks, reference, system = make(rng)
        key = (system or "") + "|" + user
        if key in seen:
            continue
        seen.add(key)
        cases.append(_case(family, level, index, user, checks, reference, system))
        index += 1


def _add_fixed(cases, seen, family, level, index, user, checks, reference, system=None):
    key = (system or "") + "|" + user
    if key in seen:
        raise RuntimeError(f"duplicate fixed case: {user}")
    seen.add(key)
    cases.append(_case(family, level, index, user, checks, reference, system))


# ---------- Random families ----------


def _arith(level):
    def make(rng):
        if level == 1:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            if rng.random() < 0.5:
                expression, value = f"{a} + {b}", a + b
            else:
                a, b = max(a, b), min(a, b)
                expression, value = f"{a} - {b}", a - b
        elif level == 2:
            if rng.random() < 0.5:
                a, b = rng.randint(100, 999), rng.randint(2, 9)
                expression, value = f"{a} * {b}", a * b
            else:
                a, b, c = rng.randint(10, 99), rng.randint(2, 9), rng.randint(11, 49)
                expression, value = f"{a} + {b} * {c}", a + b * c
        else:
            if rng.random() < 0.5:
                a, b = rng.randint(12, 99), rng.randint(12, 99)
                expression, value = f"{a} * {b}", a * b
            else:
                a, b, c, d = rng.randint(5, 30), rng.randint(5, 30), rng.randint(2, 12), rng.randint(1, 20)
                expression, value = f"({a} + {b}) * {c} - {d}", (a + b) * c - d
        user = f"What is {expression}? Reply with only the final number."
        return user, _numeric(value), str(value), None

    return make


def _wordproblem(level):
    def make(rng):
        name, thing = rng.choice(NAMES), rng.choice(THINGS)
        if level == 1:
            a, b = rng.randint(5, 60), rng.randint(5, 60)
            text = f"{name} has {a} {thing} and buys {b} more. How many {thing} does {name} have now?"
            value = a + b
        elif level == 2:
            n, days, returned = rng.randint(6, 25), rng.randint(3, 9), rng.randint(2, 15)
            text = (
                f"A shop sells {n} {thing} per day for {days} days. "
                f"Then {returned} of them are returned. How many {thing} were kept?"
            )
            value = n * days - returned
        else:
            per_box, boxes = rng.randint(6, 24), rng.randint(3, 9)
            total = per_box * boxes
            used = rng.randint(2, total - 2)
            if (total - used) % 2:
                used -= 1  # the rest must split in two evenly
            text = (
                f"A box holds {per_box} {thing}. {name} buys {boxes} boxes, uses {used} {thing}, "
                f"and then gives away half of the rest. How many {thing} are left?"
            )
            value = (total - used) // 2
        return text + " Reply with only the number.", _numeric(value), str(value), None

    return make


def _sequence(level):
    def make(rng):
        if level == 1:
            start, step = rng.randint(1, 30), rng.randint(2, 9)
            terms = [start + i * step for i in range(5)]
        elif level == 2:
            start, ratio = rng.randint(2, 9), rng.choice([2, 3, 4])
            terms = [start * ratio**i for i in range(5)]
        elif rng.random() < 0.5:
            k = rng.randint(2, 9)
            terms = [(k + i) ** 2 for i in range(5)]
        else:
            terms = [rng.randint(1, 5), rng.randint(2, 7)]
            while len(terms) < 5:
                terms.append(terms[-1] + terms[-2])
        shown, answer = terms[:4], terms[4]
        user = f"What is the next number in the sequence {', '.join(map(str, shown))}? Reply with only the number."
        return user, _numeric(answer), str(answer), None

    return make


def _format_list(level):
    low, high = {1: (2, 3), 2: (4, 5), 3: (6, 8)}[level]

    def make(rng):
        n = rng.randint(low, high)
        things = rng.choice(sorted(LIST_ITEMS))
        user = f"List exactly {n} {things} as a bulleted list and nothing else."
        checks = [{"type": "bullet_count", "value": n}]
        if level == 3:
            user += " Use one word per item."
            checks.append({"type": "max_words", "value": 2 * n})
        reference = "\n".join(f"- {item}" for item in LIST_ITEMS[things][:n])
        return user, checks, reference, None

    return make


def _format_words(level):
    low, high = {1: (30, 40), 2: (15, 25), 3: (8, 12)}[level]

    def make(rng):
        n, topic = rng.randint(low, high), rng.choice(TOPICS)
        user = f"Describe {topic} in at most {n} words."
        checks = [{"type": "max_words", "value": n}, {"type": "no_refusal"}]
        return user, checks, f"{topic.capitalize()} is interesting.", None

    return make


def _format_json(level):
    def make(rng):
        count = {1: 2, 2: 3, 3: rng.choice([4, 5])}[level]
        keys = rng.sample(KEY_POOL, count)
        user = (
            f"Return a JSON object with the keys {', '.join(keys[:-1])} and {keys[-1]} "
            "for a fictional person. Output only the JSON."
        )
        checks = [{"type": "is_json"}, {"type": "json_keys", "keys": keys}]
        return user, checks, json.dumps({key: "x" for key in keys}), None

    return make


def _extract(level):
    def make(rng):
        first = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2024, 2027)}"
        if level == 1:
            text = f"The meeting is on {first} at {rng.randint(1, 11)} pm."
            user = f"What is the date in this sentence? Reply with only the date as written: '{text}'"
            checks = [{"type": "contains", "value": first}, {"type": "max_words", "value": 5}]
            return user, checks, first, None
        if level == 2:
            second = first
            while first in second or second in first:
                second = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2024, 2027)}"
            text = f"The event starts on {first} and ends on {second}."
            user = f"On what date does the event end? Reply with only the date as written: '{text}'"
            checks = [
                {"type": "contains", "value": second},
                {"type": "not_contains", "value": first},
                {"type": "max_words", "value": 5},
            ]
            return user, checks, second, None
        invoice, quantity = rng.randint(1000, 9999), rng.randint(2, 12)
        price, shipping = rng.randint(5, 90), rng.randint(3, 20)
        total = quantity * price + shipping
        text = (
            f"Invoice {invoice} dated {first}: {quantity} items at ${price} each plus "
            f"${shipping} shipping. Total due: ${total}."
        )
        user = f"What is the total due? Reply with only the number without the dollar sign: '{text}'"
        return user, _numeric(total), str(total), None

    return make


def _sort_numbers(level):
    count = {1: 4, 2: 6, 3: 9}[level]

    def make(rng):
        numbers = rng.sample(range(1, 100), count)
        ordered = sorted(numbers)
        user = (
            "Sort these numbers in ascending order. Reply with only the numbers separated by commas: "
            + ", ".join(map(str, numbers))
        )
        pattern = r"^\D*" + r"\s*,\s*".join(str(n) for n in ordered) + r"\D*$"
        return user, [{"type": "regex", "pattern": pattern}], ", ".join(map(str, ordered)), None

    return make


def _system_rule(level):
    def make(rng):
        if level == 1:
            word = rng.choice(["ZEBRA", "ROCKET", "MANGO", "PINEAPPLE"])
            country, capital = rng.choice(CAPITALS[1])
            system = f"Always end your reply with the word {word}."
            user = f"What is the capital of {country}?"
            checks = [
                {"type": "contains", "value": capital},
                {"type": "regex", "pattern": rf"{word}\W*$"},
            ]
            return user, checks, f"The capital of {country} is {capital}, {word}", system
        if level == 2:
            name = rng.choice(NAMES)
            checks = [{"type": "regex", "pattern": r"^[^a-z]+$"}, {"type": "contains", "value": "hello"}]
            return f"Say hello to {name}.", checks, f"HELLO {name.upper()}!", "Reply in all capital letters."
        word = rng.choice(["banana", "orange", "green", "window"])
        user = rng.choice(["Why is the sky blue?", "Why do we have seasons?", "How do birds fly?", "Why is the ocean salty?"])
        system = f"Answer in exactly one sentence and end your reply with the word {word}."
        checks = [
            {"type": "sentence_count", "value": 1},
            {"type": "regex", "pattern": rf"{word}\W*$"},
        ]
        return user, checks, f"Because of how light and air interact in nature, {word}", system

    return make


# ---------- Families made from fixed lists ----------


def _fixed_cases(cases, seen, rng):
    for level in (1, 2, 3):
        index = 0
        for country, capital in CAPITALS[level]:
            alternatives = CAPITAL_ALTERNATIVES.get(capital)
            check = (
                {"type": "contains_any", "values": alternatives}
                if alternatives
                else {"type": "contains", "value": capital}
            )
            user = f"What is the capital of {country}? Answer in one sentence."
            _add_fixed(cases, seen, "factual", level, index, user, [check], f"The capital of {country} is {capital}.")
            index += 1
        for element, symbol in ELEMENTS[level]:
            user = f"What is the chemical symbol for {element}? Reply with only the symbol."
            check = {"type": "regex", "pattern": rf"\b{symbol}\b"}
            _add_fixed(cases, seen, "factual", level, index, user, [check], symbol)
            index += 1

    for level in (1, 2, 3):
        for index, word in enumerate(LETTER_WORDS[level]):
            repeated = sorted(c for c in set(word) if word.count(c) >= 2)
            letter = rng.choice(repeated or sorted(set(word)))
            count = word.count(letter)
            user = f"How many times does the letter '{letter}' appear in the word '{word}'? Reply with only the number."
            _add_fixed(cases, seen, "count_letters", level, index, user, _numeric(count), str(count))

    for level in (1, 2, 3):
        for index, word in enumerate(REVERSE_WORDS[level]):
            reverse = word[::-1]
            user = f"Reverse the letters of the word '{word}'. Reply with only the reversed word."
            _add_fixed(cases, seen, "reverse_text", level, index, user, [{"type": "exact", "value": reverse}], reverse)


RANDOM_FAMILIES = [
    ("arith", 20, _arith),
    ("wordproblem", 12, _wordproblem),
    ("sequence", 10, _sequence),
    ("format_list", 10, _format_list),
    ("format_words", 10, _format_words),
    ("format_json", 8, _format_json),
    ("extract", 12, _extract),
    ("sort_numbers", 8, _sort_numbers),
    ("system_rule", 10, _system_rule),
]


def generate_cases(seed: int = SEED) -> list[EvalCase]:
    rng = random.Random(seed)
    seen: set = set()
    cases: list = []
    for family, per_level, builder in RANDOM_FAMILIES:
        for level in (1, 2, 3):
            _fill(cases, seen, family, level, per_level, builder(level), rng)
    _fixed_cases(cases, seen, rng)
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case ids")
    return cases


def write_cases(path, cases: list[EvalCase]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            row = {
                "id": c.id,
                "category": c.category,
                "messages": c.messages,
                "checks": c.checks,
                "reference": c.reference,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")