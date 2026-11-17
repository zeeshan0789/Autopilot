"""The fixed benchmark workload: unique prompts, exact repeats and lightly reworded repeats.

The prompts come from the generated eval cases, so every request has deterministic checks and a
wrong answer (for example one served from the cache for another prompt) is caught. The
rewording is mechanical (a prefix or a suffix), which is a weak stand in for real paraphrases.
"""

import copy
import random
from dataclasses import dataclass

PARAPHRASE_VARIANTS = (
    lambda text: "Please answer this: " + text,
    lambda text: text + " Thanks!",
    lambda text: "Hi! " + text,
)


@dataclass(frozen=True)
class WorkItem:
    id: str  # w000, w001... in request order
    kind: str  # unique, repeat or paraphrase
    base_id: str  # the eval case this request comes from
    messages: list
    checks: list


def _reword(messages: list, variant: int) -> list:
    reworded = copy.deepcopy(messages)
    last = max(i for i, m in enumerate(reworded) if m["role"] == "user")
    change = PARAPHRASE_VARIANTS[variant % len(PARAPHRASE_VARIANTS)]
    reworded[last]["content"] = change(reworded[last]["content"])
    return reworded


def build_workload(cases, unique_per_family: int = 5, repeats: int = 50, paraphrases: int = 40, seed: int = 11) -> list[WorkItem]:
    """The same arguments always give the same workload. A prompt is always asked for the
    first time before it is repeated or reworded."""
    rng = random.Random(seed)
    by_family: dict = {}
    for case in cases:
        by_family.setdefault(case.category, []).append(case)

    uniques = []
    for family in sorted(by_family):
        pool = by_family[family]
        uniques.extend(rng.sample(pool, min(unique_per_family, len(pool))))

    events = []  # (time, order, kind, case, variant)
    first_time = {}
    for case in uniques:
        first_time[case.id] = rng.random() * 0.6
        events.append((first_time[case.id], 0, "unique", case, None))
    for _ in range(repeats):
        base = rng.choice(uniques)
        events.append((rng.uniform(first_time[base.id], 1.0), 1, "repeat", base, None))
    for number in range(paraphrases):
        base = rng.choice(uniques)
        events.append((rng.uniform(first_time[base.id], 1.0), 2, "paraphrase", base, number))
    events.sort(key=lambda event: (event[0], event[1]))

    items = []
    for index, (_, _, kind, case, variant) in enumerate(events):
        messages = _reword(case.messages, variant) if kind == "paraphrase" else copy.deepcopy(case.messages)
        items.append(WorkItem(f"w{index:03d}", kind, case.id, messages, copy.deepcopy(case.checks)))
    return items