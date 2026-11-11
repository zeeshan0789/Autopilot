"""Hand labels: a person decides whether an answer is good. They are the ground truth that
the LLM judge is calibrated against (next step), so the labeling is blind: the model that
produced an answer is not shown."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.evals.cases import EvalCase
from app.evals.runner import EvalRun


@dataclass(frozen=True)
class LabelItem:
    run: str
    case: EvalCase
    answer: str

    @property
    def answer_sha(self) -> str:
        return answer_sha(self.answer)


def answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]


def collect_items(cases: list[EvalCase], runs: list[EvalRun]) -> list[LabelItem]:
    by_id = {c.id: c for c in cases}
    items = []
    for run in runs:
        for result in run.results:
            case = by_id.get(result.id)
            if case is not None and result.status == "ok":
                items.append(LabelItem(run.name, case, result.answer or ""))
    return items


def load_labels(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(json.loads(line))
    return labels


def pick_items(items: list[LabelItem], labels: list[dict], count: int, seed: int = 1) -> list[LabelItem]:
    """Random items that are not labeled yet (same seed, same order)."""
    done = {(label["run"], label["id"], label["answer_sha"]) for label in labels}
    todo = [i for i in items if (i.run, i.case.id, i.answer_sha) not in done]
    random.Random(seed).shuffle(todo)
    return todo[:count]


def append_label(path, item: LabelItem, label: str) -> None:
    if label not in ("good", "bad"):
        raise ValueError("label must be 'good' or 'bad'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run": item.run, "id": item.case.id, "label": label, "answer_sha": item.answer_sha}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")