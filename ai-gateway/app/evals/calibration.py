"""Is the LLM judge any good? Compare its verdicts with hand labels.

Positive means "good". The judge is trusted only when it agrees with the human beyond chance
(Cohen's kappa), catches most of the answers the human called bad, and there are enough
labels. The thresholds are conventional choices, not laws, and with few labels the intervals
are wide: they are reported, not hidden.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.evals.labels import answer_sha
from app.metrics.cache_eval import wilson_interval

TRUST_MIN_KAPPA = 0.6
TRUST_MIN_SPECIFICITY = 0.7  # of the answers the human called bad, how many the judge also called bad
MIN_LABELS = 30
MIN_PER_CLASS = 5


@dataclass(frozen=True)
class Pair:
    human_good: bool
    judge_good: bool


def confusion(pairs: list[Pair]) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) with 'good' as the positive class and the human as the truth."""
    tp = sum(1 for p in pairs if p.human_good and p.judge_good)
    fp = sum(1 for p in pairs if not p.human_good and p.judge_good)
    fn = sum(1 for p in pairs if p.human_good and not p.judge_good)
    tn = sum(1 for p in pairs if not p.human_good and not p.judge_good)
    return tp, fp, fn, tn


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    observed = (tp + tn) / n
    expected = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
    if expected == 1.0:
        return None  # only one class everywhere, kappa is undefined
    return (observed - expected) / (1 - expected)


def bootstrap_kappa_interval(pairs: list[Pair], iterations: int = 2000, seed: int = 1):
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = rng.choices(pairs, k=len(pairs))
        kappa = cohens_kappa(*confusion(sample))
        if kappa is not None:
            values.append(kappa)
    if len(values) < max(50, iterations // 10):
        return None
    values.sort()
    return (values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])


@dataclass
class Calibration:
    judge_model: str
    rubric_version: str
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float | None
    accuracy_interval: tuple
    kappa: float | None
    kappa_interval: tuple | None
    human_good_rate: float | None
    judge_good_rate: float | None
    sensitivity: float | None  # of the answers the human called good, how many the judge also called good
    specificity: float | None
    unparsable: int
    trusted: bool
    reasons: list = field(default_factory=list)  # why the judge is not trusted
    created_at: str = ""


def calibrate(
    pairs: list[Pair],
    judge_model: str,
    rubric_version: str,
    unparsable: int = 0,
    iterations: int = 2000,
    seed: int = 1,
) -> Calibration:
    tp, fp, fn, tn = confusion(pairs)
    n = tp + fp + fn + tn
    human_good, human_bad = tp + fn, fp + tn
    kappa = cohens_kappa(tp, fp, fn, tn)
    specificity = tn / human_bad if human_bad else None

    reasons = []
    if n < MIN_LABELS:
        reasons.append(f"only {n} usable labels, at least {MIN_LABELS} are needed")
    if min(human_good, human_bad) < MIN_PER_CLASS:
        reasons.append(f"fewer than {MIN_PER_CLASS} labels in one class ({human_good} good, {human_bad} bad)")
    if kappa is None:
        reasons.append("kappa cannot be computed")
    elif kappa < TRUST_MIN_KAPPA:
        reasons.append(f"kappa {kappa:.2f} is below {TRUST_MIN_KAPPA}")
    if specificity is not None and specificity < TRUST_MIN_SPECIFICITY:
        reasons.append(f"the judge catches only {specificity * 100:.0f}% of the bad answers (needs {TRUST_MIN_SPECIFICITY * 100:.0f}%)")

    return Calibration(
        judge_model=judge_model,
        rubric_version=rubric_version,
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=(tp + tn) / n if n else None,
        accuracy_interval=wilson_interval(tp + tn, n),
        kappa=kappa,
        kappa_interval=bootstrap_kappa_interval(pairs, iterations, seed) if n else None,
        human_good_rate=human_good / n if n else None,
        judge_good_rate=(tp + fp) / n if n else None,
        sensitivity=tp / human_good if human_good else None,
        specificity=specificity,
        unparsable=unparsable,
        trusted=not reasons,
        reasons=reasons,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_calibration(path, calibration: Calibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def load_calibration(path) -> Calibration | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("accuracy_interval", "kappa_interval"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    return Calibration(**data)


def is_trusted(calibration: Calibration | None, judge_model: str, rubric_version: str) -> bool:
    """A calibration only counts for the exact judge model and rubric it was measured with."""
    return bool(
        calibration
        and calibration.trusted
        and calibration.judge_model == judge_model
        and calibration.rubric_version == rubric_version
    )


def collect_labeled(labels: list[dict], runs_by_name: dict, cases: list):
    """Match hand labels to the answers they were given for.

    Returns (items, stale, missing): items are (label, case, answer). A label is stale when
    the answer changed after it was labeled (a rerun), and missing when its run or case is gone.
    """
    by_id = {c.id: c for c in cases}
    latest = {}
    for label in labels:  # a later label for the same answer replaces an earlier one
        latest[(label["run"], label["id"], label["answer_sha"])] = label

    items, stale, missing = [], 0, 0
    for label in latest.values():
        run = runs_by_name.get(label["run"])
        case = by_id.get(label["id"])
        result = next((r for r in run.results if r.id == label["id"]), None) if run else None
        if case is None or result is None or result.status != "ok":
            missing += 1
        elif answer_sha(result.answer or "") != label["answer_sha"]:
            stale += 1
        else:
            items.append((label, case, result.answer or ""))
    return items, stale, missing