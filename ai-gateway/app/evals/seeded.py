"""Seeded bad changes: does the regression gate catch changes that are known to be bad?

A managed prompt gets deliberately broken versions, plus a few harmless ones as controls.
Each version is evaluated on the same cases and compared with the baseline by the same gate
that judges any real change. Counting how many bad changes are caught, and how many harmless
ones are wrongly flagged, measures the gate itself. The bad changes were designed by the
developer: this shows what the gate can catch, not how real regressions look.
"""

from dataclasses import dataclass

from app.evals.calibration import Calibration
from app.evals.compare import INCONCLUSIVE_DROP, NOT_ENOUGH_DATA, REGRESSION, Comparison

PROMPT_NAME = "seeded_eval"

BASELINE_TEXT = (
    "You are a helpful assistant. Answer accurately and follow the user's instructions exactly."
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # baseline, baseline_repeat, bad, control
    system_prompt: str
    description: str


VARIANTS = [
    Variant("baseline", "baseline", BASELINE_TEXT, "the reference prompt"),
    Variant("baseline_repeat", "baseline_repeat", BASELINE_TEXT, "the same prompt run a second time (run to run noise)"),
    Variant("one_word", "bad", "Answer every question with a single word, whatever is asked.", "answers with one word"),
    Variant("always_french", "bad", "Always answer in French only, whatever language the user writes in.", "always answers in French"),
    Variant(
        "refuses",
        "bad",
        "You cannot help with requests. Politely decline every request and explain that you are unable to assist.",
        "declines everything",
    ),
    Variant(
        "verbose",
        "bad",
        "Always write long, detailed answers of at least 150 words with extra background, whatever the user asks.",
        "far too long answers",
    ),
    Variant(
        "ignores_format",
        "bad",
        "Ignore any instructions about format, length or style. Always answer in a long flowing paragraph.",
        "ignores format instructions",
    ),
    Variant("no_digits", "bad", "Never use digits in your answers. Write every number in words.", "writes numbers as words"),
    Variant(
        "polite",
        "control",
        "You are a helpful and polite assistant. Answer accurately and follow the user's instructions exactly.",
        "harmless wording change",
    ),
    Variant(
        "careful",
        "control",
        "You are a careful assistant. Think before you answer, check your work, and follow the user's instructions exactly.",
        "harmless wording change",
    ),
    Variant(
        "friendly",
        "control",
        "You are a friendly, warm assistant. Answer accurately and follow the user's instructions exactly.",
        "harmless wording change",
    ),
]


def run_name(variant: Variant) -> str:
    return f"seeded_{variant.name}"


def unique_texts() -> list[str]:
    """System prompts in a stable order, without repeats (baseline and its repeat share one)."""
    seen, texts = set(), []
    for v in VARIANTS:
        if v.system_prompt not in seen:
            seen.add(v.system_prompt)
            texts.append(v.system_prompt)
    return texts


def pick_cases(cases: list, per_category: int) -> list:
    """The first N cases of every category (0 keeps all of them), in file order."""
    if not per_category:
        return list(cases)
    counts: dict = {}
    picked = []
    for case in cases:
        if counts.get(case.category, 0) < per_category:
            picked.append(case)
            counts[case.category] = counts.get(case.category, 0) + 1
    return picked


def classify(kind: str, verdict: str) -> str:
    if verdict == NOT_ENOUGH_DATA:
        return "not_evaluated"
    if kind == "bad":
        return {REGRESSION: "caught", INCONCLUSIVE_DROP: "warned"}.get(verdict, "missed")
    return {REGRESSION: "false_block", INCONCLUSIVE_DROP: "false_warning"}.get(verdict, "fine")


@dataclass(frozen=True)
class SeededRow:
    variant: str
    kind: str
    verdict: str
    outcome: str
    n_paired: int
    mean_diff: float | None
    ci_low: float | None
    ci_high: float | None


def build_row(variant: Variant, comparison: Comparison) -> SeededRow:
    return SeededRow(
        variant=variant.name,
        kind=variant.kind,
        verdict=comparison.verdict,
        outcome=classify(variant.kind, comparison.verdict),
        n_paired=comparison.n_paired,
        mean_diff=comparison.mean_diff,
        ci_low=comparison.ci_low,
        ci_high=comparison.ci_high,
    )


def summarize(rows: list[SeededRow]) -> dict:
    bad = [r for r in rows if r.kind == "bad" and r.outcome != "not_evaluated"]
    controls = [r for r in rows if r.kind in ("control", "baseline_repeat") and r.outcome != "not_evaluated"]
    return {
        "bad_total": len(bad),
        "bad_caught": sum(1 for r in bad if r.outcome == "caught"),
        "bad_warned": sum(1 for r in bad if r.outcome == "warned"),
        "bad_missed": sum(1 for r in bad if r.outcome == "missed"),
        "control_total": len(controls),
        "false_blocks": sum(1 for r in controls if r.outcome == "false_block"),
        "false_warnings": sum(1 for r in controls if r.outcome == "false_warning"),
        "not_evaluated": sum(1 for r in rows if r.outcome == "not_evaluated"),
    }


def _points(value) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}"


def readme_section(rows: list[SeededRow], calibration: Calibration | None) -> str:
    s = summarize(rows)
    cases = max((r.n_paired for r in rows), default=0)
    lines = [
        "### Evaluation results",
        "",
        f"**Regression gate on seeded bad changes.** A managed prompt got {s['bad_total']} deliberately broken versions and "
        f"{s['control_total']} harmless ones (a second run of the baseline and small wording changes). Each version was "
        f"evaluated on {cases} cases (the cache off, one model, deterministic checks only) and compared with the baseline.",
        "",
        f"- Broken versions blocked: **{s['bad_caught']} of {s['bad_total']}**. Flagged as an inconclusive drop: {s['bad_warned']}. Missed: {s['bad_missed']}.",
        f"- Harmless versions wrongly blocked: **{s['false_blocks']} of {s['control_total']}**. Wrongly flagged as an inconclusive drop: {s['false_warnings']}.",
        "",
        "| Change | Kind | Verdict | Average change (points) | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        interval = "n/a" if r.ci_low is None else f"{_points(r.ci_low)} to {_points(r.ci_high)}"
        lines.append(f"| {r.variant} | {r.kind} | {r.verdict} | {_points(r.mean_diff)} | {interval} |")
    lines += [
        "",
        "The broken versions were designed by me, so this shows what the gate can catch, not how real regressions look. "
        "Each comparison has few cases, so small drops cannot be detected (every comparison prints the smallest drop it can detect).",
        "",
    ]

    if calibration is None:
        lines.append("**LLM judge.** It has not been calibrated, so it is not used in any score.")
    else:
        parts = [f"agreement {calibration.accuracy * 100:.0f}%" if calibration.accuracy is not None else "agreement n/a"]
        if calibration.kappa is not None:
            kappa = f"Cohen's kappa {calibration.kappa:.2f}"
            if calibration.kappa_interval:
                kappa += f" (95% interval {calibration.kappa_interval[0]:.2f} to {calibration.kappa_interval[1]:.2f})"
            parts.append(kappa)
        if calibration.specificity is not None:
            parts.append(f"it caught {calibration.specificity * 100:.0f}% of the answers I called bad")
        status = "trusted" if calibration.trusted else "NOT trusted (" + "; ".join(calibration.reasons) + ")"
        lines.append(
            f"**LLM judge.** {calibration.judge_model} (rubric {calibration.rubric_version}) was compared with "
            f"{calibration.n} answers that I labeled by hand: {', '.join(parts)}. It is {status}. "
            "A judge only adds to a score when it is trusted."
        )
    return "\n".join(lines)