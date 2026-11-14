"""Train and evaluate the learned router (LightGBM).

From ai-gateway/:   python -m scripts.router_train
The test cases and the hand written cases are only used at the very end, once.
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np

from app.core.policy import policy_router
from app.evals.cases import load_cases
from app.evals.compare import paired_bootstrap
from app.mlrouter.data import load_outcomes, split_of
from app.mlrouter.evaluate import (
    TAU_GRID,
    brier,
    bootstrap_mean_ci,
    build_dataset,
    calibration_table,
    choose_models,
    oracle_picks,
    out_of_fold,
    replay,
    roc_auc,
    rule_policy_picks,
    select_tau,
)
from app.mlrouter.features import FEATURE_NAMES
from app.mlrouter.model import RouterModel
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"
CASES_FILE = BENCH / "router" / "cases.jsonl"
OUTCOMES_FILE = BENCH / "router" / "outcomes.jsonl"
HANDWRITTEN_FILE = BENCH / "eval_cases.jsonl"
MODEL_DIR = BASE_DIR / "models" / "router"
MODELS = ["qwen2.5:0.5b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # cheapest first


def short(model: str) -> str:
    return model.split("/")[-1]


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def money(value) -> str:
    return f"${value:.6f}"


def report_set(title, dataset, probs, tau, models, strong, baseline):
    n = len(dataset)
    print(f"\n{title} ({n} cases)")
    picks = {f"always {short(m)}": [m] * n for m in models}
    picks["rule policy (auto)"] = rule_policy_picks(dataset.cases, policy_router)
    picks["ML router"] = choose_models(probs, models, tau)
    picks["perfect router"] = oracle_picks(dataset, models)
    replays = {name: replay(p, dataset) for name, p in picks.items()}
    strong_cost = replays[f"always {short(strong)}"].avg_cost

    print(f"  {'policy':<22}{'pass rate (95% interval)':<30}{'avg cost':>12}{'vs always ' + short(strong):>22}   picks " + "/".join(short(m)[:8] for m in models))
    for name, r in replays.items():
        low, high = bootstrap_mean_ci(r.passed)
        versus = f"{(r.avg_cost / strong_cost - 1) * 100:+.0f}%" if strong_cost > 0 else "n/a"
        shares = "/".join(f"{s * 100:.0f}%" for s in r.shares(models).values())
        print(f"  {name:<22}{f'{pct(r.pass_rate)} ({pct(low)} to {pct(high)})':<30}{money(r.avg_cost):>12}{versus:>22}   {shares}")

    ml, base = replays["ML router"], replays[f"always {short(baseline)}"]
    for other_name, other in ((f"always {short(baseline)}", base), ("rule policy (auto)", replays["rule policy (auto)"])):
        pass_low, pass_high = paired_bootstrap([float(a - b) for a, b in zip(ml.passed, other.passed)], 2000)
        cost_low, cost_high = paired_bootstrap([float(a - b) for a, b in zip(ml.cost, other.cost)], 2000)
        change = float(ml.passed.mean() - other.passed.mean())
        cost_change = ml.avg_cost - other.avg_cost
        relative = f" ({cost_change / other.avg_cost * 100:+.0f}%)" if other.avg_cost > 0 else ""
        print(
            f"  ML router against {other_name}: pass rate {change * 100:+.1f} points (95% interval {pass_low * 100:+.1f} to {pass_high * 100:+.1f}), "
            f"cost {money(cost_change)} per request{relative} (interval {money(cost_low)} to {money(cost_high)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS, help="cheapest first")
    parser.add_argument("--baseline", default="openai/gpt-oss-20b", help="the model whose pass rate must be matched")
    parser.add_argument("--tolerance", type=float, default=0.01, help="allowed drop of the pass rate against the baseline (0 to 1)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    models = args.models
    outcomes = load_outcomes(OUTCOMES_FILE)
    generated = load_cases(CASES_FILE)
    train = build_dataset([c for c in generated if split_of(c.id) == "train"], outcomes, models)
    test = build_dataset([c for c in generated if split_of(c.id) == "test"], outcomes, models)
    hand = build_dataset(load_cases(HANDWRITTEN_FILE), outcomes, models)
    print(f"cases answered by every model: train {len(train)}, test {len(test)}, hand written {len(hand)}")
    if len(train) < 100:
        sys.exit("too few complete training cases, finish: python -m scripts.router_collect collect ...")

    cheapest = models[0]
    print(f"\n1. Cross validation on the training cases ({args.folds} folds, every case is predicted by a model that never saw it)")
    oof = out_of_fold(train, models, k=args.folds, seed=args.seed)
    y = train.passed[cheapest]
    base_rate = float(y.mean())
    print(f"   {short(cheapest)} passes {pct(base_rate)} of the training cases")
    print(f"   its predicted pass probability: AUC {roc_auc(y, oof[cheapest]):.3f} (0.5 is chance), "
          f"Brier {brier(y, oof[cheapest]):.3f} (just guessing {pct(base_rate)} every time: {brier(y, np.full(len(y), base_rate)):.3f})")
    tau = select_tau(oof, train, models, args.baseline, args.tolerance)
    if tau is None:
        tau = max(TAU_GRID)
        print(f"   WARNING: no threshold keeps the pass rate within {args.tolerance * 100:.0f} point of {short(args.baseline)}, using the strictest ({tau})")
    else:
        print(f"   chosen threshold: {tau} (the cheapest one that stays within {args.tolerance * 100:.0f} point of always {short(args.baseline)})")

    strong = models[-1]
    report_set("2. Training cases, cross validated (the threshold was chosen on these, so this is optimistic)", train, oof, tau, models, strong, args.baseline)

    router = RouterModel(models, seed=args.seed).fit(train.X, train.passed, FEATURE_NAMES)
    kinds = ", ".join(f"{short(m)} {'learned' if m in router.boosters else 'only its overall pass rate'}" for m in models)
    print(f"\nFinal model trained on all {len(train)} training cases: {kinds}")

    test_probs = router.predict(test.X)
    report_set("3. TEST cases (never used for training or for choosing the threshold)", test, test_probs, tau, models, strong, args.baseline)
    yt = test.passed[cheapest]
    print(f"   {short(cheapest)} on the test cases: AUC {roc_auc(yt, test_probs[cheapest]):.3f}, Brier {brier(yt, test_probs[cheapest]):.3f}")
    print("   predicted pass probability against what really happened (test):")
    for low, high, count, actual in calibration_table(yt, test_probs[cheapest]):
        print(f"     {low:.1f} to {high:.1f}: {count:>3} cases, really passed {pct(actual)}")

    hand_probs = router.predict(hand.X)
    report_set("4. HAND WRITTEN cases (a different style of prompt, the router never saw anything like them)", hand, hand_probs, tau, models, strong, args.baseline)
    yh = hand.passed[cheapest]
    auc = roc_auc(yh, hand_probs[cheapest])
    print(f"   {short(cheapest)} on the hand written cases: AUC {'n/a' if auc is None else f'{auc:.3f}'}")

    if cheapest in router.boosters:
        gains = router.boosters[cheapest].feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        top = sorted(zip(FEATURE_NAMES, gains), key=lambda t: -t[1])[:8]
        print(f"\nWhat the {short(cheapest)} predictor looks at (share of the total gain):")
        for name, gain in top:
            print(f"   {name:<20}{gain / total * 100:5.1f}%")

    if not args.no_save:
        ml = replay(choose_models(test_probs, models, tau), test)
        meta = {
            "version": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tau": tau,
            "tolerance": args.tolerance,
            "baseline": args.baseline,
            "trained_on": len(train),
            "test": {"cases": len(test), "pass_rate": ml.pass_rate, "avg_cost_usd": ml.avg_cost},
        }
        router.save(MODEL_DIR, meta)
        print(f"\nsaved the evaluated model to {MODEL_DIR}")


if __name__ == "__main__":
    main()