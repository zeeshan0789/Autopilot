"""Train and evaluate the learned router (LightGBM) with a novelty guard.

From ai-gateway/:
    python -m scripts.router_train                     # the model that gets saved
    python -m scripts.router_train --cv random         # threshold from random folds instead of leave one family out
    python -m scripts.router_train --embed-dims 16     # experiment: add semantic features (not saved)

The test cases and the hand written cases are only used at the end, never for training or for
choosing a threshold.
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np

from app.cache.embedder import Embedder
from app.core.policy import policy_router
from app.evals.cases import load_cases
from app.evals.compare import paired_bootstrap
from app.mlrouter.data import load_outcomes, parse_case_id, split_of
from app.mlrouter.evaluate import (
    TAU_GRID,
    apply_pca,
    brier,
    bootstrap_mean_ci,
    build_dataset,
    calibration_table,
    choose_models,
    fit_pca,
    group_folds,
    oracle_picks,
    out_of_fold,
    replay,
    roc_auc,
    rule_policy_picks,
    select_tau,
)
from app.mlrouter.features import FEATURE_NAMES
from app.mlrouter.model import RouterModel
from app.mlrouter.novelty import (
    choose_similarity_threshold,
    guarded_picks,
    leave_family_out_similarity,
    nearest_similarity,
    retained_share,
)
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"
CASES_FILE = BENCH / "router" / "cases.jsonl"
OUTCOMES_FILE = BENCH / "router" / "outcomes.jsonl"
HANDWRITTEN_FILE = BENCH / "eval_cases.jsonl"
MODEL_DIR = BASE_DIR / "models" / "router"
MODELS = ["qwen2.5:0.5b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # cheapest first
GUARDED = "ML router + novelty guard"


def short(model: str) -> str:
    return model.split("/")[-1]


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def money(value) -> str:
    return f"${value:.6f}"


def last_user_text(case) -> str:
    turns = [m["content"] for m in case.messages if m["role"] != "system"]
    return turns[-1] if turns else ""


def prompt_text(case) -> str:
    system = " ".join(m["content"] for m in case.messages if m["role"] == "system")
    return (system + "\n" + last_user_text(case)).strip()


def print_losses(dataset, probs, picks, models, baseline, limit=12) -> None:
    lost = [i for i, m in enumerate(picks) if dataset.passed[m][i] == 0]
    print(f"  Cases where the chosen model failed ({len(lost)}):")
    for i in lost[:limit]:
        case = dataset.cases[i]
        text = " ".join(last_user_text(case).split())[:70]
        fixed = "yes" if dataset.passed[baseline][i] else "no"
        print(f"    {case.id:<24}picked {short(picks[i]):<13} p({short(models[0])})={probs[models[0]][i]:.2f}  {short(baseline)} would pass: {fixed}  {text!r}")
    if len(lost) > limit:
        print(f"    ... and {len(lost) - limit} more")


def report_set(title, dataset, probs, tau, models, strong, baseline, extra=None, show_losses=False):
    n = len(dataset)
    print(f"\n{title} ({n} cases)")
    picks = {f"always {short(m)}": [m] * n for m in models}
    picks["rule policy (auto)"] = rule_policy_picks(dataset.cases, policy_router)
    picks["ML router"] = choose_models(probs, models, tau)
    for name, chosen in (extra or {}).items():
        picks[name] = chosen
    picks["perfect router"] = oracle_picks(dataset, models)
    replays = {name: replay(p, dataset) for name, p in picks.items()}
    strong_cost = replays[f"always {short(strong)}"].avg_cost

    print(f"  {'policy':<28}{'pass rate (95% interval)':<30}{'avg cost':>12}{'vs always ' + short(strong):>22}   picks " + "/".join(short(m)[:8] for m in models))
    for name, r in replays.items():
        low, high = bootstrap_mean_ci(r.passed)
        versus = f"{(r.avg_cost / strong_cost - 1) * 100:+.0f}%" if strong_cost > 0 else "n/a"
        shares = "/".join(f"{s * 100:.0f}%" for s in r.shares(models).values())
        print(f"  {name:<28}{f'{pct(r.pass_rate)} ({pct(low)} to {pct(high)})':<30}{money(r.avg_cost):>12}{versus:>22}   {shares}")

    for ml_name in [name for name in picks if name.startswith("ML router")]:
        ml = replays[ml_name]
        for other_name in (f"always {short(baseline)}", "rule policy (auto)"):
            other = replays[other_name]
            pass_low, pass_high = paired_bootstrap([float(a - b) for a, b in zip(ml.passed, other.passed)], 2000)
            cost_low, cost_high = paired_bootstrap([float(a - b) for a, b in zip(ml.cost, other.cost)], 2000)
            change = float(ml.passed.mean() - other.passed.mean())
            cost_change = ml.avg_cost - other.avg_cost
            relative = f" ({cost_change / other.avg_cost * 100:+.0f}%)" if other.avg_cost > 0 else ""
            print(
                f"  {ml_name} against {other_name}: pass rate {change * 100:+.1f} points (95% interval {pass_low * 100:+.1f} to {pass_high * 100:+.1f}), "
                f"cost {money(cost_change)} per request{relative} (interval {money(cost_low)} to {money(cost_high)})"
            )
    if show_losses:
        print_losses(dataset, probs, picks[GUARDED if GUARDED in picks else "ML router"], models, baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, help="cheapest first")
    parser.add_argument("--baseline", default="openai/gpt-oss-20b", help="the model whose pass rate must be matched")
    parser.add_argument("--tolerance", type=float, default=0.01, help="allowed drop of the pass rate against the baseline (0 to 1)")
    parser.add_argument("--cv", choices=["family", "random"], default="family", help="how the threshold is chosen")
    parser.add_argument("--folds", type=int, default=5, help="folds of the random CV")
    parser.add_argument("--abstain-share", type=float, default=0.9, help="share of unfamiliar training prompts the novelty guard must refuse")
    parser.add_argument("--embed-dims", type=int, default=0, help="add this many semantic (embedding PCA) features, 0 for none")
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

    embedder = Embedder()
    embeddings = {
        key: np.array([embedder.encode(prompt_text(c)) for c in dataset.cases], dtype=float)
        for key, dataset in (("train", train), ("test", test), ("hand", hand))
    }

    names = list(FEATURE_NAMES)
    if args.embed_dims > 0:
        mean, components = fit_pca(embeddings["train"], args.embed_dims)  # fit on the training cases only
        for key, dataset in (("train", train), ("test", test), ("hand", hand)):
            dataset.X = np.hstack([dataset.X, apply_pca(embeddings[key], mean, components)])
        names += [f"pc{i + 1}" for i in range(args.embed_dims)]
        print(f"semantic features: {args.embed_dims} principal components of the sentence embedding were added")

    cheapest, strong = models[0], models[-1]
    y = train.passed[cheapest]
    base_rate = float(y.mean())
    print(f"\n1. Cross validation on the training cases ({short(cheapest)} passes {pct(base_rate)} of them)")
    random_oof = out_of_fold(train, models, k=args.folds, seed=args.seed)
    families = [parse_case_id(c.id)[0] for c in train.cases]
    family_oof = out_of_fold(train, models, seed=args.seed, folds=group_folds(families))
    guess = brier(y, np.full(len(y), base_rate))
    print(f"   random folds ({args.folds}):       AUC {roc_auc(y, random_oof[cheapest]):.3f}, Brier {brier(y, random_oof[cheapest]):.3f}   (guessing {pct(base_rate)} every time: Brier {guess:.3f})")
    print(f"   leave one family out ({len(set(families))}): AUC {roc_auc(y, family_oof[cheapest]):.3f}, Brier {brier(y, family_oof[cheapest]):.3f}")
    print("   (random folds let the router see other prompts of the same template, leave one family out does not)")

    oof = family_oof if args.cv == "family" else random_oof
    tau = select_tau(oof, train, models, args.baseline, args.tolerance)
    if tau is None:
        tau = max(TAU_GRID)
        print(f"   WARNING: no threshold keeps the pass rate within {args.tolerance * 100:.0f} point of {short(args.baseline)}, using the strictest ({tau})")
    else:
        print(f"   threshold chosen with the {args.cv} CV: {tau} (the cheapest one that stays within {args.tolerance * 100:.0f} point of always {short(args.baseline)})")

    router = RouterModel(models, seed=args.seed).fit(train.X, train.passed, names)
    kinds = ", ".join(f"{short(m)} {'learned' if m in router.boosters else 'only its overall pass rate'}" for m in models)
    print(f"\n2. Final model trained on all {len(train)} training cases: {kinds}")

    # The novelty guard: how much does a prompt have to look like the training prompts?
    familiar, unfamiliar = leave_family_out_similarity(embeddings["train"], np.array(families))
    sim_threshold = choose_similarity_threshold(unfamiliar, args.abstain_share)
    test_sims = nearest_similarity(embeddings["test"], embeddings["train"])
    hand_sims = nearest_similarity(embeddings["hand"], embeddings["train"])
    print(f"\n3. Novelty guard (similarity threshold {sim_threshold:.3f}: {args.abstain_share * 100:.0f}% of the unfamiliar training prompts are below it)")
    print(f"   {'nearest training prompt of...':<56}{'median':>8}{'10th pct':>10}{'passes the guard':>18}")
    for label, sims in (
        ("training prompts, the same family (familiar)", familiar),
        ("training prompts, another family (unfamiliar)", unfamiliar),
        ("TEST prompts (same templates as training)", test_sims),
        ("HAND WRITTEN prompts", hand_sims),
    ):
        values = sims[~np.isnan(sims)]
        print(f"   {label:<56}{np.median(values):>8.3f}{np.percentile(values, 10):>10.3f}{pct(retained_share(sims, sim_threshold)):>18}")

    test_probs = router.predict(test.X)
    hand_probs = router.predict(hand.X)
    test_guarded = guarded_picks(test_probs, models, tau, test_sims, sim_threshold, rule_policy_picks(test.cases, policy_router))
    hand_guarded = guarded_picks(hand_probs, models, tau, hand_sims, sim_threshold, rule_policy_picks(hand.cases, policy_router))

    report_set("4. TEST cases (never used for training or for choosing a threshold)", test, test_probs, tau, models, strong, args.baseline, {GUARDED: test_guarded}, show_losses=True)
    yt = test.passed[cheapest]
    print(f"   {short(cheapest)} on the test cases: AUC {roc_auc(yt, test_probs[cheapest]):.3f}, Brier {brier(yt, test_probs[cheapest]):.3f}")

    report_set("5. HAND WRITTEN cases (a different style of prompt)", hand, hand_probs, tau, models, strong, args.baseline, {GUARDED: hand_guarded}, show_losses=True)
    auc = roc_auc(hand.passed[cheapest], hand_probs[cheapest])
    print(f"   {short(cheapest)} on the hand written cases: AUC {'n/a' if auc is None else f'{auc:.3f}'}")

    if cheapest in router.boosters:
        gains = router.boosters[cheapest].feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        print(f"\nWhat the {short(cheapest)} predictor looks at (share of the total gain):")
        for name, gain in sorted(zip(names, gains), key=lambda t: -t[1])[:8]:
            print(f"   {name:<20}{gain / total * 100:5.1f}%")

    if args.embed_dims > 0:
        print("\nExperiment with semantic features: the model was not saved.")
    elif not args.no_save:
        ml_test = replay(test_guarded, test)
        ml_hand = replay(hand_guarded, hand)
        meta = {
            "version": "v3",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tau": tau,
            "sim_threshold": sim_threshold,
            "abstain_share": args.abstain_share,
            "tolerance": args.tolerance,
            "baseline": args.baseline,
            "cv": args.cv,
            "trained_on": len(train),
            "test": {"cases": len(test), "pass_rate": ml_test.pass_rate, "avg_cost_usd": ml_test.avg_cost},
            "handwritten": {"cases": len(hand), "pass_rate": ml_hand.pass_rate, "avg_cost_usd": ml_hand.avg_cost},
        }
        router.save(MODEL_DIR, meta)
        np.save(MODEL_DIR / "train_embeddings.npy", embeddings["train"].astype(np.float32))
        print(f"\nsaved the evaluated model and the training embeddings to {MODEL_DIR}")


if __name__ == "__main__":
    main()