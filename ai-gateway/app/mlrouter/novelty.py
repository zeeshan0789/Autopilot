"""A novelty guard for the learned router.

The router was trained on prompts of a few task families and cannot judge a family it never
saw (leave one family out, AUC 0.46). So it only picks the cheap model for prompts that look
like its training prompts, measured by the cosine similarity of sentence embeddings (the
vectors must be normalized). Any other prompt goes to the fallback policy.
"""

import numpy as np


def nearest_similarity(queries, references) -> np.ndarray:
    """For every query, the highest cosine similarity to any reference."""
    queries = np.asarray(queries, dtype=float)
    references = np.asarray(references, dtype=float)
    return (queries @ references.T).max(axis=1)


def leave_family_out_similarity(embeddings, families):
    """(familiar, unfamiliar) similarity of every row, NaN where it does not exist.

    familiar: the nearest other row of the same family.
    unfamiliar: the nearest row of any other family.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    families = np.asarray(families)
    similarity = embeddings @ embeddings.T
    familiar = np.full(len(embeddings), np.nan)
    unfamiliar = np.full(len(embeddings), np.nan)
    for i in range(len(embeddings)):
        same = families == families[i]
        same[i] = False
        other = families != families[i]
        if same.any():
            familiar[i] = similarity[i, same].max()
        if other.any():
            unfamiliar[i] = similarity[i, other].max()
    return familiar, unfamiliar


def choose_similarity_threshold(unfamiliar_sims, abstain_share: float = 0.9) -> float:
    """The similarity below which `abstain_share` of the unfamiliar prompts fall."""
    values = np.asarray(unfamiliar_sims, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("no unfamiliar similarities to choose a threshold from")
    return float(np.quantile(values, abstain_share))


def retained_share(similarities, threshold: float) -> float | None:
    values = np.asarray(similarities, dtype=float)
    values = values[~np.isnan(values)]
    return float((values >= threshold).mean()) if len(values) else None


def guarded_picks(probs: dict, order: list[str], tau: float, similarities, threshold: float, fallback_picks: list[str]) -> list[str]:
    """The router's pick for familiar prompts, the fallback policy's pick for the rest."""
    from app.mlrouter.evaluate import choose_models  # imported here: evaluate does not need this module

    router_picks = choose_models(probs, order, tau)
    return [
        router_picks[i] if similarities[i] >= threshold else fallback_picks[i]
        for i in range(len(router_picks))
    ]