import numpy as np
import pytest

from app.mlrouter.novelty import (
    choose_similarity_threshold,
    guarded_picks,
    leave_family_out_similarity,
    nearest_similarity,
    retained_share,
)


def test_the_nearest_reference_decides():
    references = np.array([[1.0, 0.0], [0.0, 1.0]])
    queries = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert nearest_similarity(queries, references) == pytest.approx([1.0, 0.8])


def test_familiar_and_unfamiliar_similarity_per_row():
    embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]])
    families = ["a", "a", "b", "b"]
    familiar, unfamiliar = leave_family_out_similarity(embeddings, families)
    assert familiar == pytest.approx([0.8, 0.8, 0.8, 0.8])
    assert unfamiliar == pytest.approx([0.6, 0.96, 0.6, 0.96])


def test_a_family_with_one_prompt_has_no_familiar_similarity():
    familiar, unfamiliar = leave_family_out_similarity(np.eye(2), ["a", "b"])
    assert np.isnan(familiar).all()
    assert unfamiliar == pytest.approx([0.0, 0.0])


def test_the_threshold_is_a_quantile_of_the_unfamiliar_similarities():
    assert choose_similarity_threshold([0.5, 0.6, 0.7, 0.8, float("nan")], 0.5) == pytest.approx(0.65)
    assert choose_similarity_threshold([0.5, 0.9], 1.0) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        choose_similarity_threshold([float("nan")])


def test_retained_share_counts_prompts_at_or_above_the_threshold():
    assert retained_share([0.5, 0.7, 0.9, 0.9], 0.7) == 0.75
    assert retained_share([], 0.7) is None


def test_unfamiliar_prompts_go_to_the_fallback_policy():
    probs = {"cheap": np.array([0.99, 0.99, 0.2]), "strong": np.ones(3)}
    sims = np.array([0.95, 0.40, 0.95])
    picks = guarded_picks(probs, ["cheap", "strong"], 0.9, sims, 0.8, ["mid", "mid", "mid"])
    # row 0: familiar and confident -> cheap. Row 1: confident but unfamiliar -> fallback.
    # Row 2: familiar but not confident -> the router's own next choice (strong).
    assert picks == ["cheap", "mid", "strong"]