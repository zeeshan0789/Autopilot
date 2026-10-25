import numpy as np
import pytest

from app.cache.embedder import Embedder

class OldNameModel:
    def get_sentence_embedding_dimension(self):
        return 384


class NewNameModel:
    def get_embedding_dimension(self):
        return 384


def test_dimension_works_with_old_and_new_sentence_transformers_names():
    for model in (OldNameModel(), NewNameModel()):
        embedder = Embedder()
        embedder._model = model  # skip loading the real model
        assert embedder.dimension == 384


@pytest.mark.slow
def test_real_model_ranks_paraphrase_above_unrelated():
    embedder = Embedder()
    try:
        a = embedder.encode("What is the capital of France?")
    except Exception as e:
        pytest.skip(f"Embedding model not available ({type(e).__name__})")

    paraphrase = embedder.encode("Which city is the capital of France?")
    unrelated = embedder.encode("How do I bake a chocolate cake?")

    assert a.shape == (384,)
    assert a.dtype == np.float32
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-3  # normalized
    assert float(np.dot(a, paraphrase)) > float(np.dot(a, unrelated))