import numpy as np
import pytest

from app.cache.embedder import Embedder


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