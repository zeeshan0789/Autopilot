import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Turns text into normalized embeddings. The model loads lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # Imported here so that importing the app does not load torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        # Newer sentence-transformers versions renamed this method
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def encode(self, text: str) -> np.ndarray:
        vector = self._load().encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)