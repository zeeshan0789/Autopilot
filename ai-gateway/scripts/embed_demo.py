import time

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# (text A, text B, what we expect)
PAIRS = [
    ("What is the capital of France?", "Which city is the capital of France?", "paraphrase: same answer"),
    ("How do I reset my password?", "I forgot my password, how can I change it?", "paraphrase: same answer"),
    ("What was Apple revenue in 2024?", "What was Apple revenue in 2025?", "different year: DIFFERENT answer"),
    ("What is the capital of France?", "What is the capital of Germany?", "different country: DIFFERENT answer"),
    ("Convert 100 USD to PKR", "Convert 200 USD to PKR", "different number: DIFFERENT answer"),
    ("What is the capital of France?", "How do I bake a chocolate cake?", "unrelated"),
]


def main():
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    print(
        f"Model loaded in {time.perf_counter() - start:.1f}s, "
        f"embedding size {model.get_sentence_embedding_dimension()}"
    )

    texts = [text for a, b, _ in PAIRS for text in (a, b)]
    start = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Encoded {len(texts)} texts in {(time.perf_counter() - start) * 1000:.0f} ms")
    print()

    for i, (a, b, note) in enumerate(PAIRS):
        score = float(util.cos_sim(embeddings[2 * i], embeddings[2 * i + 1]))
        print(f"{score:.3f}  [{note}]")
        print(f"        A: {a}")
        print(f"        B: {b}")


main()